"""Checkpointed physical Metropolis draws for the ordinary-depth sampled-heating model."""

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import const, settings, units
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.receiver_heating import ReceiverHeatingLightCurve
from scripts import sample_reflected_binary as sampler
from scripts.reflected_trial import evaluate_depth, fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_quadratic_source import assess


ENGINE = "sampled-receiver-heat"


def target(fit):
    root = sampler.ROOT
    inventory = json.loads((root / "docs/ada_validation/background-m030-source-hashes.json").read_text())
    paths = {p for p in inventory if p.startswith("src/elisa/")}
    paths.update(
        (
            "src/elisa/experimental/receiver_heating.py",
            "scripts/sample_receiver_heating.py",
            "scripts/validate_quadratic_source.py",
        )
    )
    return dict(
        engine=ENGINE,
        code=sampler.digest({p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}),
        original_fit=sampler.context(fit),
        rule=asdict(CurvedEdgeRule(3, 4, slice_order=12, junction_depth=4)),
    )


def density(fit, vector, backend, parts):
    profiles = [p for part in parts for p in part["profiles"]]
    sampler.validate_physics(fit, vector, profiles, backend)
    constants = dict(G_SI=const.G, solar_mass_kg=float(units.solMass.to(units.kg)))
    for p in profiles:
        rule = asdict(CurvedEdgeRule(3, p["depth"], slice_order=12, junction_depth=4))
        if p["physical_constants"] != constants:
            raise ValueError("Physical constants differ from the fitted model")
        for meta in p["samples"][0]["rows"][0]["finite_metadata"].values():
            if any(meta["rule"].get(k) != v for k, v in rule.items()):
                raise ValueError("Unexpected eclipse rule in sampled-heating chain")
    report = assess(parts, engine=ENGINE)
    if report["usable_log_posterior"] is None:
        raise ResolutionNotQualifiedError("Sampled-heating proposal exceeded the resolution budget", report)
    if not np.isfinite(report["usable_log_posterior"]):
        raise ArithmeticError("Nonfinite sampled-heating density")
    return report


def synthesize(fit, vector, backend, depth):
    def observer(bands, rule, *, device):
        return ReceiverHeatingLightCurve(bands, replace(rule, slice_order=12, junction_depth=4), device=device)

    result = evaluate_depth(fit, vector, depth, backend, observer_type=observer)
    for p in result["profiles"]:
        rule = p["samples"][0]["rows"][0]["finite_metadata"]["curvature_field3"]["rule"]
        p.update(slice_order=rule["slice_order"], junction_depth=rule["junction_depth"])
    return result


def initialize(args, fit, truth, identity):
    if args.checkpoint.exists():
        raise FileExistsError("Starting checkpoint already exists")
    if not np.isfinite(args.scale) or args.scale <= 0:
        raise ValueError("Proposal scale must be finite and positive")
    parts = [json.loads(p.read_bytes()) for p in args.initial]
    report = density(fit, truth, args.backend, parts)
    sampler.write_record(
        args.checkpoint,
        dict(
            step=0,
            target=identity,
            backend=args.backend,
            seed=args.seed,
            scales=[args.scale] * len(truth),
            rng=np.random.PCG64(args.seed).state,
            position=truth.tolist(),
            density=report["usable_log_posterior"],
            initial=report,
            initial_inputs=[sampler.digest(p["profiles"]) for p in parts],
            events=[],
        ),
    )


def checked_part(path, key, depth):
    part = sampler.read_record(path)
    if part["key"] != key or part["depth"] != depth or any(p["depth"] != depth for p in part["profiles"]):
        raise ValueError("Saved synthesis belongs to a different trial or depth")
    return part


def step(args, fit, state):
    if args.step != state["step"] + 1:
        raise ValueError("Step does not follow the saved chain generation")
    vector, log_u, next_rng, key = sampler.trial(state)
    prior = fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas)
    if not (np.isfinite(prior) or prior == -np.inf):
        raise ArithmeticError("Invalid fit prior")
    backend = state["backend"]
    if args.command == "evaluate":
        if prior == -np.inf:
            raise ValueError("Outside the prior. Advance the chain directly.")
        if backend == "gpu" and not args.verified_bounded_gpu_window:
            raise ValueError("Forward GPU evaluation requires a verified bounded unit")
        path = sampler.part_path(args.checkpoint.parent, args.step, args.depth)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                checked_part(path, key, args.depth)
            else:
                result = synthesize(fit, vector, backend, args.depth)
                sampler.write_record(path, dict(key=key, depth=args.depth, **result))
        return
    report, inputs, accepted = None, [], False
    if np.isfinite(prior):
        parts = [checked_part(sampler.part_path(args.checkpoint.parent, args.step, d), key, d) for d in (3, 4)]
        report = density(fit, vector, backend, parts)
        inputs = [sampler.digest(p) for p in parts]
        accepted = log_u < min(0.0, report["usable_log_posterior"] - state["density"])
        if accepted:
            state.update(position=vector.tolist(), density=report["usable_log_posterior"])
    state["events"].append(
        dict(
            step=args.step,
            proposed=vector.tolist(),
            log_u=log_u,
            accepted=accepted,
            selected=state["position"],
            selected_density=state["density"],
            report=report,
            inputs=inputs,
        )
    )
    state.update(step=args.step, rng=next_rng)
    sampler.write_record(args.checkpoint, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "evaluate", "advance"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial", type=Path, nargs=2)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--scale", type=float, default=0.0006)
    parser.add_argument("--backend", choices=("reference", "cpu", "gpu"), default="reference")
    parser.add_argument("--step", type=int)
    parser.add_argument("--depth", type=int, choices=(3, 4))
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    args = parser.parse_args()
    if args.command == "init" and not args.initial:
        parser.error("init needs both complete starting depth results")
    if args.command != "init" and (args.step is None or args.step < 1):
        parser.error("supply the next positive step")
    if args.command == "evaluate" and args.depth is None:
        parser.error("evaluate needs an eclipse depth")
    args.checkpoint.parent.mkdir(exist_ok=True, parents=True)
    with (
        args.checkpoint.with_suffix(".lock").open("a") as lock,
        table_settings(),
        settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2),
    ):
        fcntl.flock(lock, fcntl.LOCK_SH if args.command == "evaluate" else fcntl.LOCK_EX)
        fit, truth = fixture()
        identity = target(fit)
        if args.command == "init":
            initialize(args, fit, truth, identity)
        else:
            state = sampler.read_record(args.checkpoint)
            if sampler.digest(state["target"]) != sampler.digest(identity):
                raise ValueError("Target data, model or software changed since initialization")
            step(args, fit, state)


if __name__ == "__main__":
    main()
