"""Replay a checkpointed physical draw with sixteen complete reflected-curve resolutions."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from elisa import const, settings, units
from elisa.analytics.params import parameters
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts import sample_reflected_binary as sampler
from scripts.check_interior_resolution import compare, grid
from scripts.receiver_phase_parts import evaluate_block, join
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import physical


def check_target(state, fit):
    inventory = json.loads((sampler.ROOT / "docs/ada_validation/background-m039-source-hashes.json").read_bytes())
    sources = {k: hashlib.sha256((sampler.ROOT / k).read_bytes()).hexdigest()
               for k in inventory if k.startswith(("src/elisa/", "scripts/"))}
    expected = dict(engine="secondary-node-reemission", source=sampler.digest(sources), original_fit=sampler.context(fit),
                    field_levels=[2, 3], donor_levels=[1, 2], eclipse_depths=[3, 4], selective_shadow=True,
                    receiver_points=3, reflections=2, slice_order=12, junction_depth=4, integration_level=3,
                    interiors=["symmetric-degree-4", "duffy-3x3"], distance_limit=0.1, likelihood_limit=0.01)
    if sampler.digest(expected) != sampler.digest(state["target"]):
        raise ValueError("Checkpoint model, observations, priors, software or resolution settings changed")


def trial_inputs(state, fit):
    vector, log_u, rng_after, key = sampler.trial(state)
    prior = float(fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas))
    if not np.isfinite(prior) and prior != -np.inf:
        raise ArithmeticError("Invalid original physical prior")
    base = None
    if np.isfinite(prior):
        base = dict(samples=[dict(rows=[dict(physical=physical(fit, vector))])],
                    source_revision=state["target"]["original_fit"]["assets"], observations=fit.y_data,
                    uncertainties=fit.y_err, reducers=fit.x_data_reducer,
                    physical_constants=dict(G_SI=const.G, solar_mass_kg=float(units.solMass.to(units.kg))))
    return vector, log_u, rng_after, key, prior, base


def block_path(checkpoint, step, rule, depth, part):
    return checkpoint.parent / f"step-{step:06d}-{rule}-depth-{depth}-part-{part}.json"


def read_block(path, binding):
    record = sampler.read_record(path)
    if record["binding"] != binding:
        raise ValueError("Phase block belongs to a different trial, rule or generation")
    return record["block"]


def block_binding(key, rule, depth, part):
    return dict(trial=key, rule=rule, depth=depth, part=part)


def evaluate(path, step, rule, depth, part, state, key, base, *, gpu_window=False):
    if base is None:
        raise ValueError("Out-of-prior draw requires no synthesis")
    filename = block_path(path, step, rule, depth, part)
    binding = block_binding(key, rule, depth, part)
    with filename.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if filename.exists():
            read_block(filename, binding)
            return
        if state["backend"] == "gpu" and not gpu_window:
            raise ValueError("GPU synthesis requires a verified bounded unit")
        start = perf_counter()
        block = evaluate_block(base, key, state["backend"], depth, part, symmetric_interior=4 if rule=="six" else None,
                               node_reemission=True)
        block["inner_seconds"] = perf_counter() - start
        sampler.write_record(filename, dict(binding=binding, block=block))


def assess(path, step, state, vector, key, prior, base, fit):
    sets, input_hashes = [], []
    for rule, degree in (("six", 4), ("duffy", None)):
        resolutions = []
        for depth in (3, 4):
            blocks = []
            for part in (0, 1):
                f = block_path(path, step, rule, depth, part)
                block = read_block(f, block_binding(key, rule, depth, part))
                if block["backend"] != state["backend"] or block["depth"] != depth or block["part"] != part:
                    raise ValueError("Incompatible backend, depth or phase indices")
                blocks.append(block)
                input_hashes.append(sampler.digest(block))
            joined = join(base, key, blocks, symmetric_interior=degree, node_reemission=True)
            # The fitter's grid checker expects the serialized profile contract used by the phase driver.
            resolutions.append(json.loads(sampler.encoded(joined)))
        sets.append(resolutions)
    profiles = [p for res in sets for item in res for p in item["profiles"]]
    sampler.validate_physics(fit, vector, profiles, state["backend"])
    report = compare(*sets, None, node_reemission=True)
    # The curve constructor recovers normalized coordinates from the physical parameters.
    # For the chain, retain the original prior at the actual proposed normalized vector.
    props = physical(fit, vector)
    recovered = parameters.vector_normalizer([props[k] for k in fit.fitable], fit.fitable, fit.normalization)
    prior_shift = prior - float(fit.ln_prior(np.asarray(recovered), fit.norm_init_vector, fit.sigmas))
    if not np.isfinite(prior_shift):
        raise ArithmeticError("Invalid prior after physical parameter round trip")
    corners = {}
    for rule, degree, items in zip(("six", "duffy"), (4, None), sets, strict=True):
        finite, _, _ = grid(items, degree, node_reemission=True)
        corners[rule] = {str(i): d+prior_shift for i,d in finite["corner_log_posteriors"].items()}
    if report["usable_log_posterior"] is not None:
        report["usable_log_posterior"] += prior_shift
    return dict(report=report, corner_log_posteriors=corners, input_hashes=input_hashes, prior_shift=prior_shift)


def transition(state, step, vector, log_u, rng_after, prior, evidence):
    proposed_density = -np.inf
    if np.isfinite(prior):
        report = evidence["report"]
        if not report["pass_finite_check"] or report["usable_log_posterior"] is None:
            raise ResolutionNotQualifiedError("Physical draw exceeds the sixteen-model resolution budget", report)
        proposed_density = report["usable_log_posterior"]
        if not np.isfinite(proposed_density):
            raise ArithmeticError("Nonfinite qualified posterior")
    accepted = bool(log_u < min(0., proposed_density - state["density"]))
    position = vector.tolist() if accepted else state["position"]
    density = proposed_density if accepted else state["density"]
    event = dict(step=step, proposed=vector.tolist(), log_u=log_u, accepted=accepted, selected=position,
                 selected_density=density, outside_prior=bool(prior == -np.inf), evidence=evidence)
    return dict(state, position=position, density=density, events=[*state["events"], event], rng=rng_after, step=step)


def run(args):
    path = args.checkpoint
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if args.command=="advance" else fcntl.LOCK_SH)
        state = sampler.read_record(path)
        if state["backend"] != args.backend:
            raise ValueError("Checkpoint backend mismatch")
        fit, _ = fixture()
        check_target(state, fit)
        if args.step == state["step"] and args.command == "advance" and state["events"]:
            return
        if args.step != state["step"] + 1:
            raise ValueError("Unexpected chain generation")
        vector, log_u, rng_after, key, prior, base = trial_inputs(state, fit)
        if args.command == "evaluate":
            evaluate(path, args.step, args.rule, args.depth, args.part, state, key, base,
                     gpu_window=args.verified_bounded_gpu_window)
            return
        evidence = None if base is None else assess(path, args.step, state, vector, key, prior, base, fit)
        if args.command == "assess":
            sampler.write_record(path.parent / f"assessment-{args.step:06d}.json",
                                 dict(key=key, evidence=evidence, outside_prior=base is None))
            return
        updated = transition(state, args.step, vector, log_u, rng_after, prior, evidence)
        sampler.write_record(path, updated)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("evaluate","assess","advance"))
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--backend", required=True, choices=("reference","gpu"))
    p.add_argument("--rule", choices=("six","duffy"))
    p.add_argument("--part", type=int, choices=(0,1))
    p.add_argument("--depth", type=int, choices=(3,4))
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    args = p.parse_args()
    if args.command == "evaluate" and None in (args.rule, args.part, args.depth):
        p.error("Evaluation needs an interior rule, eclipse depth and phase block")
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        run(args)
