"""Bind the revised Duffy control to independent, resolution-checked physical chains."""

import argparse
import fcntl
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from elisa import settings
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_stationary_eclipse import inputs, run
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings


def target_for(original):
    # The existing target pins the stellar physics. Bind the additional observer,
    # stationary solver and numerical qualification path used by this sampler.
    files = [
        "src/elisa/experimental/compact_interior.py",
        *[
            "scripts/" + name
            for name in (
                "compact_interior_fit.py",
                "benchmark_stationary_eclipse.py",
                "fused_stationary_shadow.py",
                "stationary_eclipse.cu",
                "reemission_walker_fields.py",
                "walker_roche_geometry.py",
                "benchmark_reemission_walkers.py",
            )
        ],
    ]
    sources = {name: hashlib.sha256((sampler.ROOT / name).read_bytes()).hexdigest() for name in files}
    return dict(original, compact_interior=dict(sibling_levels=1, eclipse_only=True, sources=sources))


def compact_inputs(checkpoint, injection):
    saved = sampler.read_record(checkpoint)
    target = saved["target"]
    if "compact_interior" in target:
        original = {k: v for k, v in target.items() if k != "compact_interior"}
        fit, _ = fixture()
        chain.check_target(dict(target=original), fit)
        if injection or target != target_for(original):
            raise ValueError("Expected a matching compact-interior chain and a fresh sampler proposal")
        vector, _, _, key, prior, base = chain.trial_inputs(saved, fit)
        return saved, fit, vector, key, prior, base
    # A historical checkpoint can supply a reproducible validation position, never
    # the starting density or history for a chain on the revised resolution test.
    state, fit, vector, old_key, prior, base = inputs(checkpoint, injection)
    target = target_for(target)
    state = dict(state, target=target)
    key = sampler.digest(dict(validation_position=old_key, target=target, backend=state["backend"]))
    return state, fit, vector, key, prior, base


def qualified_evidence(checkpoint, directory, injection):
    a = run(
        SimpleNamespace(
            task="assess",
            checkpoint=checkpoint,
            output=directory,
            fields=None,
            compact_interior=True,
            injection=injection,
        )
    )
    evidence = a["evidence"]
    if evidence is None or not evidence["report"]["pass_finite_check"]:
        report = a if evidence is None else evidence["report"]
        raise ResolutionNotQualifiedError("Physical position did not pass all sixteen resolution checks", report)
    return evidence


def initialize(template, directory, checkpoint, seed):
    with checkpoint.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state, _, vector, _, _, _ = compact_inputs(template, True)
        evidence = qualified_evidence(template, directory, True)
        result = dict(
            target=state["target"],
            backend=state["backend"],
            position=vector.tolist(),
            step=0,
            events=[],
            initial=evidence["report"],
            initial_input_hashes=evidence["input_hashes"],
            initial_corner_log_posteriors=evidence["corner_log_posteriors"],
            density=evidence["report"]["usable_log_posterior"],
            seed=seed,
            scales=sampler.read_record(template)["scales"],
            rng=np.random.PCG64(seed).state,
        )
        if checkpoint.exists():
            if sampler.read_record(checkpoint) != result:
                raise ValueError("A different chain already owns the checkpoint")
        else:
            sampler.write_record(checkpoint, result)


def advance(checkpoint, directory, step):
    with checkpoint.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = sampler.read_record(checkpoint)
        if "compact_interior" not in state["target"]:
            raise ValueError("Historical chains cannot advance on a different resolution control")
        if step == state["step"] and state["events"]:
            return
        if step != state["step"] + 1:
            raise ValueError("Unexpected physical chain generation")
        _, _, vector, _, prior, base = compact_inputs(checkpoint, False)
        _, log_u, rng, _ = sampler.trial(state)
        evidence = None if base is None else qualified_evidence(checkpoint, directory, False)
        sampler.write_record(checkpoint, chain.transition(state, step, vector, log_u, rng, prior, evidence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("initialize", "advance"):
        p = sub.add_parser(command)
        p.add_argument("checkpoint", type=Path)
        p.add_argument("directory", type=Path)
        if command == "initialize":
            p.add_argument("--template", type=Path, required=True)
            p.add_argument("--seed", type=int, required=True)
        else:
            p.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        if args.command == "initialize":
            initialize(args.template, args.directory, args.checkpoint, args.seed)
        else:
            advance(args.checkpoint, args.directory, args.step)
