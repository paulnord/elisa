"""Checkpointable random-walk Metropolis comparison on the reflected B/V/bolometric fixture."""

import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from scripts.compare_reflected_posteriors import replay
from scripts.reflected_trial import evaluate_depth, fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import physical
from scripts.validate_gpu_surface_error import asset_revision

ROOT = Path(__file__).resolve().parents[1]


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=lambda a: a.tolist()).encode()


def digest(obj):
    return hashlib.sha256(encoded(obj)).hexdigest()


def context(fit):
    # Pin the forward-source inventory used by the reference, plus the paired observer and transition code.
    files = json.loads((ROOT / "docs/ada_validation/background-m028-source-hashes.json").read_text())
    paths = [k for k in files if k.startswith("src/elisa/")]
    paths.extend(["src/elisa/experimental/paired_reflection.py", *[
        f"scripts/{s}.py" for s in ("sample_reflected_binary", "reflected_trial", "compare_reflected_posteriors",
                                    "compare_reflection_resolution", "compare_gpu_joint_budget", "validate_gpu_fit",
                                    "validate_gpu_surface_error", "validate_gpu_selective", "validate_gpu_atmosphere")
    ]])
    code = digest({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths})
    return dict(protocol="paired-emitting-area-v1", source=code, assets=asset_revision(),
                packages={k: version(k) for k in ("elisa", "numpy", "scipy", "astropy", "numba", "jax", "jaxlib")},
                observations=digest([fit.y_data, fit.y_err, fit.x_data_reducer, fit.x_data_reduced]),
                fitable=list(fit.fitable), normalization=fit.normalization,
                prior_mean=fit.norm_init_vector.tolist(),
                prior_sigma=[float(v) if np.isfinite(v) else None for v in fit.sigmas])


def write_record(path, data):
    """Durably replace one checksummed record, leaving the previous generation valid on interruption."""
    payload = encoded({"data": data, "sha256": digest(data)})
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
        try:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(name, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            name.unlink(missing_ok=True)


def read_record(path):
    record = json.loads(path.read_bytes())
    if digest(record["data"]) != record["sha256"]:
        raise ValueError(f"Invalid record checksum in {path.name}")
    return record["data"]


def trial(state):
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = state["rng"]
    vector = np.asarray(state["position"]) + rng.normal(size=len(state["position"])) * np.asarray(state["scales"])
    log_u = float(np.log1p(-rng.random()))
    key = digest([state["target"], state["backend"], state["step"] + 1, vector, log_u])
    return vector, log_u, rng.bit_generator.state, key


def part_path(directory, step, depth):
    return directory / f"step-{step:06d}-depth-{depth}.json"


def validate_physics(fit, vector, profiles, backend):
    expected = physical(fit, vector)
    for p in profiles:
        row = p["samples"][0]["rows"][0]
        if p["backend"] != backend or row["physical"] != expected:
            raise ValueError("Wrong physical proposal or backend in finite models")


def initialize(args, fit, truth, target):
    if args.checkpoint.exists():
        raise FileExistsError("A chain already exists at this checkpoint")
    profiles = []
    for path in args.initial:
        source = json.loads(path.read_bytes())
        profiles.extend(source["profiles"] if "profiles" in source else [source])
    if len(profiles) != 4:
        raise ValueError("Supply both paired depth results or all four individual starting profiles")
    validate_physics(fit, truth, profiles, args.backend)
    report = replay(profiles)
    if not math.isfinite(args.scale) or args.scale <= 0:
        raise ValueError("Use a finite positive proposal width in normalized coordinates")
    state = dict(step=0, backend=args.backend, target=target, position=truth.tolist(),
                 density=report["usable_log_posterior"], initial=report,
                 scales=[args.scale] * len(truth), seed=args.seed,
                 rng=np.random.PCG64(args.seed).state, events=[])
    write_record(args.checkpoint, state)


def take_step(args, fit, state):
    if args.step != state["step"] + 1:
        raise ValueError("Expected step differs from the durable chain generation")
    vector, log_u, rng_after, key = trial(state)
    prior = fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas)
    if not (np.isfinite(prior) or prior == -np.inf):
        raise ArithmeticError("Invalid original fit prior")
    directory = args.checkpoint.parent
    backend = state["backend"]
    if args.command == "evaluate":
        if prior == -np.inf:
            raise ValueError("This trial is outside the prior. Run advance directly")
        if backend == "gpu" and not args.verified_bounded_gpu_window:
            raise ValueError("Use a verified bounded GPU unit for forward synthesis")
        path = part_path(directory, args.step, args.depth)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                saved = read_record(path)
                if saved["key"] != key or saved["depth"] != args.depth:
                    raise ValueError("A different proposal owns this depth result")
                return
            result = evaluate_depth(fit, vector, args.depth, backend)
            write_record(path, dict(key=key, depth=args.depth, **result))
        return

    report = None
    input_hashes = []
    if np.isfinite(prior):
        profiles = []
        for d in (3, 4):
            path = part_path(directory, args.step, d)
            part = read_record(path)
            if part["key"] != key or part["depth"] != d or any(p["depth"] != d for p in part["profiles"]):
                raise ValueError("Depth result does not belong to this trial")
            input_hashes.append(digest(part))
            profiles.extend(part["profiles"])
        validate_physics(fit, vector, profiles, backend)
        report = replay(profiles)  # A failed accuracy check propagates before any chain or RNG write.
        new_density = report["usable_log_posterior"]
        if not np.isfinite(new_density):
            raise ArithmeticError("Nonfinite guarded posterior")
        accepted = log_u < min(0.0, new_density - state["density"])
    else:
        accepted = False
    if accepted:
        state["position"] = vector.tolist()
        state["density"] = new_density
    event = dict(step=args.step, proposed=vector.tolist(), log_u=log_u, accepted=accepted,
                 selected=state["position"], selected_density=state["density"], report=report, inputs=input_hashes)
    state["events"].append(event)
    state.update(rng=rng_after, step=args.step)
    write_record(args.checkpoint, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "evaluate", "advance"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--initial", nargs="+", type=Path)
    parser.add_argument("--backend", choices=("reference", "cpu", "gpu"), default="reference")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--scale", type=float, default=0.0006)
    parser.add_argument("--depth", type=int, choices=(3, 4))
    parser.add_argument("--step", type=int)
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    args = parser.parse_args()
    if args.command == "init" and not args.initial:
        parser.error("init requires the complete generating model in --initial")
    if args.command == "evaluate" and args.depth is None:
        parser.error("evaluate requires --depth")
    started = perf_counter()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with (args.checkpoint.with_suffix(".lock").open("a") as lock,
          table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2)):
        fcntl.flock(lock, fcntl.LOCK_SH if args.command == "evaluate" else fcntl.LOCK_EX)
        fit, truth = fixture()
        target = context(fit)
        if args.command == "init":
            initialize(args, fit, truth, target)
        else:
            state = read_record(args.checkpoint)
            if digest(state["target"]) != digest(target):
                raise ValueError("Model, data, prior or software changed since the checkpoint")
            take_step(args, fit, state)
    print(f"{args.command} finished in {perf_counter()-started:.3f} s", flush=True)


if __name__ == "__main__":
    main()
