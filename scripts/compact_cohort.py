"""Time independent walkers on the compact eclipse target, including fresh physical fields."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from functools import partial
import hashlib
from itertools import product
import json
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from elisa import settings
from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_stationary_eclipse import run
from scripts.compact_interior_fit import compact_inputs
from scripts.compare_compact_draws import selected_corners
from scripts.compare_reemission_workers import acceptance_margins
from scripts.validate_gpu_atmosphere import table_settings


def decision(state, prior, vector, log_u, rng, assessment):
    """Only an original-prior rejection or a resolution-insensitive decision advances a stream."""
    step = state["step"] + 1
    if prior == -np.inf:
        if assessment is not None:
            raise ValueError("Outside-prior proposal has a physical assessment")
        return chain.transition(state, step, vector, log_u, rng, prior, None), "outside_prior", None
    if not np.isfinite(prior):
        raise ArithmeticError("Invalid physical prior")
    evidence = assessment["evidence"]
    if evidence is None:
        if assessment.get("status") != "node_cap" or not assessment["failed_blocks"]:
            raise ValueError("Missing bound resource stop")
        return state, "node_cap", None
    report = evidence["report"]
    if not report["pass_finite_check"]:
        if report["usable_log_posterior"] is not None:
            raise ArithmeticError("Unqualified proposal has a usable density")
        return state, "accuracy_stop", None
    margins = acceptance_margins(dict(log_uniform=log_u), selected_corners(state), evidence)
    if not np.isfinite(margins).all():
        raise ArithmeticError("Nonfinite finite-resolution acceptance margins")
    new = chain.transition(state, step, vector, log_u, rng, prior, evidence)
    accepted = new["events"][-1]["accepted"]
    if (accepted and margins[0] <= 0) or (not accepted and margins[1] >= 0):
        return state, "decision_stop", margins
    return new, ("accepted" if accepted else "rejected"), margins


def restore(fixture, backend, destination):
    states = sampler.read_record(fixture)[backend]
    if not states or len({s["seed"] for s in states}) != len(states):
        raise ValueError("Expected distinct seeds")
    if any(s["backend"] != backend or s["target"] != states[0]["target"] for s in states):
        raise ValueError("Mixed backend or target")
    if destination.exists():
        raise FileExistsError(destination)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        batch = Path(tmp) / "generation"
        batch.mkdir()
        for i, state in enumerate(states):
            directory = batch / f"walker-{i:02d}"
            directory.mkdir()
            path = directory / "start.json"
            sampler.write_record(path, state)
            compact_inputs(path, False)
        batch.rename(destination)


def assemble(continuations, count):
    """Resume the two m044 histories and create independent PCG64 streams at the qualified injection."""
    if not 2 <= count <= 128:
        raise ValueError("Expected 2–128 independent streams")
    directory = sampler.ROOT / "docs/ada_validation"
    published = json.loads((directory / "background-m044.json").read_bytes())
    raw = (directory / "background-m044-checkpoints.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != published["checkpoints_sha256"]:
        raise ValueError("Published checkpoints changed")
    archive = json.loads(raw)
    first_seed, second_seed = sorted(map(int, archive))
    if second_seed != first_seed + 1:
        raise ValueError("Unexpected original seed sequence")
    result = {}
    for backend in ("reference", "gpu"):
        states = []
        for seed in range(first_seed, first_seed + count):
            original = archive[str(min(seed, second_seed))][backend]
            if seed <= second_seed:
                previous = original["selected"]["data"]
                directory = continuations / backend / str(seed)
                start = sampler.read_record(directory / "start.json")
                if previous != start:
                    raise ValueError("Continuation does not begin at the published checkpoint")
                _, _, v, _, prior, _ = compact_inputs(directory / "start.json", False)
                _, u, rng, _ = sampler.trial(start)
                assessment = sampler.read_record(directory / "assess.json") if np.isfinite(prior) else None
                if assessment is not None:
                    verify_assessment(directory / "start.json", directory, assessment)
                state, status, _ = decision(start, prior, v, u, rng, assessment)
                if status not in ("accepted", "rejected", "outside_prior") or state != sampler.read_record(
                    directory / "state.json"
                ):
                    raise ValueError("Continuation lacks a verified transition")
            else:
                state = dict(original["initial"]["data"], seed=seed, rng=np.random.PCG64(seed).state)
            states.append(state)
        result[backend] = states
    for a, b in zip(result["reference"], result["gpu"], strict=True):
        for key in ("seed", "target", "scales", "position", "rng", "step"):
            if a[key] != b[key]:
                raise ValueError(f"Different matched starting {key}")
        np.testing.assert_allclose(a["density"], b["density"], atol=1e-6, rtol=0)
    return result


def options(task, path, output, **kwargs):
    return SimpleNamespace(
        task=task,
        checkpoint=path,
        output=output,
        fields=path.parent / "fields.npz",
        injection=False,
        compact_interior=True,
        **kwargs,
    )


def verify_assessment(path, directory, saved):
    current = run(options("assess", path, directory))
    for key in ("key", "evidence", "prior", "normalized", "physical", "failed_blocks"):
        if sampler.digest(current.get(key)) != sampler.digest(saved.get(key)):
            raise ValueError(f"Assessment differs from its phase blocks at {key}")


def fields(path, *, gpu):
    state, _, _, _, _, base = compact_inputs(path, False)
    if (state["backend"] == "gpu") != gpu:
        raise ValueError("Field owner does not match the checkpoint backend")
    if base is not None:
        record = run(options("fields", path, path.parent, verified_bounded_gpu_window=gpu))
        sampler.write_record(path.with_name("fields.json"), record)


def walker(path, library):
    start = perf_counter()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        with path.with_name("draw.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            dest = path.with_name("draw")
            if dest.exists():
                raise FileExistsError("Use a fresh directory for an independent timing")
            state, _, v, key, prior, base = compact_inputs(path, False)
            _, u, rng, _ = sampler.trial(state)
            with TemporaryDirectory(dir=path.parent) as tmp:
                out = Path(tmp) / "draw"
                out.mkdir()
                field_start = perf_counter()
                if state["backend"] == "reference":
                    fields(path, gpu=False)
                observer_start = perf_counter()
                assessment = None
                if base is not None:
                    for rule, depth, part in product(("six", "duffy"), (3, 4), (0, 1)):
                        args = options(
                            "block",
                            path,
                            out,
                            rule=rule,
                            depth=depth,
                            part=part,
                            library=library,
                            audit=False,
                            verified_bounded_gpu_window=False,
                        )
                        block = run(args)
                        sampler.write_record(out / f"{rule}-{depth}-{part}.json", block)
                    assessment = run(options("assess", path, out))
                    sampler.write_record(out / "assess.json", assessment)
                selected, status, margins = decision(state, prior, v, u, rng, assessment)
                sampler.write_record(out / "state.json", selected)
                row = dict(
                    seed=state["seed"],
                    step=state["step"] + 1,
                    trial=key,
                    prior_finite=bool(np.isfinite(prior)),
                    status=status,
                    margins=margins,
                    seconds=perf_counter() - start,
                    pid=os.getpid(),
                    host_fields_seconds=observer_start - field_start,
                    affinity=sorted(os.sched_getaffinity(0)),
                )
                sampler.write_record(out / "result.json", row)
                out.rename(dest)
                return row


def replay(path, out):
    """Recompute the gate and the selected checkpoint from the original draw and its eight phase blocks."""
    state, fit, v, key, prior, base = compact_inputs(path, False)
    _, u, rng, _ = sampler.trial(state)
    assessment = None
    if base is not None:
        assessment = sampler.read_record(out / "assess.json")
        verify_assessment(path, out, assessment)
    selected, status, margins = decision(state, prior, v, u, rng, assessment)
    if selected != sampler.read_record(out / "state.json"):
        raise ValueError("Stored position, density, history or RNG differs from physical transition")
    record = out / "result.json"
    if record.exists():
        meta = sampler.read_record(record)
        if (meta["seed"], meta["step"], meta["status"], meta["margins"], meta["trial"]) != (
            state["seed"],
            state["step"] + 1,
            status,
            margins,
            key,
        ):
            raise ValueError("Worker summary differs from the reconstructed decision")
    evidence = assessment["evidence"] if assessment is not None else None
    return dict(
        start=state,
        selected=selected,
        status=status,
        margins=margins,
        proposed=v,
        prior=prior,
        log_u=u,
        evidence=evidence,
        fit=fit,
    )


def pool(root, workers, library, *, resume=False):
    starts = sorted(root.glob("walker-*/start.json"))
    states = [sampler.read_record(p) for p in starts]
    if workers not in (1, 8, 16) or len(states) < workers:
        raise ValueError("Each requested process must have an independent proposal")
    if (
        len({s["seed"] for s in states}) != len(states)
        or len({(s["backend"], sampler.digest(s["target"])) for s in states}) != 1
    ):
        raise ValueError("Expected independent streams on one physical target and backend")
    with (root / "pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        t = perf_counter()
        finished = [p for p in starts if p.with_name("draw").exists()]
        if finished and not resume:
            raise FileExistsError("Completed walkers exist; use --resume or a fresh run directory")
        rows = {}
        for path in finished:
            replay(path, path.with_name("draw"))
            rows[path] = sampler.read_record(path.with_name("draw") / "result.json")
        pending = [p for p in starts if p not in rows]
        if pending:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as processes:
                results = processes.map(partial(walker, library=library), pending, chunksize=1)
                rows.update(zip(pending, results, strict=True))
        name = "resumed-pool.json" if resume else "pool.json"
        sampler.write_record(
            root / name,
            dict(workers=workers, evaluated=len(pending), seconds=perf_counter() - t, rows=[rows[p] for p in starts]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("continuations", type=Path)
    a.add_argument("output", type=Path)
    a.add_argument("--count", type=int, default=16)
    r = sub.add_parser("restore")
    r.add_argument("fixture", type=Path)
    r.add_argument("backend", choices=("reference", "gpu"))
    r.add_argument("directory", type=Path)
    f = sub.add_parser("fields")
    f.add_argument("starts", nargs="+", type=Path)
    f.add_argument("--verified-bounded-gpu-window", action="store_true", required=True)
    p = sub.add_parser("pool")
    p.add_argument("directory", type=Path)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument(
        "--resume", action="store_true", help="Verify completed proposals and evaluate the remaining streams"
    )
    args = parser.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        if args.command == "assemble":
            sampler.write_record(args.output, assemble(args.continuations, args.count))
        elif args.command == "restore":
            restore(args.fixture, args.backend, args.directory)
        elif args.command == "fields":
            for start in args.starts:
                fields(start, gpu=args.verified_bounded_gpu_window)
        else:
            pool(args.directory, args.workers, args.library, resume=args.resume)
