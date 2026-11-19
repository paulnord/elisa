"""Compare independent, resolution-checked physical transitions using a bounded device field owner and CPU walkers."""

import argparse
import fcntl
import hashlib
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import replace
import json
import multiprocessing
import resource
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from elisa import settings
from elisa.binary_system import dynamic
from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.receiver_phase_parts import context, pack_block
from scripts.reemission_walker_fields import FieldExporter, WalkerObserver, binding, load_fields, rule_for, save_fields
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings


def initialize(root, backend, walkers):
    """Resume the two published streams and seed additional streams at their qualified starting point."""
    if not 2 <= walkers <= 128 or backend not in ("reference", "gpu"):
        raise ValueError("Select 2–128 independent walkers and a supported field backend")
    directory = sampler.ROOT / "docs/ada_validation"
    published = json.loads((directory / "background-m040.json").read_bytes())
    raw = (directory / "background-m040-starts.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != published["starting_checkpoints_sha256"]:
        raise ValueError("Published starting fixture checksum mismatch")
    starts = json.loads(raw)
    root.mkdir()
    for i in range(walkers):
        seed = 20261117 + i
        record = starts[str(seed if i < 2 else 20261117)][backend]
        state = record["data"]
        if record["sha256"] != sampler.digest(state):
            raise ValueError("Invalid published checkpoint")
        if i < 2:
            row = published["draws"][i]
            vector, log_u, rng, _ = sampler.trial(state)
            np.testing.assert_array_equal(vector, row["draw"]["proposed"])
            check = row[backend]
            evidence = {k: check[k] for k in ("corner_log_posteriors", "input_hashes", "prior_shift", "report")}
            state = chain.transition(state, 1, vector, log_u, rng, row["draw"]["prior"], evidence)
            digest = hashlib.sha256(sampler.encoded(dict(data=state, sha256=sampler.digest(state)))).hexdigest()
            if digest != check["final_record_sha256"]:
                raise ValueError("Could not reconstruct the published physical transition")
        else:
            state = dict(state, seed=seed, rng=np.random.PCG64(seed).state)
        destination = root / f"walker-{i:02d}"
        destination.mkdir()
        sampler.write_record(destination / "start.json", state)


def input_state(path):
    state = sampler.read_record(path)
    fit, _ = fixture()
    chain.check_target(state, fit)
    draw = chain.trial_inputs(state, fit)
    return state, fit, draw


def build_fields(state, fit, base, key, *, verified_gpu=False):
    if state["backend"] == "gpu" and not verified_gpu:
        raise ValueError("Device field calculation requires a verified bounded unit")
    _, _, _, _, model = context(base)
    observer = copy(fit.observer)
    observer._system = model
    solver = FieldExporter(list(fit.y_data), rule_for("six", 3), device=state["backend"], selective_shadow=True)
    return solver.collect(model, observer, binding(model, observer, state, key))


def export(path, *, verified_gpu=False):
    state, fit, (_, _, _, key, _, base) = input_state(path)
    if base is not None:
        fields = build_fields(state, fit, base, key, verified_gpu=verified_gpu)
        save_fields(path.with_name("fields.npz"), *fields)


def synthesize(path, state, key, base, fit, fields):
    _, props, _, phases, model = context(base)
    identity = binding(model, fit.observer, state, key)
    meta, arrays = load_fields(path.with_name("fields.npz"), identity) if fields is None else fields
    stages = []
    for rule in ("six", "duffy"):
        for depth in (3, 4):
            start = perf_counter()
            solver = WalkerObserver(list(fit.y_data), rule_for(rule, depth), device="reference", selective_shadow=True)
            solver.attach(model, meta, arrays, identity)
            batch = solver.batch([model], [phases], observer=fit.observer, source_revision=base["source_revision"])
            result = batch.results[0]
            elapsed = perf_counter() - start
            stages.append(dict(interior=rule, depth=depth, seconds=elapsed, stages=result.timings))
            observer = copy(fit.observer)
            observer._system = model
            # Partition the returned physical components only after synthesizing the complete orbit.
            for part, ix in enumerate(np.array_split(np.arange(len(phases)), 2)):
                subset = {
                    k: replace(
                        c, phases=c.phases[ix], component_fluxes={b: f[ix] for b, f in c.component_fluxes.items()}
                    )
                    for k, c in result.curves.items()
                }
                block = pack_block(
                    key,
                    state["backend"],
                    depth,
                    part,
                    ix,
                    phases,
                    props,
                    model.phase_shift,
                    replace(result, curves=subset),
                    batch,
                )
                resolved = observer.manage_time_series(phases=phases[ix])
                folded, _ = observer.phase_interval_reduce(resolved)
                support, _ = dynamic.phase_crv_symmetry(model, folded)
                block["phase_records"] = [r for r in block["phase_records"] if r["phase"] in support]
                np.testing.assert_array_equal(sorted(r["phase"] for r in block["phase_records"]), support)
                block["synthesis_boundary"] = "single full orbit, then partitioned for the original grid checker"
                block["inner_seconds"] = elapsed if part == 0 else 0.0
                filename = chain.block_path(path, state["step"] + 1, rule, depth, part)
                sampler.write_record(filename, dict(binding=chain.block_binding(key, rule, depth, part), block=block))
    return stages


def walker(path):
    start = perf_counter()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        state, fit, (vector, log_u, rng, key, prior, base) = input_state(path)
        step = state["step"] + 1
        if path.with_name("result.json").exists():
            raise FileExistsError("Each timed comparison requires a fresh walker directory")
        fields_start = perf_counter()
        fields = None
        if base is not None and state["backend"] == "reference":
            fields = build_fields(state, fit, base, key)
        fields_end = perf_counter()
        stages = synthesize(path, state, key, base, fit, fields) if base is not None else []
        evidence = chain.assess(path, step, state, vector, key, prior, base, fit) if base is not None else None
        sampler.write_record(
            path.with_name(f"assessment-{step:06d}.json"), dict(key=key, evidence=evidence, outside_prior=base is None)
        )
        status = "accuracy_stop"
        if evidence is None or evidence["report"]["pass_finite_check"]:
            updated = chain.transition(state, step, vector, log_u, rng, prior, evidence)
            sampler.write_record(path.with_name("state.json"), updated)
            status = (
                "outside_prior" if base is None else "accepted" if updated["events"][-1]["accepted"] else "rejected"
            )
        else:
            sampler.write_record(path.with_name("state.json"), state)
        result = dict(
            seed=state["seed"],
            trial=key,
            source_step=state["step"],
            sampled_step=step,
            status=status,
            stages=stages,
            seconds=perf_counter() - start,
            host_fields_seconds=fields_end - fields_start,
            pid=os.getpid(),
            peak_process_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            affinity=sorted(os.sched_getaffinity(0)),
        )
        sampler.write_record(path.with_name("result.json"), result)
        return result


def completed(path, state):
    """Validate a durable transition before omitting it from a resumed worker pool."""
    result = sampler.read_record(path.with_name("result.json"))
    selected = sampler.read_record(path.with_name("state.json"))
    step = state["step"] + 1
    assessment = sampler.read_record(path.with_name(f"assessment-{step:06d}.json"))
    vector, log_u, rng, key = sampler.trial(state)
    if (
        key != assessment["key"]
        or key != result["trial"]
        or step != result["sampled_step"]
        or result["seed"] != state["seed"]
        or result["source_step"] != state["step"]
    ):
        raise ValueError("Completed walker belongs to another trial")
    if result["status"] == "accuracy_stop":
        if assessment["outside_prior"] or assessment["evidence"]["report"]["pass_finite_check"] or selected != state:
            raise ValueError("Invalid resolution stop in completed walker")
    else:
        # Only the prior's finiteness enters transition after the density has been qualified.
        prior = -np.inf if assessment["outside_prior"] else 0.0
        expected = chain.transition(state, step, vector, log_u, rng, prior, assessment["evidence"])
        status = (
            "outside_prior"
            if assessment["outside_prior"]
            else "accepted"
            if expected["events"][-1]["accepted"]
            else "rejected"
        )
        if selected != expected or result["status"] != status:
            raise ValueError("Saved walker checkpoint differs from its physical transition")
    return result


def run_pool(root, workers, *, resume=False):
    paths = sorted(root.glob("walker-*/start.json"))
    if len(paths) < workers or workers not in (1, 8, 16):
        raise ValueError("Supply at least one independent proposal per worker")
    with (root / "pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        starts = [sampler.read_record(p) for p in paths]
        if len({s["seed"] for s in starts}) != len(paths) or len({sampler.digest(s["target"]) for s in starts}) != 1:
            raise ValueError("Expected independent streams on one physical target")
        if len({s["backend"] for s in starts}) != 1:
            raise ValueError("Mixed field backends in a timed comparison")
        todo, rows = [], []
        for path, state in zip(paths, starts, strict=True):
            if path.with_name("result.json").exists():
                if not resume:
                    raise ValueError("Each timing needs a fresh walker directory")
                rows.append(completed(path, state))
            else:
                todo.append(path)
        start = perf_counter()
        if todo:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                rows.extend(pool.map(walker, todo, chunksize=1))
        sampler.write_record(
            root / ("resumed-pool.json" if resume else "pool.json"),
            dict(
                workers=workers,
                sampler_pool_seconds=perf_counter() - start,
                evaluated=len(todo),
                rows=sorted(rows, key=lambda r: r["seed"]),
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("root", type=Path)
    init.add_argument("--backend", choices=("reference", "gpu"), required=True)
    init.add_argument("--walkers", type=int, default=16)
    exp = sub.add_parser("fields")
    exp.add_argument("start", type=Path, nargs="+")
    exp.add_argument("--verified-bounded-gpu-window", action="store_true")
    walk = sub.add_parser("pool")
    walk.add_argument("root", type=Path)
    walk.add_argument(
        "--resume", action="store_true", help="Validate completed checkpoints and evaluate pending walkers."
    )
    walk.add_argument("--workers", type=int, required=True)
    args = p.parse_args()
    if args.command == "init":
        initialize(args.root, args.backend, args.walkers)
    elif args.command == "fields":
        with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
            for start in args.start:
                export(start, verified_gpu=args.verified_bounded_gpu_window)
    else:
        run_pool(args.root, args.workers, resume=args.resume)
