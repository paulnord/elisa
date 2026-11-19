"""Evaluate the synthetic injection or replay a saved trial using the conditioned eclipse observer."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from elisa import const, settings, units
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_reemission_walkers import build_fields, input_state
from scripts.receiver_phase_parts import context, pack_block
from scripts.reemission_walker_fields import WalkerObserver, binding, load_fields, rule_for, save_fields
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import physical


def inputs(checkpoint, injection, *, compact=False):
    if compact:
        from scripts.compact_interior_fit import compact_inputs
        return compact_inputs(checkpoint, injection)
    if not injection:
        state, fit, (vector, _, _, key, prior, base) = input_state(checkpoint)
        if base is None:
            raise ValueError("Replay requires an in-prior physical proposal")
        return state, fit, vector, key, prior, base
    template = sampler.read_record(checkpoint)
    fit, vector = fixture()
    chain.check_target(template, fit)
    state = dict(target=template["target"], backend=template["backend"], step=-1)
    props = physical(fit, vector)
    key = sampler.digest(dict(target=state["target"], synthetic_injection=props))
    prior = float(fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas))
    if not np.isfinite(prior):
        raise ValueError("Synthetic injection is outside the original prior")
    base = dict(samples=[dict(rows=[dict(physical=props)])], source_revision=state["target"]["original_fit"]["assets"],
                observations=fit.y_data, uncertainties=fit.y_err, reducers=fit.x_data_reducer,
                physical_constants=dict(G_SI=const.G, solar_mass_kg=float(units.solMass.to(units.kg))))
    return state, fit, vector, key, prior, base


def run(args):
    start = perf_counter()
    state, fit, vector, key, prior, base = inputs(args.checkpoint, args.injection, compact=args.compact_interior)
    field_file = args.fields or args.output / "fields.npz"
    path = args.output / "checkpoint.json"
    step = state["step"] + 1
    if args.task == "fields":
        fields = build_fields(state, fit, base, key, verified_gpu=args.verified_bounded_gpu_window)
        save_fields(field_file, *fields)
        return dict(seconds=perf_counter()-start, archive_bytes=field_file.stat().st_size)
    if args.task == "assess":
        stops = []
        for rule in ("six", "duffy"):
            for depth in (3, 4):
                for part in (0, 1):
                    saved = chain.block_path(path, step, rule, depth, part)
                    expected = chain.block_binding(key, rule, depth, part)
                    if saved.exists():
                        chain.read_block(saved, expected)
                    else:
                        record = sampler.read_record(args.output / f"{rule}-{depth}-{part}.json")
                        if record.get("binding") != expected or record.get("status") != "node_cap":
                            raise ValueError("Missing phase block or bound resource-stop receipt")
                        stops.append(record["binding"])
        if stops:
            return dict(physical=physical(fit, vector), normalized=vector, prior=prior, key=key,
                        evidence=None, status="node_cap", failed_blocks=stops, seconds=perf_counter()-start)
        evidence = chain.assess(path, step, state, vector, key, prior, base, fit)
        return dict(physical=physical(fit, vector), normalized=vector, prior=prior, key=key, evidence=evidence,
                    seconds=perf_counter()-start)
    _, props, _, phases, model = context(base)
    identity = binding(model, fit.observer, state, key)
    meta, arrays = load_fields(field_file, identity)
    rule = rule_for(args.rule, args.depth)
    if args.compact_interior:
        from dataclasses import asdict
        from elisa.experimental.compact_interior import CompactDuffyRule, CompanionSixPointRule, compact_shadow_packet
        rule_class = CompactDuffyRule if args.rule == "duffy" else CompanionSixPointRule
        rule = rule_class(**asdict(rule))
    solver = WalkerObserver(list(fit.y_data), rule, device="reference", selective_shadow=True)
    if args.compact_interior:
        solver._packet = compact_shadow_packet
    solver.attach(model, meta, arrays, identity)
    fused = None
    if args.library:
        from scripts.fused_stationary_shadow import StationarySolve
        fused = StationarySolve(args.library, gpu=args.verified_bounded_gpu_window, audit=args.audit)
        solver._geometry = fused.geometry
    indices = np.array_split(np.arange(len(phases)), 2)[args.part]
    try:
        batch = solver.batch([model], [phases[indices]], observer=fit.observer, source_revision=base["source_revision"])
        block = pack_block(key, state["backend"], args.depth, args.part, indices, phases, props, model.phase_shift,
                           batch.results[0], batch)
        block["inner_seconds"] = perf_counter() - start
        sampler.write_record(chain.block_path(path, step, args.rule, args.depth, args.part),
                             dict(binding=chain.block_binding(key, args.rule, args.depth, args.part), block=block))
    except UnsupportedCurvedGeometryError as error:
        if not (args.injection or args.compact_interior) or str(error) != "Refined full shadow packet exceeds the original node cap":
            raise
        return dict(status="node_cap", binding=chain.block_binding(key, args.rule, args.depth, args.part),
                    seconds=perf_counter()-start, reason=str(error), node_cap=solver.rule.max_nodes,
                    stationary=fused.stats if fused else None)
    finally:
        if fused:
            fused.close()
    return dict(seconds=perf_counter()-start, stages=batch.results[0].timings,
                stationary=fused.stats if fused else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("fields", "block", "assess"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--compact-interior", action="store_true", help="Use a separately bound one-level interior Duffy merge")
    p.add_argument("--injection", action="store_true")
    p.add_argument("--fields", type=Path)
    p.add_argument("--rule", choices=("six", "duffy"))
    p.add_argument("--depth", type=int, choices=(3, 4))
    p.add_argument("--part", type=int, choices=(0, 1))
    p.add_argument("--library", type=Path)
    p.add_argument("--audit", action="store_true", help="Compare a stratified sample of each shadow call with NumPy")
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    args = p.parse_args()
    if args.task == "block" and None in (args.rule, args.depth, args.part):
        p.error("block requires a rule, depth and part")
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"{args.rule}-{args.depth}-{args.part}" if args.task == "block" else args.task
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        sampler.write_record(args.output / (name + ".json"), run(args))
