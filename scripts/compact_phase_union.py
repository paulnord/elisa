"""Evaluate the two observation blocks together on one freshly prepared compact-target proposal."""

from copy import copy
from dataclasses import asdict, replace
import hashlib
import json
from time import perf_counter

import numpy as np

from elisa.binary_system import dynamic
from elisa.experimental.compact_interior import CompactDuffyRule, CompanionSixPointRule, compact_shadow_packet
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_stationary_eclipse import run
from scripts.fused_stationary_shadow import StationarySolve
from scripts.receiver_phase_parts import context, pack_block
from scripts.reemission_walker_fields import WalkerObserver, binding, load_fields, rule_for


def supports(observer, model, phases):
    """Apply ELISa's original time, interval and symmetry maps at each block boundary."""
    bound = copy(observer)
    bound._system = model

    def reduce(times):
        phase, _ = bound.phase_interval_reduce(bound.manage_time_series(phases=times))
        return dynamic.phase_crv_symmetry(model, phase)[0]

    blocks = np.array_split(np.arange(len(phases)), 2)
    subsets = [reduce(phases[ix]) for ix in blocks]
    whole = reduce(phases)
    np.testing.assert_array_equal(whole, np.unique(np.concatenate(subsets)))
    return blocks, subsets, whole


class BlockDigests:
    """Stream the original per-block hashes in phase and component order, without retaining ray arrays."""

    def __init__(self, subsets, whole, solver):
        header = json.dumps({"level": solver.integration_level, "rule": asdict(solver.rule)}).encode()
        self.digests = [hashlib.sha256(header) for _ in subsets]
        self.subsets = [set(phases) for phases in subsets]
        self.sequence = iter((float(phase), name) for phase in whole for name in ("primary", "secondary"))

    def packet(self, triangles, geometry, line, rule, foreground=None):
        phase, name = next(self.sequence)
        if name != geometry.component:
            raise ArithmeticError("Unexpected order of stellar quadrature packets")
        packet = compact_shadow_packet(triangles, geometry, line, rule, foreground)
        header = json.dumps({"phase": phase, "component": name, "nodes": len(packet["mu"])}).encode()
        arrays = [np.ascontiguousarray(packet[k]) for k in ("mu", "area", "parents")]
        for selected, digest in zip(self.subsets, self.digests, strict=True):
            if phase in selected:
                digest.update(header)
                for array in arrays:
                    digest.update(array)
        return packet

    def finish(self):
        if next(self.sequence, None) is not None:
            raise ArithmeticError("Observer did not visit all phases and components")
        return [digest.hexdigest() for digest in self.digests]


def account_union_time(block, part, elapsed):
    """The legacy join adds block times. Charge the shared observer call to its first block."""
    block["inner_seconds"] = elapsed if part == 0 else 0.0
    block["timing_scope"] = "joint observation blocks; accounted in part zero"
    if part:
        block["stages"] = {}
        block["batch"] = {}


def evaluate(path, out, state, fit, key, base, library):
    from scripts.compact_cohort import options

    _, props, _, phases, model = context(base)
    identity = binding(model, fit.observer, state, key)
    metadata, arrays = load_fields(path.parent / "fields.npz", identity)
    blocks, subsets, whole = supports(fit.observer, model, phases)
    step = state["step"] + 1
    for label, cls in (("six", CompanionSixPointRule), ("duffy", CompactDuffyRule)):
        for depth in (3, 4):
            start = perf_counter()
            rule = cls(**asdict(rule_for(label, depth)))
            solver = WalkerObserver(list(fit.y_data), rule, device="reference", selective_shadow=True)
            solver.attach(model, metadata, arrays, identity)
            hashes = BlockDigests(subsets, whole, solver)
            solver._packet = hashes.packet
            fused = StationarySolve(library, gpu=False)
            solver._geometry = fused.geometry
            try:
                batch = solver.batch([model], [phases], observer=fit.observer, source_revision=base["source_revision"])
            except UnsupportedCurvedGeometryError as err:
                if str(err) != "Refined full shadow packet exceeds the original node cap":
                    raise
                # Replay each original block to identify the precise stop and preserve its successful sibling.
                for part in range(2):
                    args = options(
                        "block",
                        path,
                        out,
                        rule=label,
                        depth=depth,
                        part=part,
                        library=library,
                        audit=False,
                        verified_bounded_gpu_window=False,
                    )
                    sampler.write_record(out / f"{label}-{depth}-{part}.json", run(args))
                continue
            finally:
                fused.close()

            result = batch.results[0]
            records = result.diagnostics["phase_records"]
            np.testing.assert_array_equal([r["phase"] for r in records], whole)
            for part, (indices, subset, digest) in enumerate(zip(blocks, subsets, hashes.finish(), strict=True)):
                # pack_block stores the phase-indexed component fluxes. Normalization and third light
                # are applied by the original join after both observation blocks have been mapped back.
                curves = {}
                for name, curve in result.curves.items():
                    components = {band: flux[indices] for band, flux in curve.component_fluxes.items()}
                    curves[name] = replace(
                        curve,
                        phases=curve.phases[indices],
                        component_fluxes=components,
                        raw_fluxes=HybridBinaryLightCurve._observed_fluxes(components, model, normalize=False),
                        fluxes=HybridBinaryLightCurve._observed_fluxes(components, model, normalize=True),
                        diagnostics={**curve.diagnostics, "quadrature_sha256": digest},
                    )
                view = replace(
                    result,
                    curves=curves,
                    quadrature_sha256=digest,
                    diagnostics={**result.diagnostics, "phase_records": [r for r in records if r["phase"] in subset]},
                )
                block = pack_block(
                    key, state["backend"], depth, part, indices, phases, props, model.phase_shift, view, batch
                )
                account_union_time(block, part, batch.timings["total"])
                sampler.write_record(
                    chain.block_path(out / "checkpoint.json", step, label, depth, part),
                    dict(binding=chain.block_binding(key, label, depth, part), block=block),
                )
            # One timing per union evaluation. Summing the two block views charges the observer once.
            sampler.write_record(
                out / f"{label}-{depth}-union.json",
                dict(
                    seconds=perf_counter() - start,
                    block_phases=list(map(len, subsets)),
                    union_phases=len(whole),
                    stages=result.timings,
                    stationary=fused.stats,
                ),
            )
