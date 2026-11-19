"""Profile the host eclipse calculation on a checked physical sampler draw."""

import argparse
import cProfile
from pathlib import Path
from time import perf_counter

import numpy as np

from elisa import settings
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.gpu_curved_edges import _clip_labels, _edge_nodes
from elisa.experimental.shadow_refinement import shadow_parents
from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_reemission_walkers import completed, input_state
from scripts.receiver_phase_parts import context
from scripts.reemission_walker_fields import WalkerObserver, binding, load_fields, rule_for
from scripts.validate_gpu_atmosphere import table_settings
from scripts.walker_roche_geometry import WalkerRocheGeometry


class CountedGeometry(WalkerRocheGeometry):
    def __init__(self, binary, component):
        super().__init__(binary, component)
        self.sample_rays = self.shadow_rays = self.stationary_rays = 0

    def sample(self, vectors):
        self.sample_rays += len(vectors)
        return super().sample(vectors)

    def shadow(self, points, line):
        self.shadow_rays += len(points)
        return super().shadow(points, line)

    def potential_local(self, local):
        # shadow() evaluates the potential once on the final stationary points.
        self.stationary_rays += len(local)
        return super().potential_local(local)


class PacketProfile:
    """Time each packet separately so both foreground identity and eclipse phase survive."""

    def __init__(self, packet):
        self.packet = packet
        self.rows = []

    def __call__(self, triangles, geometry, line, rule, foreground):
        geometry.sample_rays = 0
        if foreground is not None:
            foreground.shadow_rays = foreground.stationary_rays = 0
        timer = cProfile.Profile()
        start = perf_counter()
        packet = timer.runcall(self.packet, triangles, geometry, line, rule, foreground)
        elapsed = perf_counter() - start
        entries = {entry.code: entry for entry in timer.getstats()}

        def seconds(method):
            entry = entries.get(method.__code__)
            return entry.totaltime if entry else 0.0

        # These four functions do not call one another. Count only the outer rule's planar method.
        stages = {
            "shadow_parent_selection": seconds(shadow_parents),
            "polygon_clipping": seconds(_clip_labels),
            "curved_edge_slices": seconds(_edge_nodes),
            "planar_quadrature": seconds(type(rule).planar_packet),
        }
        stages["other_packet_work"] = elapsed - sum(stages.values())
        # A second partition by physical operation. Both functions are called from several stages.
        physics = {
            "roche_surface": seconds(WalkerRocheGeometry.sample),
            "shadow": seconds(ConditionedRocheGeometry.shadow),
        }
        physics["other_packet_work"] = elapsed - sum(physics.values())
        if min(*stages.values(), *physics.values()) < 0:
            raise ArithmeticError("Overlapping eclipse timing categories")
        self.rows.append(
            dict(
                component=geometry.component,
                foreground=None if foreground is None else foreground.component,
                packet_seconds=elapsed,
                stage_seconds=stages,
                physical_operation_seconds=physics,
                surface_rays=geometry.sample_rays,
                shadow_rays=foreground.shadow_rays if foreground else 0,
                stationary_rays=foreground.stationary_rays if foreground else 0,
                nodes=packet["nodes"],
                selected_nodes=packet["selected_nodes"],
                crossing_parents=packet.get("shadow_crossing_parents", 0),
                limb_edge_roots=packet["limb_edge_roots"],
                shadow_edge_roots=packet["shadow_edge_roots"],
            )
        )
        return packet


def verify(result, path, state, key, interior, depth):
    maximum = 0.0
    for part in (0, 1):
        block = chain.read_block(
            chain.block_path(path, state["step"] + 1, interior, depth, part),
            chain.block_binding(key, interior, depth, part),
        )
        if result.quadrature_sha256 != block["quadrature_sha256"]:
            raise ValueError("Eclipse quadrature changed during replay")
        records = {r["phase"]: r for r in result.diagnostics["phase_records"]}
        if any(records[row["phase"]] != row for row in block["phase_records"]):
            raise ValueError("Eclipse refinement differs from the saved physical proposal")
        if set(result.curves) != set(block["curves"]):
            raise ValueError("Incomplete field or donor resolutions")
        for name, curve in result.curves.items():
            ref = block["curves"][name]
            if curve.signature != ref["signature"] or curve.additional_light != ref["additional_light"]:
                raise ValueError("Curve identity changed")
            for band, x in ref["components"].items():
                y = curve.component_fluxes[band][block["indices"]]
                x = np.asarray(x)
                np.testing.assert_allclose(y, x, rtol=3e-13, atol=0.0)
                live = x != 0
                maximum = max(maximum, float(np.max(np.abs((x[live] - y[live]) / x[live]), initial=0)))
    return maximum


def profile(path, interior, depth):
    state, fit, (_, _, _, key, _, base) = input_state(path)
    completed(path, state)
    if state["backend"] != "gpu" or base is None:
        raise ValueError("Supply an evaluated in-prior walker with its saved GPU optical fields")
    _, _, _, phases, model = context(base)
    identity = binding(model, fit.observer, state, key)
    meta, arrays = load_fields(path.with_name("fields.npz"), identity)
    timings = []
    for measured in (False, True):
        solver = WalkerObserver(list(fit.y_data), rule_for(interior, depth), device="reference", selective_shadow=True)
        solver.attach(model, meta, arrays, identity)
        if measured:
            solver._geometry = CountedGeometry
            packets = PacketProfile(solver._packet)
            solver._packet = packets
        start = perf_counter()
        batch = solver.batch([model], [phases], observer=fit.observer, source_revision=base["source_revision"])
        elapsed = perf_counter() - start
        result = batch.results[0]
        maximum = verify(result, path, state, key, interior, depth)
        timings.append(
            dict(
                instrumented=measured,
                full_orbit_seconds=elapsed,
                stages=result.timings,
                max_component_relative=maximum,
                quadrature_sha256=result.quadrature_sha256,
            )
        )
    rows = iter(packets.rows)
    for phase in result.diagnostics["phase_records"]:
        for component in ("primary", "secondary"):
            row = next(rows)
            if row["component"] != component or row["nodes"] != phase["components"][component]["nodes"]:
                raise ValueError("Packet profile is out of phase order")
            row["phase"] = phase["phase"]
    if next(rows, None) is not None:
        raise ValueError("Unexpected trailing packet")
    return dict(
        seed=state["seed"],
        step=state["step"] + 1,
        trial=key,
        interior=interior,
        depth=depth,
        runs=timings,
        packets=packets.rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("start", type=Path)
    p.add_argument("--interior", required=True, choices=("six", "duffy"))
    p.add_argument("--depth", required=True, type=int, choices=(3, 4))
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        sampler.write_record(args.output, profile(args.start, args.interior, args.depth))
