"""Measure projected shadow quadrature demand at each eclipse phase without constructing an over-cap curve."""

import argparse
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from elisa import const, settings, utils
from elisa.binary_system import dynamic
from elisa.experimental.gpu_reconstructed_observer import recorded_packet
from elisa.experimental.shadow_refinement import shadow_parents
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_stationary_eclipse import inputs
from scripts.receiver_phase_parts import context
from scripts.reemission_walker_fields import binding, load_fields, rule_for
from scripts.validate_gpu_atmosphere import table_settings
from scripts.walker_roche_geometry import WalkerRocheGeometry


def shadow_demand(triangles, geometry, line, rule, foreground):
    """Use the same bounded partitions and record their counts before any concatenation."""
    crossed = normal_nodes = refined_nodes = 0
    finer = replace(rule, depth=rule.depth + 1)
    for start in range(0, len(triangles), 4096):
        block = triangles[start : start + 4096]
        mask = shadow_parents(block, geometry, line, foreground)
        crossed += int(mask.sum())
        if not mask.all():
            normal_nodes += recorded_packet(block[~mask], geometry, line, rule, foreground)["nodes"]
        if mask.any():
            refined_nodes += recorded_packet(block[mask], geometry, line, finer, foreground)["nodes"]
    total = normal_nodes + refined_nodes
    return dict(
        crossing_parents=crossed,
        ordinary_nodes=normal_nodes,
        refined_nodes=refined_nodes,
        total=total,
        limit=rule.max_nodes,
        fits=total <= rule.max_nodes,
    )


def count(checkpoint, fields, part):
    state, fit, _, key, _, base = inputs(checkpoint, True)
    _, _, _, phases, model = context(base)
    _, arrays = load_fields(fields, binding(model, fit.observer, state, key))
    obs = copy(fit.observer)
    obs._system = model
    folded, _ = obs.phase_interval_reduce(obs.manage_time_series(phases=phases))
    support, _ = dynamic.phase_crv_symmetry(model, folded)
    positions = model.get_positions_method()(
        input_argument=np.array_split(support, 2)[part], return_nparray=False, calculate_from="phase"
    )
    geometry = {k: WalkerRocheGeometry(model, k) for k in ("primary", "secondary")}
    rows = []
    for p in positions:
        line = utils.rotate_item(np.eye(3), p, model.inclination) @ const.LINE_OF_SIGHT
        if np.sqrt(max(0.0, 1 - line[0] ** 2)) >= 0.5:
            continue
        source, foreground = ("primary", "secondary") if line[0] > 0 else ("secondary", "primary")
        result = shadow_demand(arrays[source], geometry[source], line, rule_for("duffy", 4), geometry[foreground])
        rows.append(dict(phase=float(p.phase), source=source, foreground=foreground, **result))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("fields", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--part", type=int, choices=(0, 1), required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        sampler.write_record(args.output, count(args.checkpoint, args.fields, args.part))
