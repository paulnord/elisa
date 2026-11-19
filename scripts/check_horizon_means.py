"""Check primary receiver means against individually traced donor nodes at the illumination boundary."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from elisa import settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits
from elisa.experimental.area_reflection import _BARY
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.horizon_receiver import HorizonReceiverReflection, _SIX
from scripts.check_primary_spectrum import native_heat
from scripts.receiver_phase_parts import context
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_reflection_area import intrinsic_state


def check(model, level):
    system, triangles = intrinsic_state(model, level)
    solver = HorizonReceiverReflection("linear", device="reference", donor_level=2)
    sampled = solver._sampled_update
    calls = deque(maxlen=2)

    def trace(receiver, donor, deposited):
        values, mean = sampled(receiver, donor, deposited)
        calls.append((receiver, donor, deposited, mean))
        return values, mean

    solver._sampled_update = trace
    _, stats = solver.evaluate(model, system, level, 2)
    order = sorted(("primary", "secondary"), key=lambda name: getattr(system, name).t_eff)
    update = dict(zip(order, calls, strict=True))["primary"]
    s = system.primary
    n = s.base_symmetry_faces_number
    limit = faces_visibility_x_limits(s.polar_radius, system.secondary.polar_radius, 1.0)["primary"]
    x = s.points[s.faces[:n], 0]
    eligible = np.flatnonzero((x.max(axis=1) >= limit) &
                              (x.min(axis=1) <= limit + 2 * s.polar_radius * system.secondary.polar_radius))
    sorted_ids = eligible[np.argsort(x[eligible].mean(axis=1), kind="stable")]
    ids = sorted_ids[np.linspace(0, len(sorted_ids)-1, min(32, len(sorted_ids)), dtype=int)]
    tri = triangles["primary"][ids]
    geom = ConditionedRocheGeometry(model, "primary")
    means = {"solver": stats["primary"]["receiver_deposits"][1][ids]}
    coords = _BARY @ tri
    weights = np.abs(np.linalg.det(tri))[:, None] / (6 * np.linalg.norm(coords, axis=2)**3)
    for label, (rays, solid) in (("three", (coords.reshape(-1, 3), weights.ravel())),
                                  ("six", _SIX.interior_nodes(tri)),
                                  ("sixteen", duffy_nodes(tri, 4)), ("thirty_six", duffy_nodes(tri, 6))):
        xyz, normals, jac = geom.sample(rays)
        area = (solid * jac).reshape(len(ids), -1)
        heat = native_heat(update, system, np.repeat(ids, area.shape[1]), xyz, normals).reshape(area.shape)
        means[label] = np.sum(area*heat, axis=1) / area.sum(axis=1)
    return dict(face_ids=ids, faces_in_band=len(eligible), emitting_nodes=len(update[1].visible_area), means=means)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        _, props, _, _, model = context(json.loads(args.baseline.read_bytes()))
        record = dict(physical=props, levels={level: check(model, level) for level in (2, 3)})
    args.output.write_text(json.dumps(record, default=lambda a: a.tolist(), allow_nan=False) + "\n")
