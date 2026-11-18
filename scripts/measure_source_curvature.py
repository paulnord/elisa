"""Compare intrinsic thermal interpolants with independent Roche gravity on order-five rays."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_nodal_source import _NodalBackend
from elisa.experimental.quadratic_source import QuadraticSourceBackend
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_reflection_area import intrinsic_state


def measure(binary, system, name):
    star = getattr(system, name)
    tri, linear, _, _ = _NodalBackend(name, "linear", device="reference").thermal_nodes(binary, star, star.temperatures)
    _, quadratic, _, control = QuadraticSourceBackend(name, "linear", device="reference").six_nodes(
        binary, star, star.temperatures
    )
    base = int(star.base_symmetry_faces_number)
    nodes, solid_angle_weights = duffy_nodes(tri[:base], 5)
    point, _, jac = ConditionedRocheGeometry(binary, name).sample(nodes)
    w = (solid_angle_weights * jac).reshape(base, 25)
    g = np.linalg.norm(calculate_potential_gradient(1.0, name, point, star.synchronicity, binary.mass_ratio), axis=1)
    target = g.reshape(base, 25) ** star.gravity_darkening
    target *= w.sum() / np.sum(w * target)
    u, v = np.meshgrid((leggauss(5)[0] + 1) / 2, (leggauss(5)[0] + 1) / 2, indexing="ij")
    bary = np.array([(1 - u) * (1 - v), u, (1 - u) * v]).reshape(3, 25)
    p1 = (linear.reshape(base, 3) / star.t_eff) ** 4 @ bary
    t = (quadratic / star.t_eff) ** 4
    p2 = t[:, :3] @ bary
    for i, j in ((0, 1), (1, 2), (2, 0)):
        p2 += (t[:, 3 + i] - (t[:, i] + t[:, j]) / 2)[:, None] * 4 * bary[i] * bary[j]

    def errors(field):
        return dict(
            area_weighted_rms=float(np.sqrt(np.sum(w * (field - target) ** 2) / w.sum())),
            maximum_absolute=float(np.max(np.abs(field - target))),
            signed_area_mean=float(np.sum(w * (field - target)) / w.sum()),
        )

    return dict(
        faces=4 * base,
        independent_quarter_rays=len(nodes),
        linear=errors(p1),
        quadratic=errors(p2),
        order4_cell_T4_closure=control["quadratic_T4_relative_error"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = json.loads(args.baseline.read_bytes())
    props = base["samples"][0]["rows"][0]["physical"]
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        fit, _ = fixture()
        _, binary = prepare_binary(fit, props)
        rows = {}
        for level in (1, 2, 3):
            system, _ = intrinsic_state(binary, level)
            rows[level] = {name: measure(binary, system, name) for name in ("primary", "secondary")}
    args.output.write_text(json.dumps(dict(physical=props, levels=rows), allow_nan=False, indent=2) + "\n")
