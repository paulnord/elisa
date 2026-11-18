"""Compare iterated finite receiver means with a 64-node CPU area integral at fixed donors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import Observer, ld, settings, utils
from elisa.binary_system.surface import faces
from elisa.experimental.area_reflection import AreaReflection
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import ReceiverSourceLightCurve
from elisa.numba_functions import reflection_effect
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

NAMES = ("primary", "secondary")


def intrinsic_state(binary, level):
    bands = ["Generic.Bessell.V", "bolometric"]
    instance = ReceiverSourceLightCurve(
        bands, CurvedEdgeRule(3, 1), device="reference", levels=(level,), integration_level=level
    )
    with settings.override(REFLECTION_EFFECT=False):
        return instance.engines[instance.variants[0].key]._fields(binary, Observer(passband=bands, system=binary))


def direct(binary, system, triangles):
    stars = [getattr(system, name) for name in NAMES]
    limits = faces.faces_visibility_x_limits(stars[0].polar_radius, stars[1].polar_radius, 1.0)
    coeff = [
        ld.get_bolometric_ld_coefficients(s.temperatures, s.log_g, s.metallicity, s.limb_darkening_coefficients)
        for s in stars
    ]
    matrix, indices = [], []
    for i, name in enumerate(NAMES):
        a, b = stars[i], stars[1 - i]
        n = a.base_symmetry_faces_number
        rays, weights = duffy_nodes(triangles[name][:n], 8)
        points, normals, jac = ConditionedRocheGeometry(binary, name).sample(rays)
        area = (jac * weights).reshape(n, 64)
        fractions = (area / area.sum(axis=1)[:, None]).ravel()
        vis = np.flatnonzero(faces.visibility_test(b.face_centres, limits[NAMES[1 - i]], NAMES[1 - i]))
        rows = np.flatnonzero(faces.visibility_test(points, limits[name], name))
        coupling = np.zeros((n, len(vis)))
        for start in range(0, len(rows), 32):
            sample = rows[start : start + 32]
            distance, join = utils.calculate_distance_matrix(
                points[sample], b.face_centres[vis], return_join_vector_matrix=True
            )
            ga = np.maximum(reflection_effect.gamma_primary(normals[sample], join), 0)
            gb = np.maximum(reflection_effect.gamma_secondary(b.normals[vis], join), 0)
            direction = ld.limb_darkening_factor(
                coefficients=coeff[1 - i][:, : len(vis)].T,
                cos_theta=gb.T,
                limb_darkening_law=settings.LIMB_DARKENING_LAW,
            ).T
            np.add.at(coupling, sample // 64, (fractions[sample, None] * ga * gb / distance**2) * direction)
        absorb = a.albedo / ld.calculate_integrated_limb_darkening_factor(settings.LIMB_DARKENING_LAW, coeff[i])[:n]
        matrix.append(absorb[:, None] * coupling * b.areas[vis])
        indices.append(b.face_symmetry_vector[vis])
    t4 = [s.temperatures[: s.base_symmetry_faces_number] ** 4 for s in stars]
    heat = [np.zeros_like(v) for v in t4]
    for _ in range(2):
        for i in [0, 1] if stars[0].t_eff <= stars[1].t_eff else [1, 0]:
            heat[i] = matrix[i] @ (heat[1 - i] + t4[1 - i])[indices[i]]
    return heat


def compare(atlas, level):
    if level not in (0, 1):
        raise ValueError("Use diagnostic levels 0 or 1")
    case = {"discretization": 12, "inclination": 85, "spin": [0.85, 1.15]}
    binary = (table_fixture if atlas == "ck04" else fixture)(case)
    system, triangles = intrinsic_state(binary, level)
    truth = direct(binary, system, triangles)
    rows = []
    for count in (1, 3):
        temperatures, _ = AreaReflection(
            settings.LIMB_DARKENING_LAW, device="reference", receiver_points=count, donor_level=level
        ).evaluate(binary, system, level, 2)
        for i, name in enumerate(NAMES):
            star = getattr(system, name)
            n = star.base_symmetry_faces_number
            delta = ((temperatures[name][:n] ** 4 - star.temperatures[:n] ** 4) - truth[i]) / star.t_eff**4
            rows.append(
                {
                    "atlas": atlas,
                    "level": level,
                    "component": name,
                    "receiver_points": count,
                    "rms_error": float(np.sqrt(np.average(delta**2, weights=star.areas[:n]))),
                    "maximum_error": float(abs(delta).max()),
                    "quarter_faces": n,
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (
        table_settings(),
        settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2, USE_SINGLE_LD_COEFFICIENTS=False),
    ):
        values = [row for atlas in ("bb", "ck04") for level in (0, 1) for row in compare(atlas, level)]
    args.output.write_text(
        json.dumps(
            {"unit": "deposited T^4 / Teff^4", "reference_receiver_points": 64, "rows": values},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
