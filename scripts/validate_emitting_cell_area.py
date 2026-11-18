"""Test emitting-cell quadrature on identical receiving rays at field levels two and three."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.binary_system.surface import faces
from elisa.experimental.area_reflection import AreaReflection, _BARY
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from scripts.validate_emitter_groups import incident
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision
from scripts.validate_reflection_area import NAMES, intrinsic_state


def projected_rule(binary, name, triangles, order):
    if order == 3:
        # The constant triangle determinant cancels when normalizing within a face.
        u = _BARY @ triangles
        rays = u.reshape(-1, 3)
        solid_angle = np.linalg.norm(rays, axis=1) ** -3
    else:
        rays, solid_angle = duffy_nodes(triangles, 4)
    pos, normals, jac = ConditionedRocheGeometry(binary, name).sample(rays)
    mass = (solid_angle * jac).reshape(len(triangles), order)
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ArithmeticError("Invalid curved area in emitting-cell diagnostic")
    return pos, normals, mass / mass.sum(axis=1)[:, None]


def fields(binary, level):
    system, triangles = intrinsic_state(binary, level)
    limit = faces.faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
    op = AreaReflection(settings.LIMB_DARKENING_LAW, device="reference", donor_level=3)
    return {
        name: (
            op._surface(binary, system, name, level, limit[name]),
            triangles[name][faces.visibility_test(getattr(system, name).face_centres, limit[name], name)],
        )
        for name in NAMES
    }


def compare(profile):
    if (
        profile["engine"] != "spatial-reflection"
        or profile["levels"] != [2, 3]
        or profile["source_revision"] != asset_revision()
        or not profile["irradiation"]
    ):
        raise ValueError("Use a matching irradiated field-2/3 spatial profile")
    p = profile["samples"][0]["rows"][0]["physical"]
    fit = prepare_fit(
        make_problem(jd=profile["jd"], free_nuisance=True, shape_parameters=True),
        phase_mode="observations",
        mesh_mode="paired",
    )
    _, binary = prepare_binary(fit, p)
    grids = [fields(binary, level) for level in [2, 3]]
    results = []
    for name, other in zip(NAMES, reversed(NAMES), strict=True):
        recv = grids[1][name][0]
        rows = np.linspace(0, len(recv.receiver) - 1, 96, dtype=int)
        scale = getattr(binary, name).t_eff ** 4
        values = {}
        for level, grid in zip([2, 3], grids, strict=True):
            donor, tri = grid[other]
            energy = donor.visible_area * donor.intrinsic[donor.visible_base]
            values[level, 1] = incident(recv, rows, donor.emitter, donor.outward, energy, donor.prefix_ld) / scale
            for order in (3, 16):
                total = np.zeros(len(rows))
                for start in range(0, len(tri), 256):
                    part = slice(start, start + 256)
                    pos, normal, mass = projected_rule(binary, other, tri[part], order)
                    light = (energy[part, None] * mass).ravel()
                    coeff = np.repeat(donor.prefix_ld[:, part], order, axis=1)
                    total += incident(recv, rows, pos, normal, light, coeff)
                values[level, order] = total / scale

        def rms(x):
            return float(np.linalg.norm(x) / np.sqrt(len(x)))

        results.append(
            dict(
                receiver=name,
                receiver_samples=len(rows),
                level2_to_level3_rms={str(n): rms(values[2, n] - values[3, n]) for n in (1, 3, 16)},
                three_to_sixteen_rms={str(k): rms(values[k, 3] - values[k, 16]) for k in (2, 3)},
                centre_to_sixteen_rms={str(k): rms(values[k, 1] - values[k, 16]) for k in (2, 3)},
            )
        )
    return dict(
        physical=p,
        rows=results,
        contract="First irradiation update. The fine receiver position, normal and absorption are fixed across comparisons. Each original visible emitting face keeps its full luminosity and bolometric LD. Only its emitting positions and directions change. RMS is deposited T^4 divided by the receiving star's Teff^4.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.profile.read_bytes()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = compare(json.loads(raw))
    result["profile_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
