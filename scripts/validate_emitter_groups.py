"""Isolate emitter position and normal errors in the first reflection update."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import ld, settings, utils
from elisa.binary_system.surface.faces import faces_visibility_x_limits
from elisa.experimental.area_reflection import AreaReflection, _emission
from elisa.numba_functions import reflection_effect
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision
from scripts.validate_reflection_area import intrinsic_state


def incident(receiver, rows, position, normal, light, coeff):
    values = []
    for start in range(0, len(rows), 16):
        sel = rows[start : start + 16]
        d, join = utils.calculate_distance_matrix(receiver.receiver[sel], position, return_join_vector_matrix=True)
        ga = np.maximum(0, reflection_effect.gamma_primary(receiver.normal[sel], join))
        gb = np.maximum(0, reflection_effect.gamma_secondary(normal, join))
        weight = ld.limb_darkening_factor(
            coefficients=coeff.T, cos_theta=gb.T, limb_darkening_law=settings.LIMB_DARKENING_LAW
        ).T
        values.extend(receiver.absorption[receiver.parent[sel]] * ((ga * gb / d**2 * weight) @ light))
    return np.array(values)


def compare(profile):
    if profile["engine"] != "area-reflection" or profile["source_revision"] != asset_revision():
        raise ValueError("Supply an area-reflection profile with the current fixture assets")
    props = profile["samples"][0]["rows"][0]["physical"]
    fit = prepare_fit(
        make_problem(jd=profile["jd"], free_nuisance=True, shape_parameters=True),
        phase_mode="observations",
        mesh_mode="paired",
    )
    _, binary = prepare_binary(fit, props)
    level = max(profile["levels"])
    if level != 3:
        raise ValueError("Use the fine level-3 profile")
    system, _ = intrinsic_state(binary, level)
    limits = faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
    names = ("primary", "secondary")
    grids = {}
    for cap in (1, 2, 3):
        solver = AreaReflection(settings.LIMB_DARKENING_LAW, device="reference", donor_level=cap)
        grids[cap] = [solver._surface(binary, system, n, level, limits[n]) for n in names]
    result = []
    for i, name in enumerate(names):
        receiver, donor = grids[3][i], grids[3][1 - i]
        # Deterministic spatial sample across the selected quarter; this RMS is per sampled point.
        rows = np.linspace(0, len(receiver.parent) - 1, 128, dtype=int)
        native = donor.intrinsic[donor.visible_base] * donor.visible_area
        baseline = incident(receiver, rows, donor.emitter, donor.outward, native, donor.prefix_ld)
        for cap in (1, 2):
            g = grids[cap][1 - i]
            if not np.array_equal(g.visible_base, donor.visible_base):
                raise ValueError("Changed emission set")
            light, coeff = _emission(np, g, np.zeros_like(g.intrinsic))
            coarse = incident(receiver, rows, g.emitter, g.outward, light, coeff)
            expanded = incident(receiver, rows, g.emitter[g.group], g.outward[g.group], native, donor.prefix_ld)
            np.testing.assert_allclose(coarse, expanded, rtol=3e-13, atol=1e-13 * getattr(system, name).t_eff ** 4)
            position = incident(receiver, rows, g.emitter[g.group], donor.outward, native, donor.prefix_ld)
            normals = incident(receiver, rows, donor.emitter, g.outward[g.group], native, donor.prefix_ld)
            changes = {
                "position_only": position - baseline,
                "normal_only": normals - baseline,
                "both": coarse - baseline,
                "mixed": coarse - position - normals + baseline,
            }
            result.append(
                {
                    "receiver": name,
                    "donor_cap": cap,
                    "sample_size": len(rows),
                    "rms_T4_over_Teff4": {
                        k: float(np.sqrt(np.mean((v / getattr(system, name).t_eff ** 4) ** 2)))
                        for k, v in changes.items()
                    },
                }
            )
    return {
        "physical": props,
        "rows": result,
        "contract": "Fixed intrinsic luminosities, coefficients, emitting faces and 128 ordered receiving samples. "
        "Position-only and normal-only perturbations retain individual donor weights. "
        "This diagnoses the first update before observation mapping.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.profile.read_bytes()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = compare(json.loads(raw))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
