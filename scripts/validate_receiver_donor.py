"""Cross donor grids at fixed receiving points on the finite reflected fitting fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import ld, settings, utils
from elisa.binary_system.surface import faces
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import ReceiverSourceLightCurve
from elisa.experimental.gpu_shared_fields import shared_parent_map
from elisa.numba_functions import reflection_effect as reflection
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_selective import prepare_binary


def incident(receiver: dict, donor: dict, name: str) -> np.ndarray:
    """Original last update at the supplied receiver and donor resolutions."""
    other = "secondary" if name == "primary" else "primary"
    a = getattr(receiver["system"], name)
    b = getattr(donor["system"], other)
    n = a.base_symmetry_faces_number
    hot = receiver["system"].primary.t_eff > receiver["system"].secondary.t_eff
    # A tie uses the original primary-first order.
    visits_last = (name == "primary") == hot
    donor_t = b.temperatures if visits_last else getattr(donor["previous"], other).temperatures
    visible = donor["visible"][other]
    vector = (donor_t[visible] / a.t_eff) ** 4 * b.areas[visible]
    rows = np.flatnonzero(receiver["visible"][name][:n])
    coeff = donor["ld"][other][:, : np.count_nonzero(visible)].T
    integral = ld.calculate_integrated_limb_darkening_factor(settings.LIMB_DARKENING_LAW, receiver["ld"][name])
    out = np.zeros(n)
    for start in range(0, len(rows), 32):
        selected = rows[start : start + 32]
        distance, join = utils.calculate_distance_matrix(
            a.face_centres[selected], b.face_centres[visible], return_join_vector_matrix=True
        )
        ga = np.maximum(reflection.gamma_primary(a.normals[selected], join), 0)
        gb = np.maximum(reflection.gamma_secondary(b.normals[visible], join), 0)
        angular = ld.limb_darkening_factor(
            coefficients=coeff, cos_theta=gb.T, limb_darkening_law=settings.LIMB_DARKENING_LAW
        ).T
        out[selected] = (a.albedo / integral[selected]) * ((ga * gb * angular / distance**2) @ vector)
    return out


def build(engine: Any, binary: Any, observer: Any) -> dict:
    engine.keep_states = True
    with settings.override(REFLECTION_EFFECT_ITERATIONS=1):
        previous, _ = engine._fields(binary, observer)
    system, cones = engine._fields(binary, observer)
    xlim = faces.faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
    grid = {"system": system, "previous": previous, "cones": cones, "visible": {}, "ld": {}, "heat": {}}
    for name in ("primary", "secondary"):
        a = getattr(system, name)
        initial = engine.last_diagnostics[name]["intrinsic_temperatures"]
        grid["heat"][name] = ((a.temperatures / a.t_eff) ** 4 - (initial / a.t_eff) ** 4)[
            : a.base_symmetry_faces_number
        ]
        grid["ld"][name] = ld.get_bolometric_ld_coefficients(
            initial, a.log_g, a.metallicity, a.limb_darkening_coefficients
        )
        grid["visible"][name] = faces.get_visibility_tests(
            a.face_centres, xlim[name], name, system.morphology, q_test=False
        )[0]
    return grid


def compare(profile: dict) -> dict:
    props = profile["samples"][0]["rows"][0]["physical"]
    fit = prepare_fit(
        make_problem(jd=profile["jd"], free_nuisance=True, shape_parameters=True),
        phase_mode="observations",
        mesh_mode="paired",
    )
    _, binary = prepare_binary(fit, props)
    engine = ReceiverSourceLightCurve(
        list(fit.y_data),
        CurvedEdgeRule(3, 1),
        device="reference",
        levels=tuple(profile["levels"]),
        integration_level=profile["integration_level"],
    )
    low, high = [build(engine.engines[v.key], binary, fit.observer) for v in engine.variants]
    result = {}
    for name in ("primary", "secondary"):
        coarse, fine = low["heat"][name], high["heat"][name]
        parent, _ = shared_parent_map(
            high["cones"][name][: len(fine)],
            low["cones"][name][: len(coarse)],
            profile["levels"][1] - profile["levels"][0],
        )
        validation = {}
        for key, grid in [("coarse", low), ("fine", high)]:
            check = incident(grid, grid, name) - grid["heat"][name]
            np.testing.assert_allclose(check, 0.0, atol=3e-13, rtol=0)
            validation[key] = float(np.max(np.abs(check)))
        at_fine = incident(high, low, name)
        weight = getattr(high["system"], name).areas[: len(fine)]
        changes = {"receiver": at_fine - coarse[parent], "donor": fine - at_fine, "total": fine - coarse[parent]}
        result[name] = {
            "receiving_points": len(fine),
            "original_deposit_max_absolute_error": validation,
            "area_weighted": {
                k: {
                    "signed_mean": float(np.average(v, weights=weight)),
                    "rms": float(np.sqrt(np.average(v**2, weights=weight))),
                    "maximum_absolute": float(abs(v).max()),
                }
                for k, v in changes.items()
            },
        }
    return {
        "physical": props,
        "levels": profile["levels"],
        "component": result,
        "unit": "deposited T^4 / receiver Teff^4",
        "contract": "Receiver change includes geometry, visibility and LD divisor. Donor change includes angular sampling, areas, intrinsic temperatures, last-iteration state and original bolometric LD ordering. Evaluated before passband and observation mapping.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        data = args.profile.read_bytes()
        report = compare(json.loads(data))
    report["input_sha256"] = hashlib.sha256(data).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
