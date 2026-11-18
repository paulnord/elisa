"""Resolve the remaining donor-position error on fixed receiving rays after two reflection sweeps."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits
from elisa.experimental.emitting_area import EmittingAreaReflection
from scripts.validate_emitter_groups import incident
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision
from scripts.validate_reflection_area import NAMES, intrinsic_state


def residuals(receiver, donor, heat, rows, scale):
    plan = donor.spatial
    index = donor.group
    centre = plan.centres[index]
    offset = plan.offset
    n = donor.native_normals
    a = donor.prefix_ld[0]
    light = (donor.intrinsic + heat)[donor.visible_base] * donor.visible_area
    target = incident(receiver, rows, plan.native_positions, n, light, donor.prefix_ld) / scale
    change = []
    for row in rows:
        h = receiver.normal[row]
        ray = receiver.receiver[row] - centre
        r = np.linalg.norm(ray, axis=1)
        u = ray / r[:, None]
        mu, b = -(u @ h), np.sum(u * n, axis=1)
        t = np.sum(u * offset, axis=1) / r
        s2 = np.sum(offset * offset, axis=1) / r**2
        H = (offset @ h) / r
        N = np.sum(offset * n, axis=1) / r
        # Expand (mu+H)(b-N)^k (1-2t+s2)^(-(k+3)/2) through degree two.
        # k=1 and k=2 have inverse-distance exponents 2 and 5/2, respectively.
        A0, A1, A2 = mu * b, H * b - mu * N, -H * N
        Q0, Q1, Q2 = mu * b * b, H * b * b - 2 * mu * b * N, mu * N * N - 2 * H * b * N
        base = (1 - a) * A0 + a * Q0
        first = (1 - a) * (A1 + 4 * t * A0) + a * (Q1 + 5 * t * Q0)
        second = (1 - a) * (A2 + 4 * t * A1 + (12 * t * t - 2 * s2) * A0) + a * (
            Q2 + 5 * t * Q1 + (17.5 * t * t - 2.5 * s2) * Q0
        )
        lit = np.unpackbits(plan.lit[row], bitorder="little")[index].astype(bool)
        start, stop = np.searchsorted(plan.horizon_row, [row, row + 1])
        opened = plan.horizon_face[start:stop]
        r0 = receiver.receiver[row] - plan.native_positions[opened]
        dist = np.linalg.norm(r0, axis=1)
        u0 = r0 / dist[:, None]
        br = np.maximum(np.sum(n[opened] * u0, axis=1), 0)
        exact = np.sum(light[opened] * np.maximum(-(u0 @ h), 0) * br * (1 - a[opened] + a[opened] * br) / dist**2)
        weight = light[lit] / r[lit] ** 2
        absorb = receiver.absorption[receiver.parent[row]] / scale
        change.append([absorb * (exact + weight @ v[lit]) for v in (base, base + first, base + first + second)])
    delta = np.array(change) - target[:, None]
    return {
        k: dict(rms=float(np.sqrt(np.mean(d * d))), maximum=float(np.max(abs(d))))
        for k, d in zip(("shared_position", "first_order", "second_order"), delta.T, strict=True)
    }


def compare(profile):
    if (
        profile["engine"] != "emitting-area"
        or profile["levels"] != [2, 3]
        or profile["source_revision"] != asset_revision()
    ):
        raise ValueError("Expected a matching field-2/3 curved emitting-area profile")
    props = profile["samples"][0]["rows"][0]["physical"]
    fit = prepare_fit(
        make_problem(jd=profile["jd"], free_nuisance=True, shape_parameters=True),
        phase_mode="observations",
        mesh_mode="paired",
    )
    _, binary = prepare_binary(fit, props)
    system, _ = intrinsic_state(binary, 3)
    limits = faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
    results = []
    for cap in (1, 2):
        op = EmittingAreaReflection(settings.LIMB_DARKENING_LAW, device="reference", donor_level=cap)
        native = [op._surface(binary, system, name, 3, limits[name]) for name in NAMES]
        surfaces, _ = op._prepare_pair(native, system, limits, 3)
        heat = [np.zeros_like(s.intrinsic) for s in surfaces]
        for sweep in range(2):
            for i in (0, 1) if system.primary.t_eff <= system.secondary.t_eff else (1, 0):
                receiver, donor = surfaces[i], surfaces[1 - i]
                if sweep == 1:
                    rows = np.linspace(0, len(receiver.parent) - 1, 96, dtype=int)
                    measured = residuals(receiver, donor, heat[1 - i], rows, getattr(system, NAMES[i]).t_eff ** 4)
                    results.append(dict(receiver=NAMES[i], cap=cap, rows=len(rows), deposited_T4_over_Teff4=measured))
                heat[i] = op._reference_update(receiver, donor, heat[1 - i])
    return dict(
        physical=props,
        rows=results,
        contract="At the final cooler-first update, compare the same donor luminosities and three curved nodes per original face. The target expands every native node on 96 fixed receiving rays. The quadratic positional correction is a local diagnostic.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    raw = args.profile.read_bytes()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = compare(json.loads(raw))
    result["profile_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
