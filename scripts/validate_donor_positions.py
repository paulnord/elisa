"""Measure the residual from sharing an emitting position after angular integration.

For a sampled subset of receiving rays only, expand every original donor, compare
the exact receiver-horizon sign within each group and test the first spatial
derivative. This diagnostic does not approximate a production likelihood.
"""

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
from elisa.experimental.angular_reflection import AngularReflection, _cpu_incident, _moments
from scripts.validate_emitter_groups import incident
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision
from scripts.validate_reflection_area import intrinsic_state


def evaluate_samples(receiver, native, grouped, rows, law, heat):
    if law not in ("linear", "cosine"):
        raise ValueError("Spatial derivative requires a linear angular law")
    group = grouped.group
    centre = grouped.emitter[group]
    offset = native.emitter - centre
    energy = (native.intrinsic + heat)[native.visible_base] * native.visible_area
    coeff = native.prefix_ld[0]
    truth = incident(receiver, rows, native.emitter, native.outward, energy, native.prefix_ld)
    predicted, resolved, taylor = [], [], []
    total_unstable = 0
    for row in rows:
        recv, h = receiver.receiver[row], receiver.normal[row]
        delta = recv - centre
        dist = np.linalg.norm(delta, axis=1)
        u = delta / dist[:, None]
        b = np.sum(native.outward * u, axis=1)
        mu = -(u @ h)
        angular = np.maximum(b, 0) * ((1 - coeff) + coeff * np.maximum(b, 0))
        common = np.maximum(mu, 0) * angular / dist**2
        # Every group with any member crossing the receiver tangent plane
        # between its native and common source locations gets explicit geometry.
        native_rays = recv - native.emitter
        native_dist = np.linalg.norm(native_rays, axis=1)
        native_u = native_rays / native_dist[:, None]
        cos_r = -(native_u @ h)
        cos_d = np.maximum(np.sum(native.outward * native_u, axis=1), 0)
        exact = np.maximum(cos_r, 0) * cos_d * (1 - coeff + coeff * cos_d) / native_dist**2
        cross = (cos_r > 0) != (mu > 0)
        unstable = np.bincount(group, weights=cross, minlength=len(grouped.emitter)) > 0
        total_unstable += int(unstable.sum())
        local = np.where(unstable[group], exact, common)
        # First derivative of mu_r * K(b) / r^2 w.r.t. the emitter displacement.
        # K(b)=(1-a)b + a*b^2; both positive-part branches must be stable.
        kprime = 1 - coeff + 2 * coeff * b
        u_offset = np.sum(u * offset, axis=1)
        correction = (
            mu * (3 * angular + b * kprime) * u_offset
            + angular * (offset @ h)
            - mu * kprime * np.sum(native.outward * offset, axis=1)
        ) / dist**3
        correction = np.where((b > 0) & (mu > 0) & ~unstable[group], correction, 0)
        absorb = receiver.absorption[receiver.parent[row]]
        predicted.append(absorb * (common @ energy))
        resolved.append(absorb * (local @ energy))
        taylor.append(absorb * ((local + correction) @ energy))
        np.testing.assert_allclose(
            absorb * (exact @ energy), truth[len(predicted) - 1], rtol=3e-13, atol=1e-15 * receiver.intrinsic.max()
        )
    moments = (
        _cpu_incident(
            receiver.receiver[rows],
            receiver.normal[rows],
            grouped.emitter,
            grouped.outward,
            _moments(np, grouped, heat),
        )
        * receiver.absorption[receiver.parent[rows]]
    )
    np.testing.assert_allclose(moments, predicted, rtol=3e-13, atol=1e-15 * receiver.intrinsic.max())
    return {
        key: np.asarray(x) - truth
        for key, x in (
            ("shared_position", predicted),
            ("resolved_receiver_horizon", resolved),
            ("first_order_with_resolved_horizon", taylor),
        )
    }, total_unstable


def compare(profile):
    if (
        profile["engine"] != "angular-reflection"
        or max(profile["levels"]) != 3
        or profile["source_revision"] != asset_revision()
    ):
        raise ValueError("Use a matching field-3 angular reflection profile")
    props = profile["samples"][0]["rows"][0]["physical"]
    fit = prepare_fit(
        make_problem(jd=profile["jd"], free_nuisance=True, shape_parameters=True),
        phase_mode="observations",
        mesh_mode="paired",
    )
    _, binary = prepare_binary(fit, props)
    system, _ = intrinsic_state(binary, 3)
    limits = faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
    records = []
    for cap in (1, 2):
        op = AngularReflection(settings.LIMB_DARKENING_LAW, device="reference", donor_level=cap)
        names = ("primary", "secondary")
        native = [op._surface(binary, system, name, 3, limits[name]) for name in names]
        groups, _ = op._prepare_pair(native, system, limits, 3)
        for i, name in enumerate(names):
            r, d, g = native[i], native[1 - i], groups[1 - i]
            rows = np.linspace(0, len(r.parent) - 1, 128, dtype=int)
            errors, unstable = evaluate_samples(r, d, g, rows, op.law, np.zeros_like(d.intrinsic))
            scale = getattr(system, name).t_eff ** 4
            records.append(
                dict(
                    receiver=name,
                    cap=cap,
                    receiver_samples=len(rows),
                    candidate_group_rays=len(rows) * len(g.emitter),
                    receiver_horizon_group_rays=unstable,
                    rms_T4_over_Teff4={
                        k: float(np.linalg.norm(v / scale) / np.sqrt(len(rows))) for k, v in errors.items()
                    },
                )
            )
    return dict(
        physical=props,
        rows=records,
        contract="First update on sampled receiving rays with original native donor luminosities. The derivative and receiver-side opening are local diagnostics.",
    )


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
