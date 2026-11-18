"""Compare reconstructed primary intensities with direct spectra on visible near-contact rays."""

import argparse
from copy import copy
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import const, settings, utils
from elisa.binary_system.curves.utils import prep_surface_params
from elisa.binary_system.surface import faces
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.area_reflection import _BARY
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
from elisa.experimental.gpu_reconstructed_field import shape_rule
from elisa.experimental.gpu_shared_fields import shared_parent_map
from elisa.experimental.primary_cubic import PrimaryCubicLightCurve
from elisa.experimental.receiver_heating import ReceiverHeatingLightCurve
from scripts.compare_irradiating_sources import final_sweep, incoming
from scripts.receiver_phase_parts import context
from scripts.validate_gpu_atmosphere import table_settings


def physical_power(binary, system, triangles, diagnostics, parents, bary, points):
    star = system.primary
    base = star.base_symmetry_faces_number
    geom = ConditionedRocheGeometry(binary, "primary")

    # Numerically integrate the analytic gravity-darkening law on the same curved parents.
    def gravity(points):
        return np.linalg.norm(
            calculate_potential_gradient(1.0, "primary", points, star.synchronicity, binary.mass_ratio), axis=-1
        )

    q, w = duffy_nodes(triangles[:base], 4)
    xyz, _, jac = geom.sample(q)
    weights = (w * jac).reshape(base, 16)
    gpower = gravity(xyz).reshape(base, 16) ** star.gravity_darkening
    norm = weights.sum() / np.sum(weights * gpower)
    fraction = weights / weights.sum(axis=1, keepdims=True)
    mass = fraction @ shape_rule()[0]
    intrinsic = (diagnostics["intrinsic_temperatures"][:base] / star.t_eff) ** 4
    gid = star.face_symmetry_vector[parents]
    g = gravity(points)
    density = intrinsic[gid] + norm * (g**star.gravity_darkening - (fraction * gpower).sum(axis=1)[gid])
    samples, means = diagnostics["reflection_quadrature"]["receiver_deposits"]
    # Solve the linear interpolant in the physical sampling basis, then apply the prescribed positive limiter.
    corners = np.linalg.solve(_BARY, samples.T).T
    slope = corners - (corners * mass).sum(axis=1, keepdims=True)
    lower = -slope.min(axis=1)
    limit = np.minimum(1.0, means / np.where(lower > 0, lower, 1.0))
    limit[lower <= 0] = 1.0
    heat = means[gid] + limit[gid] * np.sum(slope[gid] * bary, axis=1)
    return density, heat, g


def direct_spectrum(binary, system, observer, parents, density, g, mu, bands):
    star = system.primary
    if not np.all(np.isfinite(density) & (density > 0)):
        raise ArithmeticError("Direct primary temperature outside physical domain")

    view = copy(system)
    view.primary, view.secondary = copy(star), copy(system.secondary)
    s = view.primary
    s.temperatures = star.t_eff * density**0.25
    s.log_g = np.log10(g * const.G * binary.primary.mass / binary.semi_major_axis**2)
    s.faces = star.faces[parents]
    s.base_symmetry_faces_number = len(parents)
    s.face_symmetry_vector = np.arange(len(parents))
    prep_surface_params(
        view,
        return_values=False,
        write_to_containers=True,
        passband=observer.passband,
        left_bandwidth=observer.left_bandwidth,
        right_bandwidth=observer.right_bandwidth,
    )
    return np.stack([s.normal_radiance[b] * (1 - (1 - mu) * s.ld_cfs[b][:, 0]) for b in bands], axis=1)


def prepared(base, level):
    fit, _, _, _, model = context(base)
    bands = list(fit.y_data)
    engines = [
        cls(bands, SymmetricInteriorRule(3, 3), device="reference", selective_shadow=True)
        for cls in (ReceiverHeatingLightCurve, PrimaryCubicLightCurve)
    ]
    variant = next(v for v in engines[0].variants if (v.level, v.donor_level) == (level, 2))
    engine = engines[0].engines[variant.key]
    engine.keep_states = True
    system, _ = engine._fields(model, fit.observer)
    diagnostics = engine.last_diagnostics
    for curve in engines:
        curve.engines[variant.key].last_diagnostics = diagnostics
    return fit, model, engines, system, variant, diagnostics


def eclipse_rays(model, tri):
    # One interior test point per full-sphere face. None are thermal or spectral interpolation nodes.
    bary = np.random.default_rng(785401).dirichlet([2.0, 2.0, 2.0], len(tri))
    rays = np.einsum("ni,nij->nj", bary, tri)
    xyz, normal, _ = ConditionedRocheGeometry(model, "primary").sample(rays)
    foreground = ConditionedRocheGeometry(model, "secondary")
    for phase in (0.024, 0.027):
        position = model.get_positions_method()(
            input_argument=np.array([phase]), return_nparray=False, calculate_from="phase"
        )[0]
        line = utils.rotate_item(np.eye(3), position, model.inclination) @ const.LINE_OF_SIGHT
        mu = normal @ line
        inside, distance = foreground.shadow(xyz, line)
        visible = np.flatnonzero((mu > 0) & ~inside)
        order = visible[np.argsort(distance[visible])]
        # Sample both the immediate shadow rim and the rest of the uneclipsed disk.
        ids = np.concatenate((order[:96], order[96:][np.linspace(0, len(order) - 97, 96, dtype=int)]))
        if len(np.unique(ids)) != 192:
            raise ArithmeticError("Expected distinct eclipse samples")
        yield str(phase), ids, bary[ids], rays[ids], xyz[ids], mu[ids]


def compare(base, level):
    fit, model, engines, system, variant, diagnostics = prepared(base, level)
    bands = list(fit.y_data)
    fields = [c._prepare_field(model, system, fit.observer, variant)["primary"] for c in engines]
    tri = fields[0].triangles
    np.testing.assert_array_equal(tri, fields[1].triangles)
    result = {}
    for phase, ids, bary, rays, xyz, mu in eclipse_rays(model, tri):
        intrinsic, heat, gravity = physical_power(model, system, tri, diagnostics["primary"], ids, bary, xyz)
        direct = direct_spectrum(model, system, fit.observer, ids, intrinsic + heat, gravity, mu, bands)
        if not np.all(direct > 0):
            raise ArithmeticError("Expected positive direct spectra")
        values = {}
        for key, field in zip(("quadratic", "cubic"), fields, strict=True):
            moments = field.sample(ids, rays)
            intensity = moments[..., 0] - (1 - mu[:, None]) * moments[..., 1]
            diff = intensity / direct - 1
            values[key] = {
                b: {
                    "rms_relative": float(np.sqrt(np.mean(diff[:, i] ** 2))),
                    "max_relative": float(np.max(np.abs(diff[:, i]))),
                }
                for i, b in enumerate(bands)
            }
        result[str(phase)] = dict(rays=len(ids), nearest_shadow_rays=96, errors=values)
    return result


def native_heat(update, system, parents, xyz, normals):
    receiver, donor, emitted_heat, _ = update
    limits = faces.faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
    rows = np.flatnonzero(faces.visibility_test(xyz, limits["primary"], "primary"))
    rays = receiver._replace(receiver=xyz, normal=normals, parent=system.primary.face_symmetry_vector[parents])
    power = (donor.intrinsic + emitted_heat)[donor.visible_base]
    heat = np.zeros(len(xyz))
    heat[rows] = incoming(rays, rows, donor, power) / system.primary.t_eff**4
    if not np.all(np.isfinite(heat) & (heat >= 0)):
        raise ArithmeticError("Invalid direct thermal irradiation")
    return heat


def shared_levels(base):
    """Exchange intrinsic and deposited power independently on the same visible fine-grid rays."""
    fit, model, coarse_curves, coarse, _, cd = prepared(base, 2)
    _, _, fine_curves, fine, _, fd = prepared(base, 3)
    ct, ft = (
        curves[0].moment_backends["primary"].thermal_triangles(system.primary, 26)
        for curves, system in ((coarse_curves, coarse), (fine_curves, fine))
    )
    parent, _ = shared_parent_map(ft, ct, 1)
    updates = [final_sweep(model, level)["primary"] for level in (2, 3)]
    bands = list(fit.y_data)
    result = {}
    for phase, ids, bary, rays, xyz, mu in eclipse_rays(model, ft):
        ci = parent[ids]
        cb = np.linalg.solve(ct[ci].transpose(0, 2, 1), rays[..., None])[..., 0]
        cb /= cb.sum(axis=1, keepdims=True)
        if np.any(cb < 0):
            raise ArithmeticError("Held-out ray escaped its thermal parent")
        a, h, g = physical_power(model, coarse, ct, cd["primary"], ci, cb, xyz)
        b, k, same_g = physical_power(model, fine, ft, fd["primary"], ids, bary, xyz)
        np.testing.assert_array_equal(g, same_g)
        _, normals, _ = ConditionedRocheGeometry(model, "primary").sample(rays)
        hc = native_heat(updates[0], coarse, ci, xyz, normals)
        hf = native_heat(updates[1], fine, ids, xyz, normals)
        native_change = hf - hc
        residual_change = (k - hf) - (h - hc)
        np.testing.assert_allclose(native_change + residual_change, k - h, rtol=4e-15, atol=1e-18)

        def optical(power):
            return direct_spectrum(model, fine, fit.observer, ids, power, g, mu, bands)

        low, replace_intrinsic, replace_heat, high = [optical(power) for power in (a + h, b + h, a + k, b + k)]
        # Both orders retain the local nonlinearity of Planck emission and LD interpolation.
        intrinsic = ((replace_intrinsic - low) + (high - replace_heat)) / (2 * low)
        heating = ((replace_heat - low) + (high - replace_intrinsic)) / (2 * low)
        total = (high - low) / low
        np.testing.assert_allclose(intrinsic + heating, total, rtol=5e-15, atol=2e-16)
        result[phase] = {
            "rays": len(ids),
            "rms_heat_over_Teff4": {
                name: float(np.sqrt(np.mean(diff**2)))
                for name, diff in (
                    ("coarse_reconstructed_minus_native", h - hc),
                    ("fine_reconstructed_minus_native", k - hf),
                    ("native_change", native_change),
                    ("change_in_reconstruction_residual", residual_change),
                )
            },
            "rms_delta_T4_over_Teff4": {
                name: float(np.sqrt(np.mean(diff**2))) for name, diff in (("intrinsic", b - a), ("deposited", k - h))
            },
            "rms_relative_intensity": {
                band: {
                    name: float(np.sqrt(np.mean(diff[:, j] ** 2)))
                    for name, diff in (("intrinsic", intrinsic), ("deposited", heating), ("total", total))
                }
                for j, band in enumerate(bands)
            },
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--level", type=int, choices=(2, 3))
    mode.add_argument("--shared-levels", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        base = json.loads(args.baseline.read_bytes())
        result = shared_levels(base) if args.shared_levels else compare(base, args.level)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
