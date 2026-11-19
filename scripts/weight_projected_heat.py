"""Project changes in common-face heating through blackbody passbands and linear limb darkening."""

import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np

from elisa import const, settings, utils
from elisa.binary_system import dynamic
from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
from elisa.experimental.gpu_reconstructed_field import reference_barycentric
from elisa.experimental.gpu_shared_fields import shared_parent_map
from elisa.experimental.horizon_receiver import HorizonReceiverLightCurve
from elisa.experimental.primary_cubic import CUBIC_NODES, CurvedHeatingLightCurve, lagrange
from scripts.compare_common_parent_heat import means_on_common_faces, thermal_field
from scripts.receiver_phase_parts import context
from scripts.validate_gpu_atmosphere import table_settings


def disk_weights(field, parents, rays, mu, area):
    """Integrate the signed Lagrange basis for I and I*a, where I(mu) = I - (1-mu)*I*a."""
    weight = np.zeros((len(field.vertex), 2))
    for first in range(0, len(parents), 65536):
        sl = slice(first, first + 65536)
        ids = parents[sl]
        bary, _ = reference_barycentric(field.triangles, field.arrays[1], ids, rays[sl])
        basis_area = lagrange(np, bary) * (area[sl] * mu[sl])[:, None]
        nodes = field.faces[ids].ravel()
        for column, values in enumerate((basis_area, (mu[sl, None] - 1) * basis_area)):
            weight[:, column] += np.bincount(nodes, weights=values.ravel(), minlength=len(weight))
    return weight


def local_response(solver, model, observer, system, key, dh):
    engine = solver.engines[key]
    data, diagnostics = {}, {}
    for name, backend in solver.moment_backends.items():
        data[name], diagnostics[name] = solver._source_data(backend, model, getattr(system, name), engine.last_diagnostics[name])
    tri, t, g, checks = data["primary"]
    teff = system.primary.t_eff
    fine = (t / teff)**4

    def spectra(density):
        if not np.all(np.isfinite(density) & (density > 0)):
            raise ValueError("Invalid T^4 at a common optical node")
        data["primary"] = (tri, teff * density**0.25, g, checks)
        return solver._optical_fields(system, observer, engine, data, diagnostics)["primary"].vertex

    # The transferred coarse heat uses fine gravity, geometry, passbands and LD tables.
    old_heat = spectra(fine - dh)
    middle = fine - 0.5 * dh
    slopes = [(spectra(middle + h) - spectra(middle - h)) / (2*h) for h in (1e-5, 5e-6)]
    return old_heat, slopes


def measure(base, horizon_mean):
    fit, _, _, phases, model = context(base)
    observer = copy(fit.observer)
    observer._system = model
    cls = HorizonReceiverLightCurve if horizon_mean else CurvedHeatingLightCurve
    solver = cls(list(fit.y_data), SymmetricInteriorRule(3, 4, slice_order=12, junction_depth=4),
                 device="reference", selective_shadow=True)
    solver.variants = tuple(v for v in solver.variants if v.donor_level == 2)
    common, triangles, systems, maps, _ = solver._prepare(model, observer)
    pg, sg = (solver._geometry(model, name) for name in ("primary", "secondary"))
    optical, heat, ancestry = [], [], []
    for level in (2, 3):
        key = f"curvature_field{level}_donor2"
        field = solver._reconstruction[key, "primary"]
        optical.append(field)
        deposits = solver.engines[key].last_diagnostics["primary"]["reflection_quadrature"]["receiver_deposits"]
        heat.append(thermal_field(pg, systems[key].primary, field.triangles, deposits))
        ancestry.append(maps[key]["primary"])
    coarse, fine = heat
    parent, _ = shared_parent_map(fine.triangles, coarse.triangles, 1)
    np.testing.assert_array_equal(parent[ancestry[1]], ancestry[0])
    mean, surface = means_on_common_faces(coarse, fine, parent, pg, 6)
    n = systems[key].primary.base_symmetry_faces_number
    child = np.repeat(np.arange(n), len(CUBIC_NODES))
    rays = (CUBIC_NODES @ fine.triangles[:n]).reshape(-1, 3)
    total = fine.sample(child, rays) - coarse.sample(parent[child], rays)
    flat = mean[parent[child]]
    old_heat, slopes = local_response(solver, model, observer, systems[key], key, total.reshape(n, -1))
    exact = optical[1].vertex - old_heat
    terms = {"transferred_heat": exact, "face_mean": slopes[1] * flat[:, None, None],
             "within_face": slopes[1] * (total-flat)[:, None, None],
             "step_change": (slopes[1] - slopes[0]) * total[:, None, None]}
    terms["linearization_error"] = exact - terms["face_mean"] - terms["within_face"]
    folded, origin = observer.phase_interval_reduce(observer.manage_time_series(phases=phases))
    support, inverse = dynamic.phase_crv_symmetry(common, folded)
    motion = model.get_positions_method()(input_argument=support, return_nparray=False, calculate_from="phase")
    disks = []
    for position in motion:
        line = utils.rotate_item(np.eye(3), position, model.inclination) @ const.LINE_OF_SIGHT
        shadow = sg if line[0] > 0 and np.sqrt(max(0, 1 - line[0]**2)) < 0.5 else None
        packet = solver._packet(triangles["primary"], pg, line, solver.rule, shadow)
        ids = packet["parents"]
        area, mu = packet["area"] * model.semi_major_axis**2, packet["mu"]
        fp = ancestry[1][ids]
        weights = disk_weights(optical[1], fp, packet.rays, mu, area)
        flux = [f.flux(m[ids], packet.rays, mu, area) for f, m in zip(optical, ancestry, strict=True)]
        # Check the signed basis against the original per-ray ELISa intensity integration.
        np.testing.assert_allclose(np.einsum("nk,nbk->b", weights, optical[1].vertex), flux[1], rtol=3e-13)
        parts = {k: np.einsum("nc,nbc->bc", weights, v) for k, v in terms.items()}
        np.testing.assert_allclose(parts["transferred_heat"], parts["face_mean"] + parts["within_face"] +
                                   parts["linearization_error"], rtol=2e-11, atol=1e-14*np.max(flux))
        disks.append(dict(phase=float(position.phase), nodes=len(ids), primary_flux=flux,
                          changes_by_band_and_moment=parts))
    return dict(bands=solver.passbands, input_phases=phases, indices_to_visible_disks=inverse[origin],
                physical=base["samples"][0]["rows"][0]["physical"], source_revision=base["source_revision"],
                horizon_mean=horizon_mean, surface=surface, depth=4, donor_level=2, steps=[1e-5,5e-6], disks=disks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--horizon-mean", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = measure(json.loads(args.baseline.read_bytes()), args.horizon_mean)
    args.output.write_text(json.dumps(result, default=lambda a: a.tolist(), allow_nan=False) + "\n")
