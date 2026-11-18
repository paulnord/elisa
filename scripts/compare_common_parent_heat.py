"""Integrate the change of deposited T^4 on fixed coarse Roche faces and on their visible disks."""

import argparse
from copy import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from elisa import const, settings, utils
from elisa.binary_system import dynamic
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
from elisa.experimental.gpu_reconstructed_field import reference_barycentric, shape_rule
from elisa.experimental.gpu_shared_fields import shared_parent_map
from elisa.experimental.primary_cubic import CurvedHeatingLightCurve
from elisa.experimental.receiver_curvature import curvature_stencil, heat_basis, heat_controls
from scripts.receiver_phase_parts import context
from scripts.validate_gpu_atmosphere import table_settings


@dataclass
class HeatField:
    triangles: np.ndarray
    condition: np.ndarray
    controls: np.ndarray

    def sample(self, parents, rays):
        bary, _ = reference_barycentric(self.triangles, self.condition, parents, rays)
        return np.sum(heat_basis(np, bary) * self.controls[parents], axis=1)


def thermal_field(geometry, star, triangles, deposits):
    n = star.base_symmetry_faces_number
    rays, solid = duffy_nodes(triangles[:n], 4)
    area = (solid * geometry.sample(rays)[2]).reshape(n, 16)
    mass = (area / area.sum(axis=1, keepdims=True)) @ heat_basis(np, shape_rule()[0])
    stencil = curvature_stencil(star.faces, triangles, n)
    controls, _ = heat_controls(np, *deposits, stencil, mass)
    np.testing.assert_allclose((mass * controls).sum(axis=1), deposits[1], rtol=3e-13, atol=1e-17)
    return HeatField(triangles, np.linalg.cond(triangles), controls[star.face_symmetry_vector])


def means_on_common_faces(coarse, fine, parent, geometry, order):
    """Integrate both fields on the identical curved-area measure over each coarse parent."""
    size = len(coarse.triangles)
    weight = np.zeros(size)
    moment = np.zeros(size)
    second = np.zeros(size)
    for first in range(0, len(parent), 256):
        indices = np.arange(first, min(first + 256, len(parent)))
        rays, solid = duffy_nodes(fine.triangles[indices], order)
        a = solid * geometry.sample(rays)[2]
        child = np.repeat(indices, order**2)
        owner = parent[child]
        delta = fine.sample(child, rays) - coarse.sample(owner, rays)
        for array, integrand in ((weight, a), (moment, a * delta), (second, a * delta**2)):
            array += np.bincount(owner, weights=integrand, minlength=size)
    if np.any(weight <= 0) or not np.isfinite(moment + second + weight).all():
        raise ValueError("Missing or invalid common-face integral")
    mean = moment / weight
    # Orthogonal constant and zero-area-mean components on the stellar surface.
    variance = (second.sum() - np.sum(weight * mean**2)) / weight.sum()
    if variance < -1e-14 * second.sum() / weight.sum():
        raise ValueError("Negative integrated within-face variance")
    return mean, dict(
        area=weight.sum(),
        mean=moment.sum() / weight.sum(),
        rms=np.sqrt(second.sum() / weight.sum()),
        face_mean_rms=np.sqrt(np.average(mean**2, weights=weight)),
        within_rms=np.sqrt(max(0, variance)),
    )


def measure(base):
    fit, _, _, phases, model = context(base)
    observer = copy(fit.observer)
    observer._system = model
    solver = CurvedHeatingLightCurve(
        list(fit.y_data),
        SymmetricInteriorRule(3, 4, slice_order=12, junction_depth=4),
        device="reference",
        selective_shadow=True,
    )
    common, triangles, fields, maps, _ = solver._prepare(model, observer)
    pg = solver._geometry(model, "primary")
    sg = solver._geometry(model, "secondary")
    heat, optical, parent_maps = [], [], []
    for level in (2, 3):
        key = f"curvature_field{level}_donor2"
        item = solver._reconstruction[key, "primary"]
        deposits = solver.engines[key].last_diagnostics["primary"]["reflection_quadrature"]["receiver_deposits"]
        heat.append(thermal_field(pg, fields[key].primary, item.triangles, deposits))
        optical.append(item)
        parent_maps.append(maps[key]["primary"])
    coarse, fine = heat
    parent, _ = shared_parent_map(fine.triangles, coarse.triangles, 1)
    np.testing.assert_array_equal(parent[parent_maps[1]], parent_maps[0])
    four, s4 = means_on_common_faces(coarse, fine, parent, pg, 4)
    six, s6 = means_on_common_faces(coarse, fine, parent, pg, 6)
    folded, origin = observer.phase_interval_reduce(observer.manage_time_series(phases=phases))
    support, reverse = dynamic.phase_crv_symmetry(common, folded)
    motion = model.get_positions_method()(input_argument=support, return_nparray=False, calculate_from="phase")
    disks = []
    for position in motion:
        line = utils.rotate_item(np.eye(3), position, model.inclination) @ const.LINE_OF_SIGHT
        foreground = sg if line[0] > 0 and np.sqrt(max(0, 1 - line[0] ** 2)) < 0.5 else None
        packet = solver._packet(triangles["primary"], pg, line, solver.rule, foreground)
        ids = packet["parents"]
        cp, fp = (m[ids] for m in parent_maps)
        weights = packet["area"] * packet["mu"]
        row = dict(phase=float(position.phase), nodes=len(ids), projected_area=weights.sum())
        row["passband_flux"] = [
            dict(
                zip(
                    solver.passbands,
                    f.flux(p, packet.rays, packet["mu"], packet["area"] * model.semi_major_axis**2),
                    strict=True,
                )
            )
            for f, p in zip(optical, (cp, fp), strict=True)
        ]
        if len(ids):
            delta = fine.sample(fp, packet.rays) - coarse.sample(cp, packet.rays)
            face = six[cp]
            signals = dict(total=delta, face_mean=face, within_face=delta - face)
            row["heat"] = {
                k: dict(mean=np.average(v, weights=weights), rms=np.sqrt(np.average(v**2, weights=weights)))
                for k, v in signals.items()
            }
            np.testing.assert_allclose(
                row["heat"]["total"]["mean"],
                row["heat"]["face_mean"]["mean"] + row["heat"]["within_face"]["mean"],
                rtol=3e-13,
                atol=1e-18,
            )
        disks.append(row)
    return dict(
        donor_level=2,
        depth=4,
        interior_degree=4,
        source_revision=base["source_revision"],
        levels=[2, 3],
        coarse_faces=len(six),
        fine_faces=len(parent),
        surface_by_area_order={"4": s4, "6": s6},
        max_common_parent_mean_order_change=np.max(abs(six - four)),
        input_phases=phases,
        indices_to_visible_disks=reverse[origin],
        disks=disks,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        v = measure(json.loads(args.baseline.read_bytes()))
    args.output.write_text(json.dumps(v, default=lambda a: a.tolist(), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
