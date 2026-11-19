"""Trace the secondary's illumination on primary faces ranked by their contact likelihood contributions."""

import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from elisa import settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits, visibility_test
from elisa.experimental.area_reflection import _MAX_PAIRS
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_shared_fields import shared_parent_map
from elisa.experimental.horizon_receiver import HorizonReceiverReflection
from scripts.check_interior_resolution import mapped as observation_map
from scripts.compare_gpu_joint_budget import curve
from scripts.compare_irradiating_sources import incoming
from scripts.map_receiver_response import compare
from scripts.receiver_phase_parts import context
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_reflection_area import intrinsic_state


def rank_faces(run, optical):
    mapped = compare(run, optical)
    profile, = (p for p in run["profiles"] if p["donor_level"] == optical["donor_level"])
    row = profile["samples"][0]["rows"][0]
    low, high = (curve(row, f"curvature_field{i}") for i in (2, 3))
    order = optical["indices_to_visible_disks"]
    disks = [optical["disks"][i] for i in order]
    heat = np.array(optical["common_heat_difference"])
    sensitivity, roundoff = [], []
    for k, b in enumerate(optical["bands"]):
        unit = np.array([d["unit_mean_response"][k] for d in disks])
        raw = high.component_fluxes[b][:, 0] - np.array([
            sum(d["changes_by_band_and_moment"]["transferred_heat"][k]) for d in disks
        ])
        ix = profile["reducers"][b]
        third = high.additional_light / (1 - high.additional_light)
        delta = unit[ix] + third * unit.mean(axis=0)
        increment = delta @ heat
        parts, numerical = [], []
        whole = np.array([sum(d["changes_by_band_and_moment"]["face_mean"][k]) for d in disks])
        for secondary in (low, high):
            total = raw + secondary.component_fluxes[b][:, 1]
            initial = total[ix] + third * total.mean()
            final = initial + increment
            z0, z1 = initial / initial.mean(), final / final.mean()
            # Exact finite change in the ratio, resolved by face before multiplying by the Gaussian residual.
            response = (delta - z0[:, None] * delta.mean(axis=0)) / final.mean()
            np.testing.assert_allclose(response @ heat, z1 - z0, atol=1e-15)
            residual = (profile["observations"][b] - (z0 + z1)/2) / np.square(profile["uncertainties"][b])
            parts.append(residual[:, None] * response)
            # ELISa also divides by the peak before the observed-mean normalization.
            # Compare the two endpoint predictions at flux precision. Keep the
            # resulting floating-point likelihood change separate from every face.
            def canonical(primary):
                components = {b: np.column_stack((primary, secondary.component_fluxes[b][:, 1]))}
                return observation_map(replace(high, component_fluxes=components), {b: ix})[b]
            c0, c1 = canonical(raw), canonical(raw + whole)
            guard = 8 * np.finfo(float).eps
            np.testing.assert_allclose([c0, c1], [z0, z1], atol=guard, rtol=0)
            canonical_step = c1 - c0
            exact_step = response @ heat
            np.testing.assert_allclose(canonical_step, exact_step, atol=2*guard, rtol=0)
            variance = np.square(profile["uncertainties"][b])
            numerical.append((canonical_step - exact_step) * residual +
                             canonical_step * ((z0 - c0) + (z1 - c1)) / (2 * variance))
        sensitivity.extend(np.mean(parts, axis=0))
        roundoff.extend(np.mean(numerical, axis=0))
    sensitivity = np.asarray(sensitivity)
    contribution = sensitivity * heat
    np.testing.assert_allclose(contribution.sum(axis=1) + roundoff, mapped["by_observation"]["face_mean"], atol=1e-10, rtol=1e-9)
    contact = [i for i, obs in enumerate(mapped["observations"]) if
               (obs["band"].endswith(".B") and np.isclose(obs["phase"], 0.024)) or
               (obs["band"].endswith(".V") and np.isclose(obs["phase"], 0.027))]
    if len(contact) != 3:
        raise ValueError("Contact ranking requires the two repeated B measurements and the V measurement")
    score = abs(contribution[contact]).sum(axis=0)
    top = np.argsort(-score, kind="stable")[:32]
    return top, sensitivity[contact][:, top], dict(
        observations=[mapped["observations"][i] for i in contact], common_faces=len(heat), selected_faces=top,
        selected_heat_difference=heat[top], selected_signed_likelihood=contribution[contact][:, top].sum(axis=1),
        all_signed_likelihood=contribution[contact].sum(axis=1), observation_map_roundoff=np.array(roundoff)[contact],
        max_abs_observation_map_roundoff=np.max(np.abs(roundoff)), fraction_absolute_score=score[top].sum()/score.sum(),
    )


def source(model, level, solver_type=HorizonReceiverReflection):
    system, triangles = intrinsic_state(model, level)
    solver = solver_type("linear", device="reference", donor_level=2)
    sampled = solver._sampled_update
    calls = deque(maxlen=2)

    def record(receiver, donor, heat):
        values, means = sampled(receiver, donor, heat)
        calls.append((receiver, donor, heat))
        return values, means

    solver._sampled_update = record
    solver.evaluate(model, system, level, 2)
    order = sorted(("primary", "secondary"), key=lambda k: getattr(system, k).t_eff)
    receiver, donor, heat = dict(zip(order, calls, strict=True))["primary"]
    limits = faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
    faces = np.flatnonzero(visibility_test(system.secondary.face_centres, limits["secondary"], "secondary"))
    return dict(receiver=receiver, donor=donor, heat=heat, faces=np.repeat(faces, 3), triangles=triangles,
                primary_limit=limits["primary"])


def irradiance(data, xyz, normal, transported=None):
    donor = data["donor"]
    points = donor.spatial.base.native_positions
    emitting_normal = donor.native_normals
    base = donor.visible_base
    a, intrinsic, heated = donor.prefix_ld[0], donor.intrinsic[base], data["heat"][base]
    if len(xyz) * len(base) > _MAX_PAIRS:
        raise ValueError("Direct receiver/emitter trace exceeds the pair cap")
    out = []
    for i in range(0, len(xyz), 8):
        ray = points - xyz[i:i+8, None]
        r2 = np.einsum("ijk,ijk->ij", ray, ray)
        ray /= np.sqrt(r2)[..., None]
        mu = np.maximum(0, -np.einsum("ijk,jk->ij", ray, emitting_normal))
        view = np.maximum(0, np.einsum("ijk,ik->ij", ray, normal[i:i+8]))
        geom = view * mu * donor.visible_area / r2
        full = geom * (1 + (mu-1) * a)
        endpoint = full @ (intrinsic + heated)
        if transported is None:
            out.append(endpoint[:, None])
            continue
        old_a, old_q, old_h = transported
        old_power = old_q + old_h
        fixed = (geom * (1 + (mu-1) * old_a)) @ old_power
        angular = (geom * (mu-1)) @ ((a-old_a) * old_power)
        out.append(np.column_stack((fixed, angular, full @ (intrinsic-old_q), full @ (heated-old_h), endpoint)))
    return np.concatenate(out)


def trace(model, ids, sensitivity):
    low, high = (source(model, level) for level in (2, 3))
    geom = ConditionedRocheGeometry(model, "primary")
    parents = {name: shared_parent_map(high["triangles"][name], low["triangles"][name], 1)[0]
               for name in ("primary", "secondary")}
    child = np.flatnonzero(np.isin(parents["primary"], ids))
    coarse_face = parents["secondary"][high["faces"]]
    lo, hi = low["donor"], high["donor"]
    # The original symmetric solver uses coefficients in visible-prefix order. Transfer the
    # effective coefficient from its physical parent. Newly visible area uses its fine coefficient.
    visible = low["faces"][::3]
    loc = np.searchsorted(visible, coarse_face).clip(max=len(visible)-1)
    represented = visible[loc] == coarse_face
    old_a = np.where(represented, lo.prefix_ld[0, 3*loc], hi.prefix_ld[0])
    q_parent = coarse_face % len(lo.intrinsic)
    np.testing.assert_array_equal(q_parent, hi.visible_base // 4)
    transported = old_a, lo.intrinsic[q_parent], low["heat"][q_parent]
    if len(child) != 4 * len(ids):
        raise ValueError("Incomplete nested primary faces")
    labels = ("geometry_area_visibility", "bolometric_LD", "intrinsic_T4", "reemitted_T4", "total")
    results = {}
    for order in (3, 4):
        rays, solid = duffy_nodes(high["triangles"]["primary"][child], order)
        xyz, normal, jac = geom.sample(rays)
        area = solid * jac
        owners = np.repeat(child, order**2)
        trace_ids = np.repeat(parents["primary"][child], order**2)
        keep = visibility_test(xyz, high["primary_limit"], "primary")
        values = np.zeros((len(xyz), len(labels)))
        coarse = irradiance(low, xyz[keep], normal[keep])[:, 0]
        fine = irradiance(high, xyz[keep], normal[keep], transported)
        # Both endpoints use identical receiving geometry, absorption and visibility.
        scale = high["receiver"].absorption[owners[keep]] / model.primary.t_eff**4
        fine[:, (0, 4)] -= coarse[:, None]
        values[keep] = fine * scale[:, None]
        np.testing.assert_allclose(values[:, :4].sum(axis=1), values[:, 4], atol=4e-16, rtol=3e-11)
        # Check both endpoints against the independent original distance, horizon and LD routines.
        rows = np.flatnonzero(keep)[::max(1, keep.sum()//16)]
        for source_data, expected in ((low, coarse), (high, fine[:, 4]+coarse)):
            d = source_data["donor"]
            receiver = high["receiver"]._replace(receiver=xyz, normal=normal, parent=owners)
            native = incoming(receiver, rows, d, (d.intrinsic + source_data["heat"])[d.visible_base])
            np.testing.assert_allclose(native, expected[np.searchsorted(np.flatnonzero(keep), rows)] *
                                       receiver.absorption[owners[rows]], rtol=3e-12, atol=1e-16*model.primary.t_eff**4)
        means = np.array([np.average(values[trace_ids == face], weights=area[trace_ids == face], axis=0) for face in ids])
        results[str(order)] = dict(receiving_nodes=len(rays), active_receiving_nodes=int(keep.sum()),
                                   face_means=dict(zip(labels, means.T, strict=True)),
                                   contact_log_likelihood=dict(zip(labels, (sensitivity @ means).T, strict=True)))
    return dict(orders=results, newly_visible_nodes=int(np.sum(~represented)), coarse_nodes=len(lo.visible_base),
                fine_nodes=len(hi.visible_base), effective_LD_range=[old_a.min(), old_a.max(), hi.prefix_ld.min(), hi.prefix_ld.max()])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "curve", "optical"):
        p.add_argument(key, type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base, run, optical = (json.loads(getattr(args, key).read_bytes()) for key in ("baseline", "curve", "optical"))
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        _, props, _, _, model = context(base)
        ids, sensitivity, ranking = rank_faces(run, optical)
        if props != optical["physical"]:
            raise ValueError("Physical proposal changed between contact ranking and source trace")
        result = dict(physical=props, ranking=ranking, source=trace(model, ids, sensitivity))
    args.output.write_text(json.dumps(result, default=lambda v: v.tolist(), allow_nan=False) + "\n")
