"""Separate redistribution within a secondary face from feedback into its mean at fixed resolution."""

import argparse
import json
from pathlib import Path

import numpy as np

from elisa import settings
from elisa.binary_system.surface.faces import visibility_test
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import duffy_nodes
from elisa.experimental.gpu_shared_fields import shared_parent_map
from elisa.experimental.node_reemission import NodeReemissionReflection
from scripts.compare_irradiating_sources import incoming
from scripts.receiver_phase_parts import context
from scripts.trace_contact_heating import rank_faces, source
from scripts.validate_gpu_atmosphere import table_settings


def exchange(model, selected, sensitivity):
    old = {i: source(model, i) for i in (2, 3)}
    high = old[3]
    parent, _ = shared_parent_map(high["triangles"]["primary"], old[2]["triangles"]["primary"], 1)
    child = np.flatnonzero(np.isin(parent, selected))
    rays, solid = duffy_nodes(high["triangles"]["primary"][child], 4)
    xyz, normal, jac = ConditionedRocheGeometry(model, "primary").sample(rays)
    owners = np.repeat(child, 16)
    active = np.flatnonzero(visibility_test(xyz, high["primary_limit"], "primary"))
    area = solid * jac
    receiver = high["receiver"]._replace(receiver=xyz, normal=normal, parent=owners)
    result = {}
    for level, original in old.items():
        local = source(model, level, NodeReemissionReflection)
        a, b = original["donor"], local["donor"]
        for x, y in ((a.visible_area, b.visible_area), (a.prefix_ld, b.prefix_ld),
                     (a.spatial.base.native_positions, b.spatial.base.native_positions),
                     (a.native_normals, b.native_normals), (a.intrinsic[a.visible_base], b.intrinsic[b.visible_base])):
            np.testing.assert_array_equal(x, y)
        base_heat = original["heat"][a.visible_base]
        node_heat = local["heat"][b.visible_base]
        face_area = a.visible_area.reshape(-1, 3)
        face_mean = (face_area * node_heat.reshape(-1, 3)).sum(axis=1) / face_area.sum(axis=1)
        face_mean = np.repeat(face_mean, 3)
        within = node_heat - face_mean
        feedback = face_mean - base_heat
        np.testing.assert_allclose((face_area * within.reshape(-1, 3)).sum(axis=1), 0,
                                   atol=5e-15*np.max(face_area * node_heat.reshape(-1, 3)), rtol=0)
        changes = {}
        for key, power in (("within_face", within), ("mean_feedback", feedback)):
            received = np.zeros(len(rays))
            received[active] = incoming(receiver, active, a, power) / model.primary.t_eff**4
            means = np.array([np.average(received[parent[owners] == i], weights=area[parent[owners] == i])
                              for i in selected])
            changes[key] = sensitivity @ means
        # Verify the change at both endpoints using the full intrinsic plus re-emitted luminosities.
        rows = active[::max(1, len(active)//32)]
        endpoint = (incoming(receiver, rows, b, (b.intrinsic + local["heat"])[b.visible_base]) -
                    incoming(receiver, rows, a, (a.intrinsic + original["heat"])[a.visible_base]))
        direct = incoming(receiver, rows, a, within + feedback)
        np.testing.assert_allclose(endpoint, direct, rtol=3e-10, atol=1e-15*model.primary.t_eff**4)
        result[level] = dict(contact_log_likelihood=changes, within_peak_T4_over_Teff4=abs(within).max()/model.secondary.t_eff**4,
                             mean_feedback_peak_T4_over_Teff4=abs(feedback).max()/model.secondary.t_eff**4)
    return dict(levels=result, fine_minus_coarse={key:result[3]["contact_log_likelihood"][key] -
                     result[2]["contact_log_likelihood"][key] for key in ("within_face", "mean_feedback")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "curve", "optical"):
        p.add_argument(key, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    base, run, optical = (json.loads(getattr(args, k).read_bytes()) for k in ("baseline", "curve", "optical"))
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        _, props, _, _, model = context(base)
        if props != optical["physical"]:
            raise ValueError("Contact ranking and re-emission exchange have different proposals")
        ids, sensitivity, ranking = rank_faces(run, optical)
        result = dict(physical=props, selected_faces=ids, exchange=exchange(model, ids, sensitivity))
    args.output.write_text(json.dumps(result, default=lambda x:x.tolist(), allow_nan=False) + "\n")
