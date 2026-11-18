"""Spend one extra boundary subdivision on angular parents crossed by the observed shadow."""

from dataclasses import replace

import numpy as np

from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_reconstructed_observer import _Packet, partitioned_packet, recorded_packet
from elisa.experimental.quadratic_source import QuadraticSourceLightCurve


def shadow_parents(triangles, geometry, line, foreground):
    # Apply an extra level wherever any two of the seven ordinary topology probes disagree.
    # Every other parent still runs the full limb, hidden-contour and junction procedure.
    probes = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [1 / 3] * 3])
    rays = np.einsum("qj,fjk->fqk", probes, triangles)
    points, _, _ = geometry.sample(rays.reshape(-1, 3))
    blocked, distance = foreground.shadow(points, line)
    if not np.isfinite(distance).all() or not np.array_equal(blocked, distance < 0):
        raise UnsupportedCurvedGeometryError("Shadow selection requires matching finite occultation predicates")
    blocked = blocked.reshape(-1, len(probes))
    return blocked.any(axis=1) & ~blocked.all(axis=1)


def deeper_shadow_packet(triangles, geometry, line, rule, foreground=None):
    if foreground is None:
        return partitioned_packet(triangles, geometry, line, rule, foreground)
    fine_rule = replace(rule, depth=rule.depth + 1)
    parts, total, refined = [], 0, 0
    for start in range(0, len(triangles), 4096):
        block = triangles[start : start + 4096]
        mask = shadow_parents(block, geometry, line, foreground)
        refined += int(mask.sum())
        for selected, depth_rule in ((~mask, rule), (mask, fine_rule)):
            parents = np.flatnonzero(selected)
            if not len(parents):
                continue
            part = recorded_packet(block[parents], geometry, line, depth_rule, foreground)
            total += part["nodes"]
            if total > rule.max_nodes:
                raise UnsupportedCurvedGeometryError("Refined full shadow packet exceeds the original node cap")
            part["parents"] = parents[part["parents"]] + start
            parts.append(part)
    out = _Packet()
    for key in ("parents", "area", "mu"):
        out[key] = np.concatenate([p[key] for p in parts])
    out.rays = np.concatenate([p.rays for p in parts])
    out["counts"] = []
    for depth in range(max(len(p["counts"]) for p in parts)):
        records = [p["counts"][depth] for p in parts if depth < len(p["counts"])]
        out["counts"].append({k: depth if k == "depth" else sum(r[k] for r in records) for k in records[0]})
    for k in parts[0].keys() - out.keys():
        values = [p[k] for p in parts]
        out[k] = max(values) if k == "max_absolute_slice_shift" else sum(values)
    out.update(
        shadow_crossing_parents=refined, extra_shadow_depth=1, largest_partition_nodes=max(p["nodes"] for p in parts)
    )
    return out


class DeeperShadowLightCurve(QuadraticSourceLightCurve):
    """Quadratic source and emitter, with an extra level at detected shadow crossings."""

    rule_prefix = "shadow_refined_quadratic_source"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._packet = deeper_shadow_packet
