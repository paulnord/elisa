"""GPU nested boundary solves with unchanged finite CPU adaptation/clipping.

This is the m017 packet algorithm with compiled field/root evaluations. It
preserves exact-zero endpoints, positive measures and every original resource,
stationarity, cone, topology and singular-source guard. Final brackets and rare
precision seams explicitly retain original CPU arithmetic.
"""

# ruff: noqa: C901, PLR0912, PLR0915 - fixed scientific protocol
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes, split_triangles
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, _SingularSliceError, curved_edge_packet
from elisa.experimental.gpu_observer_geometry import ObserverGeometryBackend
from elisa.experimental.gpu_resident_geometry import ResidentGeometryLightCurve


def _clip_device(
    triangles: np.ndarray, source: np.ndarray, tags: np.ndarray, field: Any, flag: int, steps: int,
) -> tuple:
    """Original polygon clipping arithmetic, retaining boundary-edge identities."""
    if not len(triangles):
        return triangles, source, tags, 0
    values = field(triangles.reshape((-1, 3))).reshape((-1, 3))
    tags = tags | ((values == 0) * flag).astype(np.uint8)
    positive = values >= 0
    crossing = positive != np.roll(positive, -1, axis=1)
    roots = np.zeros_like(triangles)
    a, b = triangles[crossing], np.roll(triangles, -1, axis=1)[crossing]
    low, high = np.zeros(len(a)), np.ones(len(a))
    low_positive = positive[crossing]
    if len(a):
        low, high = field.bisect(a, b - a, low, high, low_positive, steps)
    root = a + (b - a) * ((low + high) / 2)[:, None]
    va, vb = values[crossing], np.roll(values, -1, axis=1)[crossing]
    # Preserve exact zero endpoints instead of inventing a tiny seam sliver
    # by stopping bisection a finite distance away from the known root.
    root = np.where((va == 0)[:, None], a, np.where((vb == 0)[:, None], b, root))
    roots[crossing] = root
    root_tags = (tags & np.roll(tags, -1, axis=1)) | flag
    slots, labels = np.zeros((len(triangles), 6, 3)), np.zeros((len(triangles), 6), dtype=np.uint8)
    size = np.zeros(len(triangles), dtype=int)
    for edge in range(3):
        for take, points, kinds in [
            (positive[:, edge], triangles[:, edge], tags[:, edge]),
            (crossing[:, edge], roots[:, edge], root_tags[:, edge]),
        ]:
            index = np.flatnonzero(take)
            slots[index, size[index]], labels[index, size[index]] = points[index], kinds[index]
            size += take
    pieces, ancestry, kinds = [], [], []
    for j in range(1, 3):
        take = size > j + 1
        pieces.append(slots[take][:, [0, j, j + 1]])
        kinds.append(labels[take][:, [0, j, j + 1]])
        ancestry.append(source[take])
    return np.concatenate(pieces), np.concatenate(ancestry), np.concatenate(kinds), len(a)


def _slice_device(
    triangles: np.ndarray, fields: list, flags: np.ndarray, rule: CurvedEdgeRule, source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact implicit base per Gaussian slice, with its analytic cone Jacobian.

    E(s)=(1-s)a+s b, C(s)=E+lambda(s)(c-E), v=(1-t)C+t c.
    The lambda' term cancels in det(v,v_s,v_t). The positive solid-angle
    measure is |det(a,b,c)|(1-t)(1-lambda)^2/|v|^3 ds dt.
    No numerical boundary derivative or signed strip correction is needed.
    """
    x, w = leggauss(rule.slice_order)
    x, w = (x + 1) / 2, w / 2
    t, tw = leggauss(rule.order)
    t, tw = (t + 1) / 2, tw / 2
    a, b, c = triangles.transpose(1, 0, 2)
    edge = a[:, None] * (1 - x[None, :, None]) + b[:, None] * x[None, :, None]
    direction = c[:, None] - edge
    shift = np.zeros(edge.shape[:2])
    for flag, field in enumerate(fields, 1):
        take = flags == flag
        if not take.any():
            continue
        e, d = edge[take], direction[take]
        # A shadow chord can lie inside a small occultor while continuing
        # past it leaves the shadow again. Start at the chord when negative;
        # do not skip the desired near boundary by extrapolating through it.
        initial = field(e.reshape((-1, 3))).reshape(e.shape[:2])
        lo = np.where(initial <= 0, 0.0, -0.5)
        hi = np.ones(e.shape[:2])
        fl = field((e + lo[..., None] * d).reshape((-1, 3))).reshape(lo.shape)
        fh = field((e + hi[..., None] * d).reshape((-1, 3))).reshape(lo.shape)
        if not np.isfinite(fl + fh).all():
            msg = "Curved slice has nonfinite boundary values"
            raise UnsupportedCurvedGeometryError(msg)
        bad = np.any((fl > 0) | (fh < 0), axis=1)
        if bad.any():
            mask = np.zeros(len(triangles), dtype=bool)
            mask[take] = bad
            raise _SingularSliceError(mask)
        left, right = field.bisect(
            e.reshape((-1, 3)),
            d.reshape((-1, 3)),
            lo.reshape(-1),
            hi.reshape(-1),
            np.zeros(lo.size, dtype=bool),
            rule.curve_bisections,
        )
        lo, hi = left.reshape(lo.shape), right.reshape(hi.shape)
        shift[take] = (lo + hi) / 2
    u = shift[:, :, None] + (1 - shift[:, :, None]) * t[None, None]
    vectors = edge[:, :, None] + u[..., None] * direction[:, :, None]
    det = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    weights = (
        det[:, None, None]
        * w[None, :, None]
        * tw[None, None, :]
        * (1 - t)[None, None, :]
        * (1 - shift[:, :, None]) ** 2
        / np.linalg.norm(vectors, axis=3) ** 3
    )
    # Warping may cross a chord, but may never leave the declared source leaf.
    check = np.concatenate((vectors.reshape((len(source), -1, 3)), edge + shift[..., None] * direction), axis=1)
    coefficients = np.linalg.solve(source.transpose(0, 2, 1), check.transpose(0, 2, 1))
    guard = 128 * np.finfo(float).eps * np.linalg.cond(source) * np.maximum(1.0, np.abs(coefficients).max(axis=(1, 2)))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        msg = "Curved slice lost its positive finite measure"
        raise UnsupportedCurvedGeometryError(msg)
    bad = np.any(coefficients < -guard[:, None, None], axis=(1, 2))
    if bad.any():
        raise _SingularSliceError(bad)
    return vectors.reshape((-1, 3)), weights.reshape(-1), float(np.abs(shift).max())


def observer_packet(
    triangles: np.ndarray, geometry: Any, line: np.ndarray, rule: CurvedEdgeRule, foreground: Any = None,
) -> dict:
    """Fresh finite packet; true single edges and explicitly refined junctions."""
    if not isinstance(rule, CurvedEdgeRule):
        msg = "Curved edge packet requires CurvedEdgeRule"
        raise TypeError(msg)
    geometry.line = line
    current, parents = np.array(triangles, copy=True), np.arange(len(triangles))
    accepted, ancestry, junctions, counts = [], [], [], []
    probes = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [1 / 3, 1 / 3, 1 / 3]],
    )
    for depth in range(rule.depth + max(rule.topology_depth, rule.junction_depth) + 1):
        v = np.einsum("qj,njk->nqk", probes, current)
        p, normal, _ = geometry.sample(v.reshape((-1, 3)))
        p, normal = p.reshape((-1, 7, 3)), normal.reshape((-1, 7, 3))
        mu = normal @ line
        spread = np.linalg.norm(normal - normal[:, :1], axis=2).max(axis=1)
        possible = mu.max(axis=1) > -2 * spread
        limb_band = possible & (np.min(np.abs(mu), axis=1) <= 2 * spread)
        boundary, junction, shadow = limb_band.copy(), np.zeros(len(current), dtype=bool), None
        if foreground is not None:
            blocked, distance = foreground.shadow(p.reshape((-1, 3)), line)
            shadow = distance.reshape((-1, 7))
            diameter = np.linalg.norm(p - p[:, :1], axis=2).max(axis=1)
            shadow_band = possible & (np.abs(shadow).min(axis=1) <= 2 * diameter)
            junction = limb_band & shadow_band
            boundary |= shadow_band
            if not np.array_equal(blocked.reshape((-1, 7)), shadow < 0):
                msg = "Curved-edge shadow predicate/margin disagree"
                raise UnsupportedCurvedGeometryError(msg)
        refine = boundary & (depth < rule.depth)
        junction_refine = junction & (depth < rule.depth + rule.junction_depth)
        topology_refine = np.zeros(len(current), dtype=bool)
        if depth >= rule.depth:
            for field in [mu, *([] if shadow is None else [shadow])]:
                same = np.all((field[:, :3] >= 0) == (field[:, :1] >= 0), axis=1)
                hidden = np.any((field >= 0) != (field[:, :1] >= 0), axis=1)
                topology_refine |= possible & same & hidden
                if depth >= rule.depth + rule.topology_depth and np.any(topology_refine & ~junction_refine):
                    msg = "Curved-edge sampled contour topology exhausted the fixed rule"
                    raise UnsupportedCurvedGeometryError(msg)
        refine |= junction_refine | topology_refine
        take = possible & ~refine
        accepted.append(current[take])
        ancestry.append(parents[take])
        junctions.append(junction[take])
        counts.append(
            {
                "depth": depth,
                "leaves": len(current),
                "refined": int(refine.sum()),
                "accepted": int(take.sum()),
                "boundary_leaves": int(boundary.sum()),
                "topology_refined": int(topology_refine.sum()),
                "junction_refined": int(junction_refine.sum()),
            },
        )
        if (sum(len(a) for a in accepted) + 4 * int(refine.sum())) * rule.order**2 > rule.max_nodes:
            msg = "Curved-edge quadrature exceeds explicit node cap; abort protocol"
            raise UnsupportedCurvedGeometryError(msg)
        if not refine.any():
            break
        current, parents = split_triangles(current[refine]), np.repeat(parents[refine], 4)
    leaves, parent, mixed = np.concatenate(accepted), np.concatenate(ancestry), np.concatenate(junctions)

    geometry.line = line
    limb = geometry.field(line)
    fields = [limb]
    if foreground is not None:
        foreground.line = line
        fields.append(geometry.field(line, foreground))
    fragments, source = leaves, np.arange(len(leaves))
    tags = np.zeros((len(leaves), 3), dtype=np.uint8)
    roots = []
    for flag, field in enumerate(fields, 1):
        fragments, source, tags, n = _clip_device(fragments, source, tags, field, flag, rule.edge_bisections)
        roots.append(n)
    # Exact contour-aligned seam edges can produce zero-measure triangles.
    # Remove only exact zeros, never tolerance-discard a positive patch.
    determinant = np.einsum("ij,ij->i", fragments[:, 0], np.cross(fragments[:, 1], fragments[:, 2]))
    duplicate = np.any(np.all(fragments == np.roll(fragments, -1, axis=1), axis=2), axis=1)
    nonzero = (determinant != 0) & ~duplicate
    fragments, source, tags = fragments[nonzero], source[nonzero], tags[nonzero]
    nodes = len(fragments) * rule.order**2
    if nodes > rule.max_nodes:
        msg = "Clipped curved-edge quadrature exceeds explicit node cap; abort protocol"
        raise UnsupportedCurvedGeometryError(msg)
    common = tags & np.roll(tags, -1, axis=1)
    edges = common != 0
    chord = mixed[source]
    if np.any((edges.sum(axis=1) > 1) & ~chord):
        msg = "Curved-edge fragment has multiple unresolved contours outside its junction protocol"
        raise UnsupportedCurvedGeometryError(msg)
    curved = (edges.sum(axis=1) == 1) & ~chord
    allocation_nodes = int(curved.sum()) * rule.order * rule.slice_order + int((~curved).sum()) * rule.order**2
    if allocation_nodes > rule.max_nodes:
        msg = "Curved-edge slice order exceeds explicit node cap before allocation; abort protocol"
        raise UnsupportedCurvedGeometryError(msg)
    vectors, weights, ids = [], [], []
    if np.any(~curved):
        v, w = duffy_nodes(fragments[~curved], rule.order)
        vectors.append(v)
        weights.append(w)
        ids.append(np.repeat(source[~curved], rule.order**2))
    max_shift, singular = 0.0, set()
    for edge in range(3):
        take = curved & edges[:, edge]
        if not take.any():
            continue
        indices = np.flatnonzero(take)
        while len(indices):
            f = fragments[indices][:, [edge, (edge + 1) % 3, (edge + 2) % 3]]
            try:
                v, w, shift = _slice_device(f, fields, common[indices, edge], rule, leaves[source[indices]])
                break
            except _SingularSliceError as error:
                singular.update(source[indices[error.mask]].tolist())
                indices = indices[~error.mask]
        if not len(indices):
            continue
        max_shift = max(max_shift, shift)
        vectors.append(v)
        weights.append(w)
        ids.append(np.repeat(source[indices], rule.order * rule.slice_order))
    singular_triangles = 0
    if singular:
        bad_ids = np.array(sorted(singular), dtype=int)
        # Reintegrate the entire original leaf, not its already clipped chord
        # fragment; otherwise an outward-curving crescent could be lost.
        for j in range(len(ids)):
            keep = ~np.isin(ids[j], bad_ids)
            vectors[j], weights[j], ids[j] = vectors[j][keep], weights[j][keep], ids[j][keep]
        fine = leaves[bad_ids]
        if len(fine) * 4**rule.singular_depth * rule.order**2 > rule.max_nodes:
            msg = "Singular curved-edge subdivision exceeds explicit node cap before allocation; abort protocol"
            raise UnsupportedCurvedGeometryError(msg)
        for _ in range(rule.singular_depth):
            fine = split_triangles(fine)
        origin = np.repeat(bad_ids, 4**rule.singular_depth)
        labels = np.zeros((len(fine), 3), dtype=np.uint8)
        for flag, field in enumerate(fields, 1):
            fine, origin, labels, _ = _clip_device(fine, origin, labels, field, flag, rule.edge_bisections)
        singular_triangles = len(fine)
        if sum(len(a) for a in ids) + len(fine) * rule.order**2 > rule.max_nodes:
            msg = "Singular curved-edge refinement exceeds explicit node cap; abort protocol"
            raise UnsupportedCurvedGeometryError(msg)
        v, w = duffy_nodes(fine, rule.order)
        keep = w > 0
        vectors.append(v[keep])
        weights.append(w[keep])
        ids.append(np.repeat(origin, rule.order**2)[keep])
    nodes = sum(len(a) for a in ids)
    if nodes > rule.max_nodes:
        msg = "Curved-edge slice order exceeds explicit node cap; abort protocol"
        raise UnsupportedCurvedGeometryError(msg)
    if vectors:
        vectors, weights, ids = np.concatenate(vectors), np.concatenate(weights), np.concatenate(ids)
        _, normal, jacobian = geometry.sample(vectors)
        mu = normal @ line
        valid = mu > 0
        area = weights * jacobian
    else:
        mu, area, ids, valid = np.zeros(0), np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
    return {
        "mu": mu[valid],
        "area": area[valid],
        "parents": parent[ids[valid]],
        "counts": counts,
        "nodes": nodes,
        "selected_nodes": int(valid.sum()),
        "limb_edge_roots": roots[0],
        "shadow_edge_roots": roots[1] if len(roots) > 1 else 0,
        "curved_edge_triangles": int((curved & ~np.isin(source, sorted(singular))).sum()),
        "junction_chord_triangles": int(chord.sum()),
        "singular_chord_leaves": len(singular),
        "singular_chord_triangles": singular_triangles,
        "max_absolute_slice_shift": max_shift,
    }


class ObserverDeviceLightCurve(ResidentGeometryLightCurve):
    """Complete fresh reflected curves with explicit hybrid observer port.

    Reference uses original CPU field/observer equations. Device field geometry
    remains the m018 graph; observer roots/silhouettes/edge iterations now run
    on device. CPU adaptation, clipping, ancestry, final root brackets and
    required host transfers stay measured. No fit or continuum qualification.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.observer_backend = None if self.device == "reference" else ObserverGeometryBackend(device=self.device)
        if self.observer_backend is not None:
            self._geometry = self.observer_backend.bind
            self._packet = observer_packet
        else:
            self._packet = curved_edge_packet
        self.last_observer_stats = {}

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        if self.observer_backend is not None:
            self.observer_backend.stats = {}
        result = super().batch(*args, **kwargs)
        self.last_observer_stats = {} if self.observer_backend is None else dict(self.observer_backend.stats)
        return replace(
            result,
            results=tuple(
                replace(
                    r,
                    diagnostics={
                        **r.diagnostics,
                        "observer_contract": ("Device nested roots/silhouettes; original CPU topology/clipping "
                                      "and final bracket checks."),
                    },
                )
                for r in result.results
            ),
        )
