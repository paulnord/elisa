"""Conservative reconstruction of the original face-deposited irradiating flux."""

from __future__ import annotations

from typing import Any

import numpy as np

from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError


def receiver_stencil(faces: np.ndarray, triangles: np.ndarray, base: int) -> tuple[np.ndarray, np.ndarray]:
    """Map three neighbouring deposit differences to tangent-plane corner slopes.

    Pair edges on the full closed surface before folding the indices, so a
    receiver on a symmetry seam sees its physical neighbour across that seam.
    """
    edges = np.sort(np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1), axis=-1).reshape(-1, 2)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    pairs = order.reshape(-1, 2)
    if not np.array_equal(edges[pairs[:, 0]], edges[pairs[:, 1]]) or np.any(
        np.all(edges[pairs[1:, 0]] == edges[pairs[:-1, 0]], axis=1)
    ):
        msg = "Receiver gradient requires two distinct incident faces per edge"
        raise UnsupportedCurvedGeometryError(msg)
    adjacent = np.empty(len(edges), dtype=int)
    adjacent[pairs] = pairs[:, ::-1] // 3
    neighbours = adjacent.reshape(-1, 3)[:base]
    centres = triangles.mean(axis=1)
    centres /= np.linalg.norm(centres, axis=1)[:, None]
    normal = centres[:base]
    tangent = triangles[:base, 1] - triangles[:base, 0]
    tangent -= normal * np.sum(normal * tangent, axis=1)[:, None]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    basis = np.stack((tangent, np.cross(normal, tangent)), axis=2)
    design = (centres[neighbours] - normal[:, None]) @ basis
    condition = np.linalg.cond(design)
    if not np.isfinite(condition).all() or condition.max() > 1e5:
        msg = "Ill-conditioned receiver gradient stencil"
        raise UnsupportedCurvedGeometryError(msg)
    offsets = (triangles[:base] - normal[:, None]) @ basis
    return neighbours % base, offsets @ np.linalg.pinv(design)


def receiver_slopes(xp: Any, heat: Any, neighbours: Any, transform: Any, mass: Any) -> tuple:
    """Limit only the slope, preserving mean deposited T⁴ and nonnegative heating."""
    delta = heat[neighbours] - heat[:, None]
    slope = xp.einsum("fij,fj->fi", transform, delta)
    slope = slope - xp.sum(mass * slope, axis=1)[:, None]
    drop = xp.maximum(0.0, -xp.min(slope, axis=1))
    scale = xp.minimum(1.0, heat / xp.where(drop > 0, drop, 1.0))
    scale = xp.where(drop > 0, scale, 1.0)
    return slope * scale[:, None], scale
