"""Batched triangle geometry and observer projection; no eclipse clipping.

Provided topology is not recomputed as physical parameters change. This kernel
must not be paired with stale coverage/thermal state to claim a full light curve.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from elisa.experimental.gpu_likelihood import _jax_device

_XYZ = 3
_FACE_NDIM = 2
_MAX_COORDINATE_EDGE_RATIO = 1e4
_MIN_TRIANGLE_SINE = 0.05  # conservative conditioning gate for FP64 unit normals


class TriangleGeometry:
    """Compute ELISa triangular areas, centroids, outward normals and LOS cosines.

    Vertices are already in ELISa's co-rotating binary frame (secondary x must be
    placed appropriately); com_x supplies its center. LOS is fixed to (-1,0,0).
    Inputs/temporaries scale with proposals times faces; chunk to your budget.
    Centres are centroids of SUPPLIED vertices, not necessarily cached ELISa
    star.face_centres: the CPU build can cache centres before mesh correction
    while keeping corrected points/areas. This is not full container-state parity.
    """

    def __init__(self, *, device: str = "gpu") -> None:
        self._jax, jnp, self.device = _jax_device(device)

        def geometry(vertices: Any, faces: Any, com: Any, azimuth: Any, inclination: Any) -> dict[str, Any]:
            triangles = vertices[:, faces, :]
            centres = jnp.mean(triangles, axis=2)
            a = triangles[:, :, 1] - triangles[:, :, 0]
            b = triangles[:, :, 2] - triangles[:, :, 0]
            cross = jnp.cross(a, b)
            norm = jnp.linalg.norm(cross, axis=-1)
            relative = centres - jnp.stack((com, jnp.zeros_like(com), jnp.zeros_like(com)), axis=-1)[:, None, :]
            normal = cross / jnp.where(norm > 0, norm, 1)[..., None]
            orientation_dot = jnp.sum(normal * relative, axis=-1)
            orientation_scale = jnp.linalg.norm(relative, axis=-1)
            eps = 64 * jnp.finfo(jnp.float64).eps
            orientation_ambiguous = jnp.abs(orientation_dot) <= eps * orientation_scale
            edge_a, edge_b = jnp.linalg.norm(a, axis=-1), jnp.linalg.norm(b, axis=-1)
            edge_scale = edge_a * edge_b
            coordinate_scale = jnp.max(jnp.abs(triangles), axis=(-1, -2))
            coordinate_ambiguous = coordinate_scale > _MAX_COORDINATE_EDGE_RATIO * jnp.minimum(edge_a, edge_b)
            triangle_ambiguous = norm <= _MIN_TRIANGLE_SINE * edge_scale
            orientation = jnp.sign(orientation_dot)
            normal *= orientation[..., None]
            valid = (
                (norm > 0)
                & ~orientation_ambiguous
                & ~triangle_ambiguous
                & ~coordinate_ambiguous
                & (orientation != 0)
                & jnp.isfinite(norm)
                & jnp.isfinite(orientation_dot)
                & jnp.all(jnp.isfinite(centres), axis=-1)
                & jnp.all(jnp.isfinite(normal), axis=-1)
            )
            normal = jnp.where(valid[..., None], normal, jnp.nan)
            alpha, beta = azimuth + jnp.pi / 2, jnp.pi / 2 - inclination
            ca, sa, cb, sb = jnp.cos(alpha), jnp.sin(alpha), jnp.cos(beta), jnp.sin(beta)
            # Same row-vector z then inverse-y rotations as utils.rotate_item.
            nx = normal[..., 0] * ca[:, None] - normal[..., 1] * sa[:, None]
            ny = normal[..., 0] * sa[:, None] + normal[..., 1] * ca[:, None]
            nz = normal[..., 2]
            rotated = jnp.stack((nx * cb[:, None] - nz * sb[:, None], ny, nx * sb[:, None] + nz * cb[:, None]), axis=-1)
            mu = -rotated[..., 0]
            return {
                "centres": centres,
                "areas": norm / 2,
                "normals": normal,
                "observer_normals": rotated,
                "los_cosines": mu,
                "facing": mu > 0,
                "visibility_ambiguous": jnp.abs(mu) <= 64 * jnp.finfo(jnp.float64).eps,
                "orientation_ambiguous": orientation_ambiguous,
                "triangle_ambiguous": triangle_ambiguous,
                "coordinate_ambiguous": coordinate_ambiguous,
                "valid_faces": valid,
                "proposal_valid": jnp.all(valid, axis=1),
            }

        self._geometry = self._jax.jit(geometry)

    def __call__(self, vertices: Any, faces: Any, com_x: Any, azimuth: Any, inclination: Any) -> dict[str, Any]:
        """Return device geometry. Angles are radians, one orientation per row.

        Check proposal_valid and all ambiguity masks before using new masks.
        Near-COM-coplanar outward orientation and ill-conditioned thin faces
        are flagged separately. Triangle sin(angle)<=0.05 is unsupported,
        a deliberately conservative conditioning restriction for FP64 normals.
        Coordinates above 1e4 times the shortest defining edge are also
        unsupported to limit subtraction-conditioning problems. These gates
        do not certify arbitrary inputs to an exact error bound. Unsupported geometry is not a physical prior
        rejection; downstream code must fall back or explicitly fail.
        Degenerate/unorientable triangles are unsupported, not hidden as zero
        flux. Near-limb masks may differ by floating-point roundoff: retain CPU
        face indexing/coverage until full clipping parity is established.
        """
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        v = np.array(vertices, dtype=np.float64, copy=True)
        f = np.array(faces, copy=True)
        arrays = [np.array(x, dtype=np.float64, copy=True) for x in (com_x, azimuth, inclination)]
        if v.ndim != _XYZ or v.shape[-1] != _XYZ or not v.shape[0] or not v.shape[1] or not np.isfinite(v).all():
            msg = "Vertices must be finite (proposals, vertices, 3) arrays"
            raise ValueError(msg)
        if f.ndim != _FACE_NDIM or f.shape[-1] != _XYZ or not f.shape[0] or not np.issubdtype(f.dtype, np.integer):
            msg = "Faces must be nonempty integer triangle indices"
            raise ValueError(msg)
        if np.any(f < 0) or np.any(f >= v.shape[1]):
            msg = "Face index out of bounds"
            raise ValueError(msg)
        if any(x.shape != (v.shape[0],) or not np.isfinite(x).all() for x in arrays):
            msg = "Orientation and center arrays must match proposal count and be finite"
            raise ValueError(msg)
        _, az, inc = arrays
        if np.any(np.abs(az) > 2 * np.pi) or np.any((inc < 0) | (inc > np.pi)):
            msg = "Azimuth must lie in [-2pi,2pi] and inclination in [0,pi]"
            raise ValueError(msg)
        return self._geometry(*(self._jax.device_put(x, self.device, may_alias=False) for x in (v, f, *arrays)))
