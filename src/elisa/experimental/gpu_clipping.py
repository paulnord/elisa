"""FP64 device Sutherland-Hodgman intersections, retaining CPU half-planes.

The subject is a cyclic convex hull; each clipper is a triangle. No coordinate
rounding or widened half-plane is introduced. Buffers use the CPU's conservative
8*n capacity. Unsupported numeric/buffer states are explicit, never zero flux
or a physical-prior penalty. CPU hull construction is a separate responsibility.
"""

# ruff: noqa: SLF001, PLR2004 - experimental kernel composition
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa.experimental.gpu_likelihood import _jax_device

_XY, _TRIANGLE = 2, 3
_MAX_HULL, _MAX_FACES, _MAX_ROWS = 128, 256, 64


@dataclass(frozen=True)
class PreparedClippingBatch:
    """Immutable padded polygons/counts/triangles, with no evaluation transfers."""

    _owner: Any
    _arrays: tuple[Any, ...]

    def evaluate(self) -> dict[str, Any]:
        if not self._owner._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        return self._owner._areas(*self._arrays)

    def block_until_ready(self) -> PreparedClippingBatch:
        for array in self._arrays:
            array.block_until_ready()
        return self


class ConvexTriangleClipper:
    """Intersect padded convex polygons with batches of triangles on device.

    Inputs are (rows,hull_capacity,2), (rows,), (rows,faces,3,2). Counts permit
    empty hulls, as in the CPU algorithm. Exact zero triangle determinant gives
    valid zero area. Status: 0 supported, 1 buffer exceeded, 2 nonfinite numeric
    result. The caller must inspect status before using areas. Convexity/cyclic
    order are preconditions, shared with ELISa's original clipping routine.
    """

    def __init__(self, *, device: str = "gpu") -> None:
        self._jax, jnp, self.device = _jax_device(device)
        barrier = self._jax.lax.optimization_barrier

        def cross(a: Any, b: Any) -> Any:
            return barrier(a[..., 0] * b[..., 1]) - barrier(a[..., 1] * b[..., 0])

        def clip_one(polygon: Any, count: Any, triangle: Any) -> tuple[Any, Any]:
            capacity = 8 * polygon.shape[0]
            indices = jnp.arange(capacity)
            tri = barrier(triangle - triangle[0])
            determinant = cross(tri[1], tri[2])
            sign = jnp.where(determinant > 0, 1.0, -1.0)
            source = jnp.zeros((capacity, 2), dtype=jnp.float64)
            source = source.at[:polygon.shape[0]].set(barrier(polygon - triangle[0]))
            n = jnp.where(determinant == 0, 0, count)

            def plane(edge: Any, state: Any) -> Any:
                vertices, n, bad, overflow = state
                a = tri[edge]
                direction = barrier(tri[(edge + 1) % 3] - a)
                start = vertices[(indices + jnp.maximum(n, 1) - 1) % jnp.maximum(n, 1)]
                end = vertices
                fs = sign * cross(direction, barrier(start - a))
                fe = sign * cross(direction, barrier(end - a))
                active = indices < n
                crossing = active & ((fs >= 0) != (fe >= 0))
                keep = active & (fe >= 0)
                denominator = barrier(fs - fe)
                ratio = fs / jnp.where(crossing, denominator, 1.0)
                intersection = start + barrier(ratio[:, None] * barrier(end - start))
                values = jnp.stack((intersection, end), axis=1).reshape((-1, 2))
                emit = jnp.stack((crossing, keep), axis=1).reshape(-1)
                rank = jnp.cumsum(emit, dtype=jnp.int32) - 1
                total = jnp.sum(emit, dtype=jnp.int32)
                target = jnp.zeros_like(vertices).at[jnp.where(emit, rank, capacity)].set(values, mode="drop")
                bad |= jnp.any(active & (~jnp.isfinite(fs) | ~jnp.isfinite(fe)))
                bad |= jnp.any(crossing & (~jnp.isfinite(denominator)
                                         | ~jnp.all(jnp.isfinite(intersection), axis=1)))
                return target, jnp.minimum(total, capacity), bad, overflow | (total > capacity)

            source, n, bad, overflow = self._jax.lax.fori_loop(
                0, 3, plane, (source, n, ~jnp.isfinite(determinant), jnp.zeros((), dtype=bool)),
            )
            local = barrier(source - source[0])
            terms = cross(local[:-1], local[1:])
            area = 0.5 * jnp.abs(jnp.sum(jnp.where(jnp.arange(1, capacity) < n, terms, 0.0)))
            area = jnp.where(n < 3, 0.0, area)
            status = jnp.where(overflow, 1, jnp.where(bad | ~jnp.isfinite(area), 2, 0))
            return jnp.where(status == 0, area, jnp.nan), status

        def areas(polygons: Any, counts: Any, triangles: Any) -> dict[str, Any]:
            one_row = self._jax.vmap(clip_one, in_axes=(None, None, 0))
            result, status = self._jax.vmap(one_row)(polygons, counts, triangles)
            return {"areas": result, "status": status, "row_valid": jnp.all(status == 0, axis=1)}

        self._areas = self._jax.jit(areas)

    def prepare(self, polygons: Any, counts: Any, triangles: Any) -> PreparedClippingBatch:
        """Validate bounded host shapes and copy an immutable device snapshot."""
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        p, t = (np.array(x, dtype=np.float64, copy=True) for x in (polygons, triangles))
        n = np.array(counts, copy=True)
        if (p.ndim != _TRIANGLE or p.shape[-1] != _XY or not 1 <= p.shape[0] <= _MAX_ROWS
                or not _TRIANGLE <= p.shape[1] <= _MAX_HULL
                or t.ndim != 4 or t.shape[0] != p.shape[0] or t.shape[-2:] != (_TRIANGLE, _XY)
                or not 1 <= t.shape[1] <= _MAX_FACES):
            msg = "Expected bounded polygon (rows,3..128,2) and triangle (rows,1..256,3,2) batches; rows<=64"
            raise ValueError(msg)
        if (n.shape != (p.shape[0],) or not np.issubdtype(n.dtype, np.integer)
                or np.any(n < 0) or np.any(n > p.shape[1])):
            msg = "Integer hull counts must match rows and padded polygon capacity"
            raise ValueError(msg)
        if not np.isfinite(p).all() or not np.isfinite(t).all():
            msg = "Clipping coordinates must be finite"
            raise ValueError(msg)
        return PreparedClippingBatch(self, tuple(self._jax.device_put(x, self.device, may_alias=False)
                                                for x in (p, n.astype(np.int32), t)))
