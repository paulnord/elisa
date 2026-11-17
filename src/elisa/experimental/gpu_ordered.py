"""Device surface arithmetic with NumPy-compatible contiguous FP64 sum order.

Independent JAX implementation of the public NumPy pairwise layout: eight
lanes, 128-element leaves, eight-aligned binary splits and serial tails. See
https://github.com/numpy/numpy/blob/v2.5.5/numpy/_core/src/umath/loops_utils.h.src
Actual selected counts, rather than padding, determine the reduction tree.
This prevents harmless forward roundoff from perturbing sensitive numerical
optimizer trajectories. It is not a universal bitwise transcendental guarantee.
"""

# ruff: noqa: SLF001 - reuse host validation and immutable device snapshot type
from __future__ import annotations

from typing import Any

import numpy as np

from elisa.experimental.gpu_surface import PreparedSurfaceBatch, SurfaceFluxIntegrator

_LEAF, _LANES, _MAX_FACES, _MAX_ROWS = 128, 8, 8192, 16
_ROW_NDIM = 2


class OrderedSurfaceFluxIntegrator:
    """Retain original CPU operation boundaries and selected-length sum layout."""

    def __init__(  # noqa: C901, PLR0915 - explicit ordered reduction and FP64 expression stages
        self, law: str, *, device: str = "gpu",
    ) -> None:
        self._validation = SurfaceFluxIntegrator(law, device=device)
        self._jax, self.device, self._ncoeff = (
            self._validation._jax, self._validation.device, self._validation._ncoeff,
        )
        import jax.numpy as jnp  # noqa: PLC0415 - optional dependency

        barrier = self._jax.lax.optimization_barrier

        def leaf(values: Any, counts: Any, starts: Any) -> Any:
            indices = starts[:, None] + jnp.arange(_LEAF)[None]
            block = jnp.take_along_axis(values, jnp.clip(indices, 0, values.shape[1]-1), axis=1)
            rows = jnp.arange(len(counts))
            lanes = block[:, :_LANES]
            rounded = counts-counts % _LANES

            def add_group(i: Any, accumulators: Any) -> Any:
                index = _LANES*i + jnp.arange(_LANES)
                more = jnp.take_along_axis(block, jnp.broadcast_to(index, accumulators.shape), axis=1)
                return jnp.where((_LANES*i < rounded)[:, None], barrier(accumulators+more), accumulators)

            lanes = self._jax.lax.fori_loop(1, _LEAF//_LANES, add_group, lanes)
            pairs = [barrier(lanes[:, i]+lanes[:, i+1]) for i in range(0, _LANES, 2)]
            total = barrier(barrier(pairs[0]+pairs[1])+barrier(pairs[2]+pairs[3]))

            def add_tail(i: Any, current: Any) -> Any:
                index = rounded+i
                more = block[rows, jnp.minimum(index, _LEAF-1)]
                return jnp.where(index < counts, barrier(current+more), current)

            total = self._jax.lax.fori_loop(0, _LANES, add_tail, total)

            def serial(i: Any, current: Any) -> Any:
                return jnp.where(i < counts, barrier(current+block[:, i]), current)

            short = self._jax.lax.fori_loop(0, _LANES, serial, jnp.full(counts.shape, -0.0))
            return jnp.where(counts < _LANES, short, total)

        def pairwise(values: Any, counts: Any, starts: Any, capacity: int) -> Any:
            if capacity <= _LEAF:
                return leaf(values, counts, starts)
            split = counts > _LEAF
            left = counts//2
            left -= left % _LANES
            left = jnp.where(split, left, counts)
            right = jnp.where(split, counts-left, 0)
            bound = capacity//2+_LANES-1
            a = pairwise(values, left, starts, bound)
            b = pairwise(values, right, starts+left, bound)
            return jnp.where(split, barrier(a+b), a)

        def reduce(values: Any, counts: Any) -> Any:
            return barrier(pairwise(values, counts, jnp.zeros_like(counts), values.shape[1])+0.0)

        self._sum = self._jax.jit(reduce)

        def integrate(radiance: Any, cosine: Any, coverage: Any, coefficients: Any, counts: Any) -> Any:
            mu = jnp.maximum(cosine, 0)
            a = coefficients[..., 0]
            if law in ("linear", "cosine"):
                correction = barrier(barrier(1-a)+barrier(a*mu))
            else:
                correction = barrier(1-barrier(a*barrier(1-mu)))
                if law == "logarithmic":
                    term = barrier(barrier(coefficients[..., 1]*mu)*jnp.log(jnp.where(mu > 0, mu, 1)))
                else:
                    term = barrier(coefficients[..., 1]*barrier(1-jnp.sqrt(mu)))
                correction = barrier(correction-term)
            correction = jnp.where(cosine > 0, correction, 0)
            product = barrier(cosine*coverage)
            product = barrier(product*correction)
            faces = barrier(radiance*product)
            return faces, reduce(faces, counts)

        self._integrate = self._jax.jit(integrate)

    def prepare(
        self, radiance: Any, cosine: Any, coverage: Any, coefficients: Any, counts: Any,
    ) -> PreparedSurfaceBatch:
        """Validate/upload physical rows plus exact unpadded selected-face counts."""
        shape = np.shape(radiance)
        if len(shape) != _ROW_NDIM or shape[0] > _MAX_ROWS or shape[1] > _MAX_FACES:
            msg = "Ordered surface integration supports at most 16 rows/8192 padded faces"
            raise ValueError(msg)
        count = np.array(counts, copy=True)
        if (not np.issubdtype(count.dtype, np.integer) or count.shape != (shape[0],)
                or np.any(count < 0) or np.any(count > shape[1])):
            msg = "Selected counts must match rows and lie within the padded width"
            raise ValueError(msg)
        count = count.astype(np.int64)
        packet = self._validation.prepare(radiance, cosine, coverage, coefficients)
        return PreparedSurfaceBatch(self, (*packet._arrays, self._jax.device_put(count, self.device, may_alias=False)))
