"""Memory-bounded original irradiation for explicit finer circular meshes.

Only receiver rows are tiled; every donor contributes to each original sum.
No pair cutoff, temperature/LD approximation, fixed physical cache or continuum
extrapolation is introduced. CPU visibility/bolometric LD and original prefix
coefficient ordering remain those of DeviceReflection.
"""

from __future__ import annotations

from typing import Any

from elisa.experimental.gpu_reflection import DeviceReflection, UnsupportedReflectionError

_MAX_FACES, _MAX_PAIRS = 40000, 256 * 1024**2
_MIN_BLOCK, _MAX_BLOCK = 8, 64


class TiledDeviceReflection(DeviceReflection):
    """Recompute bounded coupling tiles during cooler-first sequential iteration.

    Uses <=64 receiver rows times the full donor count at once, instead of
    retaining quadratic distance/gamma/coupling matrices. This trades repeated
    geometry work for memory. Diagnostic outputs omit those full matrices.
    The total face/pair/iteration limits still bound computational work.
    """

    def __init__(self, law: str, *, device: str = "gpu", block_size: int = 32) -> None:
        if (isinstance(block_size, bool) or not isinstance(block_size, int)
                or not _MIN_BLOCK <= block_size <= _MAX_BLOCK):
            msg = "Reflection tile size must be an integer from 8 to 64"
            raise UnsupportedReflectionError(msg)
        super().__init__(law, device=device)
        self.block_size = block_size
        jax = self._jax
        import jax.numpy as jnp  # noqa: PLC0415 - optional dependency

        barrier = jax.lax.optimization_barrier

        def update(receiver: Any, donor: Any, factors: Any, ratio: Any) -> Any:
            c, n, _, _, _, _, sym, _, _ = receiver
            dc, dn, dt, da, dld, vis, _, _, lookup = donor
            count = len(sym)
            padded = ((count+block_size-1)//block_size)*block_size
            centres = jnp.pad(c[sym], ((0, padded-count), (0, 0)), mode="edge")
            normals = jnp.pad(n[sym], ((0, padded-count), (0, 0)), mode="edge")
            ratios = jnp.pad(ratio, (0, padded-count), mode="edge")
            vector = barrier(barrier(factors[lookup]*barrier(dt**4)[vis])*da[vis])

            def tile(index: Any, state: Any) -> Any:
                result, valid = state
                start = index*block_size
                distance, _, mu, q = self._geometry(
                    jax.lax.dynamic_slice(centres, (start, 0), (block_size, 3)), dc[vis],
                    jax.lax.dynamic_slice(normals, (start, 0), (block_size, 3)), dn[vis],
                )
                coupling = barrier(q*self._directional(mu, dld[:, :len(vis)]))
                incident = jnp.matmul(coupling, vector, precision=jax.lax.Precision.HIGHEST)
                correction = barrier(1+barrier(jax.lax.dynamic_slice(ratios, (start,), (block_size,))*incident))
                valid &= jnp.all(jnp.isfinite(coupling)) & jnp.all(distance > 0)
                return jax.lax.dynamic_update_slice(result, correction, (start,)), valid

            result, valid = jax.lax.fori_loop(0, padded//block_size, tile, (jnp.ones(padded), jnp.array(1, dtype=bool)))
            return result[:count], valid

        def evaluate(primary: Any, secondary: Any, parameters: Any, iterations: Any) -> dict[str, Any]:
            _, _, ta, _, lda, _, sa, ma, _ = primary
            _, _, tb, _, ldb, _, sb, mb, _ = secondary
            ia, ib = self._integral(lda), self._integral(ldb)
            ratio_a = barrier(barrier(parameters[0]/ia[sa])/barrier(ta**4)[sa])
            ratio_b = barrier(barrier(parameters[1]/ib[sb])/barrier(tb**4)[sb])

            def iteration(_: Any, state: Any) -> Any:
                a, b, valid = state

                def primary_first() -> Any:
                    new_a, va = update(primary, secondary, b, ratio_a)
                    new_b, vb = update(secondary, primary, new_a, ratio_b)
                    return new_a, new_b, valid & va & vb

                def secondary_first() -> Any:
                    new_b, vb = update(secondary, primary, a, ratio_b)
                    new_a, va = update(primary, secondary, new_b, ratio_a)
                    return new_a, new_b, valid & va & vb

                return jax.lax.cond(parameters[2] <= parameters[3], primary_first, secondary_first)

            # Validate geometry even for direct zero-iteration packet evaluation.
            ra, rb, valid = jax.lax.fori_loop(
                0, jnp.maximum(iterations, 1), iteration,
                (jnp.ones(sa.shape), jnp.ones(sb.shape), jnp.array(1, dtype=bool)),
            )
            ra, rb = jnp.where(iterations > 0, ra, 1.), jnp.where(iterations > 0, rb, 1.)
            a = barrier(ta.at[sa].set(barrier(ta[sa]*ra**0.25))[ma])
            b = barrier(tb.at[sb].set(barrier(tb[sb]*rb**0.25))[mb])
            valid &= (jnp.all(jnp.isfinite(a) & (a > 0)) & jnp.all(jnp.isfinite(b) & (b > 0))
                      & jnp.all(ia > 0) & jnp.all(ib > 0))
            return {"temperatures": (a, b), "reflection_factor": (ra, rb), "proposal_valid": valid}

        self._evaluate = jax.jit(evaluate)

    def _limits(self) -> tuple[int, int]:
        return _MAX_FACES, _MAX_PAIRS
