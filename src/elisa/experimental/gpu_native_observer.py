"""Native FP64 nested Roche/observer roots for the existing intersection path.

Pallas/Triton runs the bounded radial, stationary-shadow and edge loops inside
one device program per tile. Explicit round-to-nearest scalar operations retain
stage rounding; no approximate math, TF32 or tensor cores are requested. This
optional path needs JAX's CUDA Pallas backend. Existing CPU/default APIs do not
import Pallas or change their equations, protocols or failure semantics.
"""

# ruff: noqa: C901, PLR0915, PLR2004, PLC0415 - literal fixed physical protocol
from __future__ import annotations

from typing import Any

import numpy as np

_EPS = np.finfo(float).eps
_TILE = 128


def _arithmetic(jax: Any, jnp: Any, *, native: bool) -> tuple:
    """Prevent contraction/reassociation across every declared scalar stage."""
    if native:
        from jax.experimental.pallas import triton as plt

        def operation(name: str, *args: Any) -> Any:
            values = jnp.broadcast_arrays(*[jnp.asarray(a, dtype=jnp.float64) for a in args])
            code = f"{name}.rn.f64 $0, " + ", ".join(f"${i + 1}" for i in range(len(values))) + ";"
            return plt.elementwise_inline_asm(
                code,
                args=values,
                constraints="=d," + ",".join("d" for _ in values),
                pack=1,
                result_shape_dtypes=[jax.ShapeDtypeStruct(values[0].shape, jnp.float64)],
            )[0]

        return tuple((lambda *a, _n=n: operation(_n, *a)) for n in ["add", "sub", "mul", "div", "sqrt"])
    barrier = jax.lax.optimization_barrier
    return tuple(
        (lambda *a, _f=f: barrier(_f(*a))) for f in [jnp.add, jnp.subtract, jnp.multiply, jnp.divide, jnp.sqrt]
    )


def field_kernel(jax: Any, jnp: Any, component: str, kind: str, *, native: bool) -> Any:
    """Retain the physical domain and 80/12 updates in a scalar-axis graph."""
    add, sub, mul, div, sqrt = _arithmetic(jax, jnp, native=native)
    primary = component == "primary"

    def dot(x: Any, y: Any) -> Any:
        return add(add(mul(x[0], y[0]), mul(x[1], y[1])), mul(x[2], y[2]))

    def norm(x: Any) -> Any:
        return sqrt(dot(x, x))

    def properties(p: Any, *, first: bool) -> tuple:
        q, spin, target = p
        mass, other = (1.0, q) if first else (q, 1.0)
        rotation = mul(mul(spin, spin), add(1.0, q))
        return q, target, mass, other, rotation, (-q if first else 1.0), (0.0 if first else sub(0.5, mul(0.5, q)))

    def gradient(local: Any, p: Any, *, first: bool) -> Any:
        _q, _target, mass, other, rotation, linear, _extra = properties(p, first=first)
        companion = (add(local[0], -1.0 if first else 1.0), local[1], local[2])
        r, c = norm(local), norm(companion)
        r3, c3 = mul(mul(r, r), r), mul(mul(c, c), c)
        return tuple(
            sub(
                sub(add(div(mul(mass, x), r3), div(mul(other, y), c3)), mul(rotation, x) if k < 2 else 0.0),
                linear if k == 0 else 0.0,
            )
            for k, (x, y) in enumerate(zip(local, companion, strict=True))
        )

    def sample(p: Any, vectors: Any) -> tuple:
        q, target, mass, other, rotation, linear, extra = properties(p, first=primary)
        vnorm = norm(vectors)
        direction = tuple(div(v, jnp.where(vnorm > 0, vnorm, 1.0)) for v in vectors)
        xy = add(mul(direction[0], direction[0]), mul(direction[1], direction[1]))

        def potential(r: Any) -> Any:
            r2 = mul(r, r)
            sep = sqrt(add(add(1.0, r2), mul(mul(-2.0 if primary else 2.0, r), direction[0])))
            value = add(div(mass, r), div(other, sep))
            value = add(value, mul(mul(linear, r), direction[0]))
            value = add(value, mul(mul(mul(0.5, rotation), r2), xy))
            return add(value, extra)

        lo, hi = jnp.full_like(vnorm, 1e-6), jnp.full_like(vnorm, 0.25)
        fl, fh = sub(potential(lo), target), sub(potential(hi), target)
        tolerance = mul(32 * _EPS, jnp.maximum(1.0, target))
        valid = (q >= 0.5) & (q <= 2.0) & (p[1] >= 0.5) & (p[1] <= 1.5) & (target > 0)
        valid &= jnp.isfinite(q + p[1] + target) & jnp.isfinite(vnorm) & (vnorm > 0)
        valid &= (fl > tolerance) & (fh < -tolerance) & jnp.isfinite(fl) & jnp.isfinite(fh)

        def update(_: Any, bounds: Any) -> Any:
            low, high = bounds
            mid = add(low, mul(0.5, sub(high, low)))
            above = potential(mid) > target
            return jnp.where(above, mid, low), jnp.where(above, high, mid)

        lo, hi = jax.lax.fori_loop(0, 80, update, (lo, hi))
        r = add(lo, mul(0.5, sub(hi, lo)))
        local = tuple(mul(r, v) for v in direction)
        g = gradient(local, p, first=primary)
        magnitude, radial = norm(g), dot(g, direction)
        valid &= jnp.isfinite(magnitude) & (magnitude > 0) & (radial > 0)
        residual = sub(potential(r), target)
        valid &= jnp.isfinite(residual) & (jnp.abs(residual) <= mul(3e-13, jnp.maximum(1.0, target)))
        point = (add(local[0], 0.0 if primary else 1.0), local[1], local[2])
        normal = tuple(div(v, jnp.where(magnitude > 0, magnitude, 1.0)) for v in g)
        return point, normal, valid

    def shadow(p: Any, points: Any, line: Any) -> tuple:
        first = not primary
        q, target, mass, other, rotation, linear, extra = properties(p, first=first)
        inner = div(mass, target)
        delta = (sub(points[0], 0.0 if first else 1.0), points[1], points[2])
        along = dot(delta, line)
        base = tuple(sub(v, mul(along, los)) for v, los in zip(delta, line, strict=True))
        rho = norm(base)
        selected = (rho >= inner) & (rho < 0.25)
        scratch = tuple(jnp.where(selected, v, f) for v, f in zip(base, [0.0, 0.1, 0.0], strict=True))
        rhos = jnp.where(selected, rho, 0.1)

        def update(_: Any, state: Any) -> Any:
            t, ok = state
            local = tuple(add(v, mul(t, los)) for v, los in zip(scratch, line, strict=True))
            g = gradient(local, p, first=first)
            companion = (add(local[0], -1.0 if first else 1.0), local[1], local[2])
            derivative = jnp.zeros_like(t)
            for v, m in [(local, mass), (companion, other)]:
                r2, projection = dot(v, v), dot(v, line)
                # r2**2.5 from the original derivative, with explicit FP64 sqrt.
                denominator = mul(mul(r2, r2), sqrt(r2))
                derivative = add(derivative, div(mul(m, sub(r2, mul(3.0, mul(projection, projection)))), denominator))
            derivative = sub(derivative, mul(rotation, add(mul(line[0], line[0]), mul(line[1], line[1]))))
            good = (derivative > 0) & jnp.isfinite(derivative)
            step = div(dot(g, line), jnp.where(good, derivative, 1.0))
            bound = div(rhos, 4.0)
            t = sub(t, jnp.clip(step, -bound, bound))
            return t, ok & (good | ~selected)

        t, ok = jax.lax.fori_loop(0, 12, update, (jnp.zeros_like(rho), jnp.ones_like(selected)))
        local = tuple(add(v, mul(t, los)) for v, los in zip(scratch, line, strict=True))
        g = gradient(local, p, first=first)
        residual = jnp.abs(dot(g, line))
        ok &= ~selected | (
            jnp.isfinite(norm(g)) & (jnp.abs(t) <= div(rhos, 2.0)) & (residual <= mul(64 * _EPS, norm(g)))
        )
        companion = (add(local[0], -1.0 if first else 1.0), local[1], local[2])
        value = add(div(mass, norm(local)), div(other, norm(companion)))
        value = add(value, mul(mul(0.5, rotation), add(mul(local[0], local[0]), mul(local[1], local[1]))))
        value = add(add(value, mul(linear, local[0])), extra)
        distance = jnp.where(selected, div(mul(mul(sub(target, value), rhos), rhos), mass), sub(rho, inner))
        domain = jnp.isfinite(q + p[1] + target) & (q >= 0.5) & (q <= 2.0) & (p[1] >= 0.5) & (p[1] <= 1.5)
        domain &= (inner > 1e-6) & (inner < 0.25)
        ok &= domain & jnp.isfinite(distance)
        branch = (jnp.abs(sub(rho, inner)) <= 128 * _EPS) | (jnp.abs(sub(rho, 0.25)) <= 128 * _EPS)
        return distance, ok, branch

    def evaluate(p: Any, vectors: Any, line: Any, foreground: Any) -> tuple:
        point, normal, valid = sample(p, vectors)
        if kind == "limb":
            return dot(normal, line), valid, jnp.zeros_like(valid)
        distance, ok, branch = shadow(foreground, point, line)
        return distance, valid & ok, branch

    return evaluate


def native_root_kernel(jax: Any, jnp: Any, component: str, kind: str) -> Any:
    """One launch per padded proposal/tile grid, including every nested loop."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    field = field_kernel(jax, jnp, component, kind, native=True)

    def solve(
        parameters: Any,
        a: Any,
        d: Any,
        line: Any,
        foreground: Any,
        low: Any,
        high: Any,
        positive: Any,
        *,
        steps: int,
    ) -> dict:
        proposals, nodes = low.shape

        def kernel(
            p_ref: Any,
            a_ref: Any,
            d_ref: Any,
            l_ref: Any,
            f_ref: Any,
            lo_ref: Any,
            hi_ref: Any,
            s_ref: Any,
            out_lo: Any,
            out_hi: Any,
            out_valid: Any,
            out_resume: Any,
        ) -> None:
            b = pl.program_id(0)
            i = pl.program_id(1) * _TILE + jnp.arange(_TILE)
            mask = i < nodes
            p = tuple(jnp.broadcast_to(p_ref[b, k], (_TILE,)) for k in range(3))
            fg = tuple(jnp.broadcast_to(f_ref[b, k], (_TILE,)) for k in range(3))
            ln = tuple(jnp.broadcast_to(l_ref[b, k], (_TILE,)) for k in range(3))
            aa = tuple(plt.load(a_ref.at[b, i, k], mask=mask, other=1.0 if k == 0 else 0.0) for k in range(3))
            dd = tuple(plt.load(d_ref.at[b, i, k], mask=mask, other=0.0) for k in range(3))
            lo = plt.load(lo_ref.at[b, i], mask=mask, other=0.0)
            hi = plt.load(hi_ref.at[b, i], mask=mask, other=1.0)
            sign = plt.load(s_ref.at[b, i], mask=mask, other=False)
            add, _sub, mul, div, _sqrt = _arithmetic(jax, jnp, native=True)

            def update(k: Any, state: Any) -> Any:
                left, right, valid, resume = state
                mid = div(add(left, right), 2.0)
                vectors = tuple(add(v, mul(w, mid)) for v, w in zip(aa, dd, strict=True))
                value, good, branch = field(p, vectors, ln, fg)
                active = resume == steps
                uncertain = (jnp.abs(value) <= 128 * _EPS) | branch
                stop = active & uncertain
                advance = active & ~uncertain
                same = (value >= 0) == sign
                return (
                    jnp.where(advance & same, mid, left),
                    jnp.where(advance & ~same, mid, right),
                    valid & (good | ~active),
                    jnp.where(stop, k, resume),
                )

            lo, hi, valid, resume = jax.lax.fori_loop(
                0,
                steps,
                update,
                (lo, hi, jnp.ones((_TILE,), dtype=bool), jnp.full((_TILE,), steps, dtype=jnp.int32)),
            )
            for ref, value in [(out_lo, lo), (out_hi, hi), (out_valid, valid.astype(jnp.int32)), (out_resume, resume)]:
                plt.store(ref.at[b, i], value, mask=mask)

        def shape(dt: Any) -> Any:
            return jax.ShapeDtypeStruct((proposals, nodes), dt)

        out = pl.pallas_call(
            kernel,
            out_shape=(shape(jnp.float64), shape(jnp.float64), shape(jnp.int32), shape(jnp.int32)),
            grid=(proposals, (nodes + _TILE - 1) // _TILE),
            compiler_params=plt.CompilerParams(num_warps=4),
            name=f"roche_{component}_{kind}_nested",
        )(parameters, a, d, line, foreground, low, high, positive)
        return {"low": out[0], "high": out[1], "valid": out[2].astype(bool), "resume": out[3]}

    return jax.jit(solve, static_argnames=("steps",))
