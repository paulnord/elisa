"""Fresh device observer rays, conditioned silhouettes and nested edge solves.

CPU adaptation/clipping and exact original final-bracket verification remain
explicit. No CPU callback runs inside a compiled graph. Precision-ambiguous
seams and failed final brackets use original CPU arithmetic, with counted costs;
physical/domain/resource failures always abort outside priors.
"""

# ruff: noqa: PLR2004, PLR0915 - owned experimental composition
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_batched_geometry import MAX_NODES, BatchedRocheGeometry

_EPS = np.finfo(float).eps


def shadow_kernel(geometry: BatchedRocheGeometry) -> Any:
    """Original12Newton updates and64eps stationary gate, star-local FP64."""
    jnp, jax = geometry.jnp, geometry.jax
    primary = geometry.component == "primary"
    centre = jnp.array([0.0 if primary else 1.0, 0.0, 0.0])
    shift = jnp.array([-1.0 if primary else 1.0, 0.0, 0.0])
    barrier = jax.lax.optimization_barrier

    def evaluate(parameters: Any, points: Any, lines: Any) -> dict:
        q, spin, target = [parameters[:, i : i + 1] for i in range(3)]
        mass = jnp.ones_like(q) if primary else q
        other = q if primary else jnp.ones_like(q)
        inner = mass / target
        line = lines[:, None, :]
        delta = points - centre
        base = delta - jnp.sum(delta * line, axis=-1)[..., None] * line
        rho = jnp.linalg.norm(base, axis=-1)
        selected = (rho >= inner) & (rho < 0.25)
        scratch = jnp.where(selected[..., None], base, jnp.array([0.0, 0.1, 0.0]))
        rhos = jnp.where(selected, rho, 0.1)

        def gradient(local: Any) -> Any:
            companion = local + shift
            g = barrier(mass[..., None] * local / jnp.linalg.norm(local, axis=-1)[..., None] ** 3)
            g += barrier(other[..., None] * companion / jnp.linalg.norm(companion, axis=-1)[..., None] ** 3)
            g -= barrier(spin[..., None] ** 2 * (1 + q[..., None]) * local * jnp.array([1.0, 1.0, 0.0]))
            return (
                g
                + jnp.concatenate([q if primary else -jnp.ones_like(q), jnp.zeros_like(q), jnp.zeros_like(q)], axis=1)[
                    :,
                    None,
                    :,
                ]
            )

        def update(_: Any, state: Any) -> Any:
            t, ok = state
            local = barrier(scratch + t[..., None] * line)
            g = gradient(local)
            derivative = jnp.zeros_like(t)
            for v, m in [(local, mass), (local + shift, other)]:
                r2 = jnp.sum(v * v, axis=-1)
                derivative += m * (r2 - 3 * jnp.sum(v * line, axis=-1) ** 2) / r2**2.5
            derivative -= spin**2 * (1 + q) * jnp.sum(line[..., :2] ** 2, axis=-1)
            good = (derivative > 0) & jnp.isfinite(derivative)
            step = barrier(jnp.sum(g * line, axis=-1) / jnp.where(good, derivative, 1.0))
            t = barrier(t - jnp.clip(step, -rhos / 4, rhos / 4))
            return t, ok & (good | ~selected)

        t, ok = jax.lax.fori_loop(0, 12, update, (jnp.zeros_like(rho), jnp.ones_like(selected)), unroll=4)
        local = barrier(scratch + t[..., None] * line)
        g = gradient(local)
        residual = jnp.abs(jnp.sum(g * line, axis=-1))
        good = jnp.all(jnp.isfinite(g), axis=-1) & (jnp.abs(t) <= rhos / 2)
        good &= residual <= 64 * _EPS * jnp.linalg.norm(g, axis=-1)
        ok &= good | ~selected
        radius = jnp.linalg.norm(local, axis=-1)
        companion = jnp.linalg.norm(local + shift, axis=-1)
        value = mass / radius + other / companion
        value += 0.5 * spin**2 * (1 + q) * jnp.sum(local[..., :2] ** 2, axis=-1)
        value += (-q if primary else jnp.ones_like(q)) * local[..., 0]
        value += 0.0 if primary else 0.5 - 0.5 * q
        inside = jnp.where(selected, value > target, rho < inner)
        distance = jnp.where(selected, (target - value) * rhos * rhos / mass, rho - inner)
        domain = jnp.all(jnp.isfinite(parameters), axis=1)[:, None]
        domain &= (q >= 0.5) & (q <= 2) & (spin >= 0.5) & (spin <= 1.5) & (inner > 1e-6) & (inner < 0.25)
        ok &= domain & jnp.isfinite(distance) & jnp.all(jnp.isfinite(points), axis=-1)
        return {
            "blocked": inside,
            "precision_branch": (jnp.abs(rho - inner) <= 128 * _EPS) | (jnp.abs(rho - 0.25) <= 128 * _EPS),
            "distance": distance,
            "valid": ok,
            "stationary_relative_residual": jnp.where(selected, residual / jnp.linalg.norm(g, axis=-1), 0.0),
        }

    return evaluate


class ObserverGeometryBackend:
    """Reusable compiled shapes, freshly bound physical rows and LOS every call."""

    def __init__(self, *, device: str = "gpu", max_nodes: int = MAX_NODES) -> None:
        self.geometry = {
            n: BatchedRocheGeometry(n, device=device, max_nodes=max_nodes, root_unroll=8)
            for n in ["primary", "secondary"]
        }
        self.jax, self.jnp, self.device = (
            self.geometry["primary"].jax,
            self.geometry["primary"].jnp,
            self.geometry["primary"].device,
        )
        self.max_nodes = max_nodes
        self.shadows = {n: shadow_kernel(g) for n, g in self.geometry.items()}
        self.samples = {n: self.jax.jit(g.sample_kernel) for n, g in self.geometry.items()}
        self.shadow_calls = {n: self.jax.jit(f) for n, f in self.shadows.items()}
        self.fields, self.roots = {}, {}
        self.stats = {}
        for name in self.geometry:
            other = "secondary" if name == "primary" else "primary"
            for kind in ["limb", "shadow"]:

                def make_field(component: str, foreground: str, mode: str) -> Any:
                    def evaluate_field(parameters: Any, vectors: Any, line: Any, foreground_parameters: Any) -> dict:
                        sample = self.geometry[component].sample_kernel(
                            parameters,
                            vectors,
                            self.jnp.ones(vectors.shape[:2], dtype=bool),
                        )
                        if mode == "limb":
                            return {
                                "values": self.jnp.sum(sample["normals"] * line[:, None, :], axis=-1),
                                "valid": sample["valid"],
                                "precision_branch": self.jnp.zeros_like(sample["valid"]),
                            }
                        shade = self.shadows[foreground](foreground_parameters, sample["points"], line)
                        return {
                            "values": shade["distance"],
                            "valid": sample["valid"] & shade["valid"],
                            "precision_branch": shade["precision_branch"],
                        }

                    return evaluate_field

                value = make_field(name, other, kind)
                self.fields[name, kind] = self.jax.jit(value)

                def make_roots(field: Any) -> Any:
                    def solve(
                        parameters: Any,
                        a: Any,
                        d: Any,
                        line: Any,
                        foreground_parameters: Any,
                        low: Any,
                        high: Any,
                        low_positive: Any,
                        *,
                        steps: int,
                    ) -> dict:
                        def update(_: Any, state: Any) -> Any:
                            lo, hi, valid, precision = state
                            mid = (lo + hi) / 2
                            r = field(parameters, a + d * mid[..., None], line, foreground_parameters)
                            same = (r["values"] >= 0) == low_positive
                            return (
                                self.jnp.where(same, mid, lo),
                                self.jnp.where(same, hi, mid),
                                valid & r["valid"],
                                precision | (self.jnp.abs(r["values"]) <= 128 * _EPS) | r["precision_branch"],
                            )

                        lo, hi, valid, precision = self.jax.lax.fori_loop(
                            0,
                            steps,
                            update,
                            (low, high, self.jnp.ones_like(low_positive), self.jnp.zeros_like(low_positive)),
                        )
                        return {"low": lo, "high": hi, "valid": valid, "precision_hit": precision}

                    return self.jax.jit(solve, static_argnames=("steps",))

                self.roots[name, kind] = make_roots(value)

    def bind(self, binary: Any, component: str) -> Any:
        return _BoundGeometry(self, binary, component)

    def count(self, key: str, amount: float = 1.0) -> None:
        self.stats[key] = self.stats.get(key, 0.0) + amount

    def padded(self, value: Any) -> tuple:
        if not self.jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled for observer geometry"
            raise RuntimeError(msg)
        v = np.asarray(value, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
            msg = "Observer geometry requires finite(N,3)coordinates"
            raise UnsupportedCurvedGeometryError(msg)
        n = len(v)
        if n > self.max_nodes:
            msg = "Observer geometry node cap exceeded before allocation"
            raise UnsupportedCurvedGeometryError(msg)
        pad = 1 << max(0, n - 1).bit_length()
        # Bounded powers of two avoid recompiling every adaptive leaf count.
        if pad > self.max_nodes:
            msg = "Observer padded node cap exceeded before allocation"
            raise UnsupportedCurvedGeometryError(msg)
        padded = np.empty((1, pad, 3))
        padded[:, :n] = v
        padded[:, n:] = v[0] if n else [1.0, 0.0, 0.0]
        return self.jax.device_put(padded, self.device), n, pad

    def checked(self, result: dict, n: int) -> dict:
        host = self.jax.tree.map(np.asarray, result)
        if not np.all(host["valid"][0, :n]):
            msg = "Observer geometry failed the original inner-branch/stationary FP64 gates"
            raise UnsupportedCurvedGeometryError(msg)
        return host


class _BoundGeometry(ConditionedRocheGeometry):
    def __init__(self, backend: ObserverGeometryBackend, binary: Any, component: str) -> None:
        super().__init__(binary, component)
        self.backend = backend
        self.parameters = backend.jax.device_put(
            np.array([[binary.mass_ratio, self.star.synchronicity, self.star.surface_potential]]),
            backend.device,
        )
        self.reference = ConditionedRocheGeometry(binary, component)
        self.line = None

    def sample(self, vectors: Any) -> tuple:
        start = perf_counter()
        b = self.backend
        placed, n, pad = b.padded(vectors)
        mask = b.jax.device_put(np.arange(pad)[None, :] < n, b.device)
        r = b.checked(b.samples[self.component](self.parameters, placed, mask), n)
        values = [r[k][0, :n].copy() for k in ["points", "normals", "area_per_solid_angle"]]
        if n and self.line is not None:
            ambiguous = np.abs(values[1] @ self.line) <= 128 * _EPS
            if ambiguous.any():
                original = self.reference.sample(np.asarray(vectors)[ambiguous])
                for a, v in zip(values, original, strict=True):
                    a[ambiguous] = v
                b.count("CPU_precision_seam_rows", int(ambiguous.sum()))
        b.count("sample_calls")
        b.count("sample_padded_nodes", pad)
        b.count("sample_seconds", perf_counter() - start)
        return tuple(values)

    def shadow(self, points: Any, line: Any) -> tuple:
        b = self.backend
        value = np.asarray(line)
        if value.shape != (3,) or not np.isfinite(value).all() or abs(np.linalg.norm(value) - 1) > 64 * _EPS:
            msg = "Observer shadow requires a finite unit LOS"
            raise UnsupportedCurvedGeometryError(msg)
        placed, n, _pad = b.padded(points)
        lines = b.jax.device_put(value[None], b.device)
        r = b.checked(b.shadow_calls[self.component](self.parameters, placed, lines), n)
        blocked, distance = r["blocked"][0, :n].copy(), r["distance"][0, :n].copy()
        ambiguous = (np.abs(distance) <= 128 * _EPS) | r["precision_branch"][0, :n]
        if ambiguous.any():
            original = self.reference.shadow(np.asarray(points)[ambiguous], value)
            blocked[ambiguous], distance[ambiguous] = original
            b.count("CPU_precision_shadow_rows", int(ambiguous.sum()))
        b.count("shadow_calls")
        return blocked, distance

    def field(self, line: Any, foreground: Any = None) -> Any:
        return _Field(self, line, foreground)


class _Field:
    def __init__(self, geometry: _BoundGeometry, line: Any, foreground: Any = None) -> None:
        self.geometry, self.line, self.foreground = geometry, np.asarray(line), foreground
        self.kind = "limb" if foreground is None else "shadow"
        b = geometry.backend
        self.placed_line = b.jax.device_put(self.line[None], b.device)
        self.foreground_parameters = geometry.parameters if foreground is None else foreground.parameters

    def reference(self, vectors: Any) -> Any:
        g = self.geometry.reference
        if self.foreground is None:
            return g.sample(vectors)[1] @ self.line
        return self.foreground.reference.shadow(g.sample(vectors)[0], self.line)[1]

    def __call__(self, vectors: Any) -> Any:
        b = self.geometry.backend
        placed, n, _ = b.padded(vectors)
        r = b.checked(
            b.fields[self.geometry.component, self.kind](
                self.geometry.parameters,
                placed,
                self.placed_line,
                self.foreground_parameters,
            ),
            n,
        )
        values = r["values"][0, :n].copy()
        ambiguous = (np.abs(values) <= 128 * _EPS) | r["precision_branch"][0, :n]
        if ambiguous.any():
            values[ambiguous] = self.reference(np.asarray(vectors)[ambiguous])
            b.count("CPU_precision_field_rows", int(ambiguous.sum()))
        return values

    def bisect(self, a: Any, d: Any, low: Any, high: Any, low_positive: Any, steps: int) -> tuple:
        b = self.geometry.backend
        start = perf_counter()
        aa, n, pad = b.padded(a)
        dd, _, _ = b.padded(d)

        def place(v: Any) -> Any:
            return b.jax.device_put(np.pad(np.asarray(v)[None], ((0, 0), (0, pad - n)), mode="edge"), b.device)

        r = b.checked(
            b.roots[self.geometry.component, self.kind](
                self.geometry.parameters,
                aa,
                dd,
                self.placed_line,
                self.foreground_parameters,
                place(low),
                place(high),
                place(low_positive),
                steps=max(0, steps - 8),
            ),
            n,
        )
        lo, hi = r["low"][0, :n].copy(), r["high"][0, :n].copy()
        # Original normals have small FP64 plateaus near tangency: a final
        # bracket alone need not identify the original bisection path. Keep
        # the last8decisions in the original arithmetic, after a device prefix.
        # This preserves the finite CPU protocol, without a tolerance cut.
        for _ in range(min(8, steps)):
            mid = (lo + hi) / 2
            same = (self.reference(a + d * mid[:, None]) >= 0) == low_positive
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        b.count("CPU_tail_bisection_decisions", n * min(8, steps))

        # Verify the final dyadic bracket in exact original CPU arithmetic.
        # These checks/counted replays are included in every enclosing timer.
        fl, fh = self.reference(a + d * lo[:, None]), self.reference(a + d * hi[:, None])
        replay = r["precision_hit"][0, :n] | ((fl >= 0) != low_positive) | ((fh >= 0) == low_positive)
        if replay.any():
            al, dl = a[replay], d[replay]
            lp = low_positive[replay]
            left, right = np.array(low[replay], copy=True), np.array(high[replay], copy=True)
            for _ in range(steps):
                mid = (left + right) / 2
                same = (self.reference(al + dl * mid[:, None]) >= 0) == lp
                left, right = np.where(same, mid, left), np.where(same, right, mid)
            lo[replay], hi[replay] = left, right
        b.count("edge_solve_calls")
        b.count("edge_root_rows", n)
        b.count("CPU_bracket_replay_rows", int(replay.sum()))
        b.count("edge_solve_seconds", perf_counter() - start)
        return lo, hi
