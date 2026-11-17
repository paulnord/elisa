"""Fused radial graphs and checkpointed CPU precision decisions for intersections.

The current curved-intersection algorithm, physical equations and all accuracy
thresholds remain. Native nested kernels are an opt-in execution change. A
precision-ambiguous edge lane stops before that decision; original CPU arithmetic
resumes from the saved bracket and iteration. Nothing becomes a prior cut.
"""

# ruff: noqa: PLR2004 - composition of owned experimental kernels
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from elisa.experimental.gpu_batched_geometry import MAX_NODES
from elisa.experimental.gpu_native_observer import native_root_kernel
from elisa.experimental.gpu_observer_edges import ObserverDeviceLightCurve
from elisa.experimental.gpu_observer_geometry import ObserverGeometryBackend, _BoundGeometry, _Field

_EPS = np.finfo(float).eps


class FusedObserverGeometryBackend(ObserverGeometryBackend):
    """Same80radial iterations fused inside native per-tile device loops.

    CPU tails are declared before evaluation. A lane that encounters the
    unchanged128eps precision/branch trigger saves its pre-decision bracket and
    resumes on CPU. Resource/FP64/stationary errors retain original semantics.
    """

    def __init__(self, *, device: str = "gpu", max_nodes: int = MAX_NODES, cpu_tail_steps: int = 2) -> None:
        if isinstance(cpu_tail_steps, bool) or not isinstance(cpu_tail_steps, int) or not 0 <= cpu_tail_steps <= 8:
            msg = "Declare CPU tail steps as an integer0-8"
            raise ValueError(msg)
        super().__init__(device=device, max_nodes=max_nodes)
        self.cpu_tail_steps = cpu_tail_steps
        for key, field in self.fields.items():

            def make_roots(value: Any) -> Any:
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
                    stop = self.jnp.full(low.shape, steps, dtype=self.jnp.int32)

                    def update(k: Any, state: Any) -> Any:
                        lo, hi, valid, resume = state
                        mid = (lo + hi) / 2
                        v = value(parameters, a + d * mid[..., None], line, foreground_parameters)
                        active = resume == steps
                        uncertain = (self.jnp.abs(v["values"]) <= 128 * _EPS) | v["precision_branch"]
                        freeze = active & uncertain
                        resume = self.jnp.where(freeze, k, resume)
                        advance = active & ~uncertain
                        same = (v["values"] >= 0) == low_positive
                        return (
                            self.jnp.where(advance & same, mid, lo),
                            self.jnp.where(advance & ~same, mid, hi),
                            valid & (v["valid"] | ~active),
                            resume,
                        )

                    lo, hi, valid, resume = self.jax.lax.fori_loop(
                        0,
                        steps,
                        update,
                        (low, high, self.jnp.ones_like(low_positive), stop),
                    )
                    return {"low": lo, "high": hi, "valid": valid, "resume": resume}

                return self.jax.jit(solve, static_argnames=("steps",))

            self.roots[key] = native_root_kernel(self.jax, self.jnp, *key) if device == "gpu" else make_roots(field)

    def bind(self, binary: Any, component: str) -> Any:
        return _FusedBoundGeometry(self, binary, component)


class _FusedBoundGeometry(_BoundGeometry):
    def field(self, line: Any, foreground: Any = None) -> Any:
        return _ResumedField(self, line, foreground)


class _ResumedField(_Field):
    def bisect(self, a: Any, d: Any, low: Any, high: Any, low_positive: Any, steps: int) -> tuple:
        b = self.geometry.backend
        start = perf_counter()
        aa, n, pad = b.padded(a)
        dd, _, _ = b.padded(d)

        def place(v: Any) -> Any:
            return b.jax.device_put(np.pad(np.asarray(v)[None], ((0, 0), (0, pad - n)), mode="edge"), b.device)

        prefix = max(0, steps - b.cpu_tail_steps)
        result = b.checked(
            b.roots[self.geometry.component, self.kind](
                self.geometry.parameters,
                aa,
                dd,
                self.placed_line,
                self.foreground_parameters,
                place(low),
                place(high),
                place(low_positive),
                steps=prefix,
            ),
            n,
        )
        lo, hi = result["low"][0, :n].copy(), result["high"][0, :n].copy()
        resume = result["resume"][0, :n]
        # Replay only the original remaining decisions, from the first unsafe
        # decision (or the declared CPU tail). Never classify with a tolerance.
        for k in range(int(resume.min()) if n else steps, steps):
            take = resume <= k
            mid = (lo[take] + hi[take]) / 2
            same = (self.reference(a[take] + d[take] * mid[:, None]) >= 0) == low_positive[take]
            lo[take], hi[take] = np.where(same, mid, lo[take]), np.where(same, hi[take], mid)
        fl, fh = self.reference(a + d * lo[:, None]), self.reference(a + d * hi[:, None])
        replay = ((fl >= 0) != low_positive) | ((fh >= 0) == low_positive)
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
        b.count("CPU_tail_bisection_decisions", int(np.sum(steps - resume)))
        b.count("CPU_prefix_checkpoint_rows", int(np.sum(resume < prefix)))
        b.count("CPU_bracket_replay_rows", int(replay.sum()))
        b.count("edge_solve_seconds", perf_counter() - start)
        return lo, hi


class FusedObserverLightCurve(ObserverDeviceLightCurve):
    """Complete current intersection curves with fused graphs/resumed decisions.

    Reference mode is unchanged original CPU finite synthesis. Device states,
    CPU topology/clipping/LD bridges and complete timing boundaries remain.
    This does not qualify a mesh, physical fit or sampler at1e-4.
    """

    def __init__(self, *args: Any, cpu_tail_steps: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.device != "reference":
            self.observer_backend = FusedObserverGeometryBackend(device=self.device, cpu_tail_steps=cpu_tail_steps)
            self._geometry = self.observer_backend.bind
