"""Device radial Roche-surface solver for a bounded detached-star domain.

This is changing-geometry physics, NOT a complete light-curve backend. It solves
ELISa's modified Kopal potential at supplied directions, with actual mass ratio,
synchronicity and surface potential varying across a batch. Triangulation,
coverage, temperatures, atmosphere/LD and observation synthesis are not included.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from elisa.experimental.gpu_likelihood import _jax_device

_LOWER = 1e-6
_UPPER = 0.25
_ITERATIONS = 80
_Q_MIN, _Q_MAX = 0.5, 2.0
_S_MIN, _S_MAX = 0.5, 1.5


class RocheRaySolver:
    """Solve radii in SMA units with circular separation fixed to one.

    Inner-branch ray primitive, not a full detached mesh generator.
    Domain q∈[0.5,2], synchronicity∈[0.5,1.5], r∈[1e-6,0.25]. On this
    interval the radial potential derivative is strictly negative for every
    direction. Endpoint sign checks therefore certify a unique inner root.
    Unsupported roots are flagged, never replaced by a spherical approximation.
    This targets small inner-branch rays; it is not validated for
    near-lobe-filling/overcontact surfaces.
    Validity certifies only supplied rays, not a closed surface or morphology.
    ELISa candidate-dependent angle placement and mesh correction are absent.
    Device storage/temporaries scale as proposals times directions; callers
    must chunk large batches to their allocation budget.
    """

    def __init__(self, component: str, *, device: str = "gpu") -> None:
        if component not in ("primary", "secondary"):
            msg = "Component must be primary or secondary"
            raise ValueError(msg)
        self.component = component
        self._jax, jnp, self.device = _jax_device(device)

        def solve(q: Any, sync: Any, target: Any, phi: Any, theta: Any) -> dict[str, Any]:
            q, sync, target = q[:, None], sync[:, None], target[:, None]
            sin_theta = jnp.sin(theta)[None, :]
            cs = jnp.cos(phi)[None, :] * sin_theta
            c = 2 * cs
            d = q * cs if component == "primary" else cs
            e = 0.5 * sync**2 * (1 + q) * sin_theta**2

            def potential(r: Any) -> Any:
                r2 = r**2
                separation = jnp.sqrt(1 + r2 - c * r)
                if component == "primary":
                    return 1 / r + q / separation - d * r + e * r2
                return q / r + 1 / separation - d * r + e * r2 + (0.5 - 0.5 * q)

            shape = (q.shape[0], phi.shape[0])
            low, high = jnp.full(shape, _LOWER), jnp.full(shape, _UPPER)
            flo, fhi = potential(low) - target, potential(high) - target
            # Endpoint uncertainty is unsupported, not a physical rejection.
            endpoint_tol = 32 * jnp.finfo(jnp.float64).eps * jnp.maximum(1, target)
            ambiguous = (jnp.abs(flo) <= endpoint_tol) | (jnp.abs(fhi) <= endpoint_tol)
            bracketed = (flo > endpoint_tol) & (fhi < -endpoint_tol) & jnp.isfinite(flo) & jnp.isfinite(fhi)

            def bisect(_i: Any, bounds: Any) -> Any:
                lo, hi = bounds
                mid = lo + 0.5 * (hi - lo)
                above = potential(mid) > target
                return jnp.where(above, mid, lo), jnp.where(above, hi, mid)

            low, high = self._jax.lax.fori_loop(0, _ITERATIONS, bisect, (low, high))
            radius = low + 0.5 * (high - low)
            residual = potential(radius) - target
            valid = bracketed & jnp.isfinite(residual) & (jnp.abs(residual) <= 1e-12 * jnp.maximum(1, target))
            status = jnp.where(valid, 0, jnp.where(ambiguous, 2, jnp.where(bracketed, 3, 1)))
            radius = jnp.where(valid, radius, jnp.nan)
            x = radius * sin_theta * jnp.cos(phi)[None, :]
            y = radius * sin_theta * jnp.sin(phi)[None, :]
            z = radius * jnp.cos(theta)[None, :]
            return {
                "radii": radius,
                "status": status,
                "points_local": jnp.stack((x, y, z), axis=-1),
                "potential_residual": residual,
                "valid": valid,
                "proposal_valid": jnp.all(valid, axis=1),
                "radius_interval_width": high - low,
            }

        self._solve = self._jax.jit(solve)

    def __call__(self, q: Any, synchronicity: Any, potential: Any, phi: Any, theta: Any) -> dict[str, Any]:
        """Return device radii/points, residuals and explicit validity masks.

        Parameters are equally sized one-dimensional proposal arrays; directions
        are matching one-dimensional radian arrays shared across proposals.
        Theta is colatitude from the north pole. q=M_secondary/M_primary for
        BOTH components. 80 bisections are limited by floating-point arithmetic, not 2^-80
        accuracy; inspect residual and interval width.
        Check proposal_valid before using any model. Status: 0=accepted ray,
        1=outside this primitive's bracket, 2=ambiguous near endpoint,
        3=numerical residual failure. False masks mean UNSUPPORTED, not
        physically impossible: a sampler must fall back to CPU or fail, never
        silently assign -infinity unless that restricted prior was intentional.
        Endpoint ambiguity uses 32*eps*max(1,target); no gradients are validated
        for this branched bisection (not an HMC/autodiff implementation). points_local uses each
        component's local +x axis toward its companion, as ELISa spherical mesh
        helpers do; it is not the secondary's final binary-frame placement.
        """
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        arrays = [np.array(v, dtype=np.float64, copy=True) for v in (q, synchronicity, potential, phi, theta)]
        qq, ss, pp, ph, th = arrays
        if any(v.ndim != 1 or v.size == 0 or not np.isfinite(v).all() for v in arrays):
            msg = "Inputs must be nonempty finite one-dimensional arrays"
            raise ValueError(msg)
        if qq.shape != ss.shape or qq.shape != pp.shape or ph.shape != th.shape:
            msg = "Proposal shapes and direction shapes must match"
            raise ValueError(msg)
        if np.any((qq < _Q_MIN) | (qq > _Q_MAX) | (ss < _S_MIN) | (ss > _S_MAX) | (pp <= 0)):
            msg = "Unsupported domain: q in [0.5,2], synchronicity in [0.5,1.5], potential positive"
            raise ValueError(msg)
        if np.any((ph < 0) | (ph > 2 * np.pi) | (th < 0) | (th > np.pi)):
            msg = "Angles require phi in [0,2pi] and theta in [0,pi]"
            raise ValueError(msg)
        return self._solve(*(self._jax.device_put(v, self.device, may_alias=False) for v in arrays))
