"""Fresh mutual irradiation on device with original ELISa iteration semantics.

CPU selects the original visibility/symmetry indices and bolometric LD sources.
Distances, directional LD, coupling matrices, cooler-first sequential updates
and mirrored final temperatures execute in FP64 on the requested device.
The original coefficient prefix order in the symmetry helper is intentional.
"""

# ruff: noqa: SLF001 - composition of experimental immutable device snapshots
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa import const, ld
from elisa.binary_system.surface import faces
from elisa.experimental.gpu_likelihood import _jax_device

_COMPONENTS = ("primary", "secondary")
_MAX_FACES, _MAX_PAIRS, _MAX_ITERATIONS = 8192, 2097152, 32


class UnsupportedReflectionError(ValueError):
    """A device reflection contract/resource failure requires full CPU synthesis."""


@dataclass(frozen=True)
class PreparedReflection:
    """Immutable current-candidate snapshot; evaluate performs no host transfers."""

    _owner: Any
    _arrays: tuple[Any, ...]
    visible_indices: tuple[np.ndarray, ...]
    symmetry_indices: tuple[np.ndarray, ...]

    def evaluate(self) -> dict[str, Any]:
        if not self._owner._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        return self._owner._evaluate(*self._arrays)

    def block_until_ready(self) -> PreparedReflection:
        for item in self._arrays:
            for array in item if isinstance(item, tuple) else (item,):
                array.block_until_ready()
        return self


class DeviceReflection:
    """Original symmetric circular detached reflection, rebuilt per candidate.

    No temperature/geometry/LD is cached. LD table interpolation and exact
    visibility selection remain CPU bridges. Reflection uses intrinsic T/LD
    once, not re-interpolated coefficients at each irradiation iteration.
    """

    def __init__(self, law: str, *, device: str = "gpu") -> None:  # noqa: C901, PLR0915
        self._jax, jnp, self.device = _jax_device(device)
        if law not in ("linear", "cosine", "logarithmic", "square_root"):
            msg = "Unsupported device reflection limb-darkening law"
            raise UnsupportedReflectionError(msg)
        self.law = law
        self.ncoeff = 1 if law in ("linear", "cosine") else 2
        barrier = self._jax.lax.optimization_barrier

        def integral(coefficients: Any) -> Any:
            value = barrier(1-barrier(coefficients[0]/3))
            if law == "logarithmic":
                value = barrier(value+barrier(barrier(2*coefficients[1])/9))
            elif law == "square_root":
                value = barrier(value-barrier(coefficients[1]/5))
            return barrier(const.PI*value)

        def directional(mu: Any, coefficients: Any) -> Any:
            a = coefficients[0]
            if law in ("linear", "cosine"):
                value = barrier(barrier(1-a)+barrier(a*mu))
            else:
                value = barrier(1-barrier(a*barrier(1-mu)))
                term = (barrier(barrier(coefficients[1]*mu)*jnp.log(jnp.where(mu > 0, mu, 1)))
                        if law == "logarithmic" else barrier(coefficients[1]*barrier(1-jnp.sqrt(mu))))
                value = barrier(value-term)
            return jnp.where(mu > 0, value, 0)

        def geometry(a: Any, b: Any, na: Any, nb: Any) -> tuple[Any, ...]:
            delta = barrier(b[None]-a[:, None])
            products = barrier(delta*delta)
            distance = jnp.sqrt(barrier(barrier(products[..., 0]+products[..., 1])+products[..., 2]))
            join = barrier(delta/distance[..., None])
            p = barrier(na[:, None]*join)
            s = barrier(-nb[None]*join)
            ga = jnp.maximum(barrier(barrier(p[..., 0]+p[..., 1])+p[..., 2]), 0)
            gb = jnp.maximum(barrier(barrier(s[..., 0]+s[..., 1])+s[..., 2]), 0)
            q = barrier(barrier(ga*gb)/barrier(distance**2))
            return distance, ga, gb, q

        def evaluate(primary: Any, secondary: Any, parameters: Any, iterations: Any) -> dict[str, Any]:
            # c,n,t,A,LD,visible,symmetry,mirror,visible-to-base-reflection
            ca, na, ta, aa, lda, va, sa, ma, ra = primary
            cb, nb, tb, ab, ldb, vb, sb, mb, rb = secondary
            da, gaa, gba, qa = geometry(ca[sa], cb[vb], na[sa], nb[vb])
            db, gab, gbb, qb = geometry(ca[va], cb[sb], na[va], nb[sb])
            # Preserve the original helper's coefficient PREFIX, rather than
            # substituting geometrically selected LD indices in these matrices.
            coupling_a = barrier(qa*directional(gba, ldb[:, :len(vb)]))
            coupling_b = barrier(qb*directional(gab.T, lda[:, :len(va)]).T)
            teff_a, teff_b = barrier(ta**4), barrier(tb**4)
            ia, ib = integral(lda), integral(ldb)
            ratio_a = barrier(barrier(parameters[0]/ia[sa])/teff_a[sa])
            ratio_b = barrier(barrier(parameters[1]/ib[sb])/teff_b[sb])

            def update_a(r_b: Any) -> Any:
                vector = barrier(barrier(r_b[rb]*teff_b[vb])*ab[vb])
                incident = jnp.matmul(coupling_a, vector, precision=self._jax.lax.Precision.HIGHEST)
                return barrier(1+barrier(ratio_a*incident))

            def update_b(r_a: Any) -> Any:
                vector = barrier(barrier(r_a[ra]*teff_a[va])*aa[va])
                incident = jnp.matmul(vector, coupling_b, precision=self._jax.lax.Precision.HIGHEST)
                return barrier(1+barrier(ratio_b*incident))

            def iteration(_: Any, state: Any) -> Any:
                a, b = state

                def primary_first() -> Any:
                    new_a = update_a(b)
                    return new_a, update_b(new_a)

                def secondary_first() -> Any:
                    new_b = update_b(a)
                    return update_a(new_b), new_b

                return self._jax.lax.cond(parameters[2] <= parameters[3], primary_first, secondary_first)

            r_a, r_b = self._jax.lax.fori_loop(
                0, iterations, iteration, (jnp.ones(sa.shape), jnp.ones(sb.shape)),
            )
            t_a = barrier(ta.at[sa].set(barrier(ta[sa]*r_a**0.25))[ma])
            t_b = barrier(tb.at[sb].set(barrier(tb[sb]*r_b**0.25))[mb])
            valid = (jnp.all(jnp.isfinite(t_a) & (t_a > 0)) & jnp.all(jnp.isfinite(t_b) & (t_b > 0))
                     & jnp.all(jnp.isfinite(coupling_a)) & jnp.all(jnp.isfinite(coupling_b))
                     & jnp.all(ia > 0) & jnp.all(ib > 0) & jnp.all(da > 0) & jnp.all(db > 0))
            return {"temperatures": (t_a, t_b), "reflection_factor": (r_a, r_b),
                    "coupling": (coupling_a, coupling_b), "distance": (da, db),
                    "gamma": ((gaa, gba), (gab, gbb)), "proposal_valid": valid}

        self._evaluate = self._jax.jit(evaluate)

    def prepare(  # noqa: C901, PLR0912, PLR0915 - validate original symmetry and bounded device state
        self, system: Any, iterations: int, *, device_states: dict[str, Any] | None = None,
    ) -> PreparedReflection:
        """Snapshot intrinsic corrected state and original CPU bolometric LD."""
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        if (system.morphology != "detached" or system.eccentricity != 0
                or system.has_spots() or system.has_pulsations()):
            msg = "Device reflection requires circular detached unspotted/nonpulsating state; use full CPU"
            raise UnsupportedReflectionError(msg)
        if not isinstance(iterations, (int, np.integer)) or not 0 <= iterations <= _MAX_ITERATIONS:
            msg = "Device reflection supports 0 to 32 iterations; use full CPU"
            raise UnsupportedReflectionError(msg)
        stars = [getattr(system, name) for name in _COMPONENTS]
        xlim = faces.faces_visibility_x_limits(stars[0].polar_radius, stars[1].polar_radius, 1.0)
        visible, symmetry, inputs = [], [], []
        for name, star in zip(_COMPONENTS, stars, strict=True):
            n = len(star.faces)
            if n > _MAX_FACES or not star.symmetry_test():
                msg = "Device reflection exceeds 8192 symmetric faces per component; use full CPU"
                raise UnsupportedReflectionError(msg)
            vis, sym = faces.get_visibility_tests(star.face_centres, xlim[name], name, system.morphology, q_test=True)
            v, s = np.flatnonzero(vis), np.flatnonzero(sym)
            mirror = np.array(star.face_symmetry_vector, dtype=np.int64, copy=True)
            base = int(star.base_symmetry_faces_number)
            lookup = np.full(n, -1, dtype=np.int64)
            lookup[s] = np.arange(len(s))
            if (mirror.shape != (n,) or n != 4*base or not np.array_equal(mirror, np.tile(np.arange(base), 4))
                    or not len(s) or not np.array_equal(s, v[:len(s)]) or np.any(lookup[mirror[v]] < 0)):
                msg = "Unsupported reflection quarter-face indexing; use full CPU"
                raise UnsupportedReflectionError(msg)
            arrays = [np.array(getattr(star, key), dtype=np.float64, copy=True)
                      for key in ("face_centres", "normals", "temperatures", "areas")]
            if (any(not np.isfinite(a).all() for a in arrays) or any(a.shape != (n, 3) for a in arrays[:2])
                    or any(a.shape != (n,) or np.any(a <= 0) for a in arrays[2:])):
                msg = "Unsupported reflection surface arrays; use full CPU"
                raise UnsupportedReflectionError(msg)
            coefficients = ld.get_bolometric_ld_coefficients(
                star.temperatures, star.log_g, star.metallicity, star.limb_darkening_coefficients,
            )
            if coefficients.shape != (self.ncoeff, n) or not np.isfinite(coefficients).all():
                msg = "Unsupported reflection bolometric LD arrays; use full CPU"
                raise UnsupportedReflectionError(msg)
            if device_states is None:
                state = [self._jax.device_put(a, self.device, may_alias=False) for a in arrays]
            else:
                state = [device_states[name][key] for key in ("face_centres", "normals", "temperatures", "areas")]
                if any(a.shape != b.shape or a.dtype != np.dtype("float64") or a.device != self.device
                       for a, b in zip(state, arrays, strict=True)):
                    msg = "Reflection device state shape/dtype/device mismatch"
                    raise RuntimeError(msg)
            state.extend(self._jax.device_put(a, self.device, may_alias=False)
                         for a in (coefficients, v, s, mirror, lookup[mirror[v]]))
            inputs.append(tuple(state))
            v.flags.writeable = s.flags.writeable = False
            visible.append(v)
            symmetry.append(s)
        pairs = len(symmetry[0])*len(visible[1])+len(visible[0])*len(symmetry[1])
        if pairs > _MAX_PAIRS:
            msg = "Device reflection exceeds 2097152 prepared face pairs; use full CPU"
            raise UnsupportedReflectionError(msg)
        parameters = np.array([stars[0].albedo, stars[1].albedo, stars[0].t_eff, stars[1].t_eff])
        if (not np.isfinite(parameters).all() or np.any(parameters[:2] < 0)
                or np.any(parameters[:2] > 1) or np.any(parameters[2:] <= 0)):
            msg = "Unsupported reflection albedo/temperature parameters; use full CPU"
            raise UnsupportedReflectionError(msg)
        return PreparedReflection(self, (*inputs, self._jax.device_put(parameters, self.device, may_alias=False),
                                         self._jax.device_put(np.int32(iterations), self.device)),
                                  tuple(visible), tuple(symmetry))
