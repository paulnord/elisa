"""Intrinsic gravity-darkened temperatures; no irradiation or atmosphere model."""

from __future__ import annotations

from typing import Any

import numpy as np

from elisa import const
from elisa.experimental.gpu_likelihood import _jax_device

_FACE_NDIM = 2
_BETA_MIN, _BETA_MAX = 0.0, 1.0


class GravityDarkeningTemperature:
    """Apply ELISa's unspotted intrinsic temperature law with energy closure.

    T_i = T_eff * [sum(A)/sum(A*g**beta)]**(1/4) * g_i**(beta/4).
    This is algebraically the CPU polar-reference formula: its constant
    reference gravity cancels. Normalize gravity and area by row maxima to
    avoid needless overflow, without changing the prescription. All supplied
    faces need positive finite area and gravity from the same proposal/build
    stage. One unsupported face makes the stellar normalization unsupported.

    Outputs are kelvin and intrinsic emitted bolometric surface flux W/m**2.
    Surface flux is sigma*T**4, not normal radiance or observed light. Reflection,
    spots, atmosphere/LD interpolation and eclipse coverage are not included.
    Unsupported results require explicit failure or CPU fallback, never an
    implicit physical-prior rejection or zero-flux shortcut.
    """

    def __init__(self, *, device: str = "gpu") -> None:
        self._jax, jnp, self.device = _jax_device(device)

        def temperatures(gravity: Any, areas: Any, t_eff: Any, beta: Any) -> dict[str, Any]:
            faces_valid = jnp.isfinite(gravity) & (gravity > 0) & jnp.isfinite(areas) & (areas > 0)
            admitted = jnp.all(faces_valid, axis=1)
            safe_g, safe_a = jnp.where(faces_valid, gravity, 1), jnp.where(faces_valid, areas, 1)
            g = safe_g / jnp.max(safe_g, axis=1)[:, None]
            a = safe_a / jnp.max(safe_a, axis=1)[:, None]
            weight = g ** beta[:, None]
            area_sum = jnp.sum(a, axis=1)
            denominator = jnp.sum(a * weight, axis=1)
            factor = (area_sum / denominator)[:, None] ** 0.25 * g ** (0.25 * beta[:, None])
            t = t_eff[:, None] * factor
            surface_flux = const.STEFAN_BOLTZMAN_CONST * t**4
            luminosity_ratio = jnp.sum(a * factor**4, axis=1) / area_sum
            numerical = (
                jnp.isfinite(denominator)
                & (denominator > 0)
                & jnp.all(jnp.isfinite(t) & (t > 0) & jnp.isfinite(surface_flux) & (surface_flux > 0), axis=1)
                & jnp.isfinite(luminosity_ratio)
            )
            valid = admitted & numerical
            return {
                "temperatures": jnp.where(valid[:, None], t, jnp.nan),
                "surface_flux": jnp.where(valid[:, None], surface_flux, jnp.nan),
                "luminosity_ratio": jnp.where(valid, luminosity_ratio, jnp.nan),
                "valid_faces": faces_valid,
                "proposal_valid": valid,
                "status": jnp.where(~admitted, 1, jnp.where(numerical, 0, 2)),
            }

        self._temperatures = self._jax.jit(temperatures)

    @staticmethod
    def validate_parameters(t_eff: Any, beta: Any, proposals: int) -> tuple[np.ndarray, np.ndarray]:
        """Return private finite, supported host parameter snapshots."""
        t, b = (np.array(x, dtype=np.float64, copy=True) for x in (t_eff, beta))
        if any(x.shape != (proposals,) or not np.isfinite(x).all() for x in (t, b)):
            msg = "Temperature and beta must be finite arrays matching proposals"
            raise ValueError(msg)
        if np.any(t <= 0) or np.any((b < _BETA_MIN) | (b > _BETA_MAX)):
            msg = "Intrinsic temperature must be positive and beta in [0,1]"
            raise ValueError(msg)
        return t, b

    def __call__(self, face_gravity: Any, areas: Any, t_eff: Any, beta: Any) -> dict[str, Any]:
        """Snapshot face arrays and return device temperatures and validity.

        Face arrays are (proposals, faces). Invalid/nonfinite face values yield
        status 1; numeric overflow/underflow yields status 2. Those are explicit
        unsupported calculations, not physical-prior rejection probabilities.
        """
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        g, a = (np.array(x, dtype=np.float64, copy=True) for x in (face_gravity, areas))
        if g.ndim != _FACE_NDIM or not g.shape[0] or not g.shape[1] or a.shape != g.shape:
            msg = "Gravity and areas must have matching nonempty (proposals, faces) shapes"
            raise ValueError(msg)
        t, b = self.validate_parameters(t_eff, beta, g.shape[0])
        return self._temperatures(*(self._jax.device_put(x, self.device, may_alias=False) for x in (g, a, t, b)))
