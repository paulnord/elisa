"""Fresh blackbody surface flux and optical-depth normal intensity on device.

Uses the original CPU spectral grid/response/Simpson convention. The same
passband's integrated limb-darkening law divides hemispheric surface flux;
projected coverage and directional limb darkening are applied downstream once.
Reflection remains CPU physics and its updated temperatures must be uploaded.
"""

# ruff: noqa: SLF001 - composition of experimental device kernels
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import const
from elisa.experimental.gpu_blackbody import BlackbodyPassbandFlux, _sample_passbands

if TYPE_CHECKING:
    from collections.abc import Mapping

_COMPONENTS = ("primary", "secondary")
_MAX_FACES, _MAX_BANDS = 8192, 8


class UnsupportedRadianceError(ValueError):
    """A bounded spectral evaluation needs full CPU fallback, never a prior cut."""


def passband_revision(passbands: Mapping[str, Any]) -> str:
    """Fingerprint actual current spectral samples, including custom responses.

    Re-sample on every fresh curve. This intentionally includes that CPU cost
    while permitting reuse of immutable quadrature/code, never brightness.
    """
    if not 1 <= len(passbands) <= _MAX_BANDS:
        msg = "Device radiance supports 1 to 8 passbands; use full CPU"
        raise UnsupportedRadianceError(msg)
    try:
        samples = _sample_passbands(passbands)
    except ValueError as exc:
        raise UnsupportedRadianceError(str(exc)) from exc
    digest = sha256()
    for name, (wave, response) in zip(passbands, samples, strict=True):
        label = name.encode()
        digest.update(len(label).to_bytes(8, "little"))
        digest.update(label)
        digest.update(len(wave).to_bytes(8, "little"))
        digest.update(wave.tobytes())
        digest.update(response.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedRadianceBatch:
    """Immutable fresh temperatures, LD and symmetry map; no evaluation transfers."""

    _owner: Any
    _arrays: tuple[Any, ...]

    def evaluate(self) -> dict[str, Any]:
        if not self._owner._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        return self._owner._evaluate(*self._arrays, *self._owner._spectrum._quadrature)

    def block_until_ready(self) -> PreparedRadianceBatch:
        self._owner._spectrum.block_until_ready()
        for array in self._arrays:
            array.block_until_ready()
        return self


class DeviceBlackbodyRadiance:
    """Compose fresh Planck flux, CPU symmetry and optical-depth normalization.

    Fixed passband quadrature is a snapshot. prepare() uploads fresh per-face
    LD and, after CPU reflection, temperatures. Without reflection, corrected
    temperature arrays may remain device inputs. Emission uses the CPU's base
    symmetry patch and mirrors it to all faces before dividing by each face's
    actual same-band LD integral. Unequal component meshes are zero padded;
    padding never participates in temperature or brightness normalization.
    """

    def __init__(self, passbands: Mapping[str, Any], law: str, *, device: str = "gpu") -> None:
        if law not in ("linear", "cosine", "logarithmic", "square_root"):
            msg = "Unsupported device optical-depth limb-darkening law"
            raise UnsupportedRadianceError(msg)
        try:
            self._spectrum = BlackbodyPassbandFlux(passbands, device=device)
        except ValueError as exc:
            raise UnsupportedRadianceError(str(exc)) from exc
        self._jax, self.device = self._spectrum._jax, self._spectrum.device
        self.passbands, self.law = tuple(passbands), law
        self.ncoeff = 1 if law in ("linear", "cosine") else 2
        import jax.numpy as jnp  # noqa: PLC0415 - optional dependency

        barrier = self._jax.lax.optimization_barrier

        def evaluate(temperatures: Any, coefficients: Any, mirror: Any, active: Any, *quadrature: Any) -> dict:
            emitted = barrier(self._spectrum._integrate(temperatures, *quadrature))
            flux = jnp.take_along_axis(emitted["surface_flux"], mirror[:, None, :], axis=2)
            # Keep CPU operation order/rounding at each division/addition.
            integral = barrier(1 - barrier(coefficients[..., 0] / 3))
            if law == "logarithmic":
                integral = barrier(integral + barrier(barrier(2 * coefficients[..., 1]) / 9))
            elif law == "square_root":
                integral = barrier(integral - barrier(coefficients[..., 1] / 5))
            integral = barrier(const.PI * integral)
            intensity = flux / integral
            numerical = jnp.all(~active[:, None, :] | (
                jnp.isfinite(integral) & (integral > 0) & jnp.isfinite(intensity) & (intensity > 0)
            ), axis=(1, 2))
            valid = emitted["proposal_valid"] & numerical
            return {
                "normal_radiance": jnp.where(active[:, None, :],
                                             jnp.where(valid[:, None, None], intensity, jnp.nan), 0),
                "surface_flux": jnp.where(active[:, None, :], flux, 0),
                "ld_integral": integral, "coefficients": coefficients,
                "proposal_valid": valid,
                "status": jnp.where(~emitted["proposal_valid"], emitted["status"], jnp.where(numerical, 0, 3)),
            }

        self._evaluate = self._jax.jit(evaluate)

    def prepare(  # noqa: C901 - explicit CPU/device temperature and symmetry validation
        self, system: Any, ld_cfs: Mapping[str, Any], *, device_temperatures: Mapping[str, Any] | None = None,
    ) -> PreparedRadianceBatch:
        """Snapshot this candidate's CPU-compatible base-patch temperatures/LD."""
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        stars = [getattr(system, name) for name in _COMPONENTS]
        if any(not star.symmetry_test() or star.atmosphere != "bb" for star in stars):
            msg = "Device radiance requires symmetric unspotted/nonpulsating blackbody stars"
            raise UnsupportedRadianceError(msg)
        width = max(len(star.faces) for star in stars)
        bases = [int(star.base_symmetry_faces_number) for star in stars]
        if width > _MAX_FACES or min(bases) <= 0:
            msg = "Device radiance mesh exceeds bounded symmetric-face capacity; use full CPU"
            raise UnsupportedRadianceError(msg)
        base_width = max(bases)
        mirror, active = np.zeros((2, width), dtype=np.int64), np.zeros((2, width), dtype=bool)
        coefficients = np.zeros((2, len(self.passbands), width, self.ncoeff))
        temperatures = []
        import jax.numpy as jnp  # noqa: PLC0415 - device-only padding

        for comp, (name, star, base) in enumerate(zip(_COMPONENTS, stars, bases, strict=True)):
            count = len(star.faces)
            mapping = np.asarray(star.face_symmetry_vector)
            if mapping.shape != (count,) or np.any(mapping < 0) or np.any(mapping >= base):
                msg = "Unsupported CPU face symmetry mapping for device radiance"
                raise UnsupportedRadianceError(msg)
            mirror[comp, :count], active[comp, :count] = mapping, True
            for band_index, band in enumerate(self.passbands):
                values = np.array(ld_cfs[name][band], dtype=np.float64, copy=True)
                if values.shape != (count, self.ncoeff) or not np.isfinite(values).all():
                    msg = "Fresh LD coefficient shape/values are unsupported; use full CPU"
                    raise UnsupportedRadianceError(msg)
                coefficients[comp, band_index, :count] = values
            if device_temperatures is None:
                t = np.array(star.symmetry_faces(star.temperatures), dtype=np.float64, copy=True)
                if t.shape != (base,):
                    msg = "Unsupported CPU base-patch temperature shape"
                    raise UnsupportedRadianceError(msg)
                t = self._jax.device_put(t, self.device, may_alias=False)
            else:
                t = device_temperatures[name]
                if t.shape != (count,) or t.dtype != np.float64 or t.device != self.device:
                    msg = "Device temperatures must be same-device FP64 complete corrected face arrays"
                    raise ValueError(msg)
                t = t[:base]
            temperatures.append(jnp.concatenate((t, jnp.broadcast_to(t[0], (base_width - base,)))))
        arrays = (jnp.stack(temperatures), *(self._jax.device_put(x, self.device, may_alias=False)
                                            for x in (coefficients, mirror, active)))
        return PreparedRadianceBatch(self, arrays)
