"""Bounded device Planck quadrature on ELISa's actual fixed passband grids.

Returns hemispheric emitted passband surface flux, before optical-depth/limb
correction. It does not interpolate atmosphere tables or compute eclipses.
"""

# ruff: noqa: SLF001 - package-internal composition of device kernels
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa import const, umpy
from elisa.atm.atm import get_standard_wavelengths
from elisa.experimental.gpu_likelihood import _jax_device

_FACE_NDIM = 2
_MAX_BANDS, _MAX_WAVES, _MAX_TOTAL_WAVES = 8, 4096, 16384
_MAX_CHUNK = 128
_MIN_WAVES = 3
_ANGSTROM_M = 1e-10
_BASIS_CHUNK = 32


def _simpson_weights(wavelength: np.ndarray) -> np.ndarray:
    """Snapshot the exact installed CPU quadrature as a linear functional.

    Applying the existing CPU Simpson wrapper to blocked basis vectors also
    captures its even-sample/end-interval convention. Setup uses O(N*32)
    temporary memory, without another independently maintained Simpson formula.
    """
    count = len(wavelength)
    weights = np.empty(count, dtype=np.float64)
    for first in range(0, count, _BASIS_CHUNK):
        size = min(_BASIS_CHUNK, count - first)
        basis = np.zeros((size, count))
        basis[np.arange(size), first + np.arange(size)] = 1
        weights[first : first + size] = umpy.simps(basis, wavelength, axis=1)
    return weights


def _sample_passbands(passbands: Mapping[str, Any]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Sample the actual current response functions on the exact CPU grids.

    These samples, rather than band labels/table identity, define a quadrature
    revision. A changed prepared Observer or response callable must invalidate
    any reused spectral engine. No temperature/physical state is part of this
    fixed-response snapshot.
    """
    standard = np.array(get_standard_wavelengths(), dtype=np.float64, copy=True)
    if standard.ndim != 1 or not np.isfinite(standard).all() or np.any(standard <= 0):
        msg = "Standard wavelength grid must be finite and positive"
        raise ValueError(msg)
    samples = []
    for name, band in passbands.items():
        if not isinstance(name, str) or not name:
            msg = "Passband names must be nonempty strings"
            raise ValueError(msg)
        lo, hi = float(band.left_bandwidth), float(band.right_bandwidth)
        if not np.isfinite([lo, hi]).all() or not 0 < lo < hi or band.wave_to_si_mult != _ANGSTROM_M:
            msg = "Supported passbands require positive finite Angstrom boundaries"
            raise ValueError(msg)
        mask = (standard >= lo) & (standard <= hi)
        count = np.count_nonzero(mask)
        wave = np.unique(np.concatenate([np.linspace(lo, hi, count), standard[mask]]))
        if not _MIN_WAVES <= len(wave) <= _MAX_WAVES:
            msg = "Passband quadrature requires 3 to 4096 wavelength samples"
            raise ValueError(msg)
        response = np.broadcast_to(np.asarray(band.akima(wave), dtype=np.float64), wave.shape)
        if not np.isfinite(response).all():
            msg = "Passband response must be finite on its quadrature grid"
            raise ValueError(msg)
        samples.append((wave, response.copy()))
    return tuple(samples)


def _prepare_quadrature(passbands: Mapping[str, Any], chunk_size: int) -> tuple[np.ndarray, ...]:
    """Construct private CPU-compatible spectral samples and integration weights."""
    samples = _sample_passbands(passbands)
    waves = [wave for wave, _ in samples]
    weights = [_simpson_weights(wave) * np.maximum(response, 0) * (const.PI * _ANGSTROM_M)
               for wave, response in samples]
    union = np.unique(np.concatenate(waves))
    if len(union) > _MAX_TOTAL_WAVES:
        msg = "Combined quadrature exceeds 16384 wavelength samples"
        raise ValueError(msg)
    padded = ((len(union) + chunk_size - 1) // chunk_size) * chunk_size
    wave_m = np.pad(union * _ANGSTROM_M, (0, padded - len(union)), constant_values=1e-6)
    band_weights = np.zeros((len(passbands), padded))
    for row, (wave, weight) in enumerate(zip(waves, weights, strict=True)):
        band_weights[row, np.searchsorted(union, wave)] = weight
    return wave_m, band_weights, union


class BlackbodyPassbandFlux:
    """Stream Planck samples in bounded chunks; retain only integrated bands.

    Fixed passbands use precisely the CPU's union of the standard wavelengths
    and a same-count uniform grid between each passband's boundaries. Akima
    throughput is sampled once, including CPU-compatible clipping of negative
    boundary overshoot. Simpson weights and the pi/Angstrom conversion are
    fixed snapshots; temperatures stay continuous and are never rounded.

    Input/output are (proposals, faces) K and (proposals, bands, faces) W/m**2.
    Each row requires every temperature finite/positive and every integrated
    flux finite/positive. Unsupported rows return NaN and explicit status,
    never zero brightness or an implicit prior rejection. Callers must bound
    proposal/face counts: temporary spectrum scales as rows*faces*chunk_size,
    rather than rows*faces*all_wavelengths. No allocator peak limit is implied.
    """

    def __init__(self, passbands: Mapping[str, Any], *, device: str = "gpu", chunk_size: int = 32) -> None:
        if not isinstance(passbands, Mapping) or not 1 <= len(passbands) <= _MAX_BANDS:
            msg = "Supply between 1 and 8 named ELISa passbands"
            raise ValueError(msg)
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or not 1 <= chunk_size <= _MAX_CHUNK:
            msg = "Wavelength chunk_size must be an integer in [1,128]"
            raise ValueError(msg)
        wave_m, band_weights, union = _prepare_quadrature(passbands, chunk_size)
        self.passbands = tuple(passbands)
        self.wavelength_samples = len(union)
        self.chunk_size = chunk_size
        self._jax, jnp, self.device = _jax_device(device)
        arrays = (
            ((2 * const.PLANCK_CONST * const.C**2) / wave_m**5).reshape(-1, chunk_size),
            ((const.PLANCK_CONST * const.C / const.BOLTZMAN_CONST) / wave_m).reshape(-1, chunk_size),
            band_weights.reshape(len(passbands), -1, chunk_size).transpose(1, 0, 2),
        )
        self._quadrature = tuple(self._jax.device_put(x, self.device, may_alias=False) for x in arrays)
        bands = len(passbands)

        def integrate(temperature: Any, prefactor: Any, exponent_factor: Any, band_weights: Any) -> dict[str, Any]:
            faces_valid = jnp.isfinite(temperature) & (temperature > 0)
            safe_t = jnp.where(faces_valid, temperature, 1)

            def add_chunk(total: Any, samples: tuple[Any, ...]) -> tuple[Any, None]:
                prefactor, exponent_factor, weights = samples
                exponent = exponent_factor[None, None, :] / safe_t[..., None]
                # exp(-x)/[-expm1(-x)] avoids overflowing exp(x) in cold UV
                # tails and preserves accuracy in the long-wavelength limit.
                inverse = jnp.exp(-exponent) / -jnp.expm1(-exponent)
                spectrum = prefactor[None, None, :] * inverse
                return total + jnp.einsum("rfw,bw->rbf", spectrum, weights), None

            flux, _ = self._jax.lax.scan(
                add_chunk,
                jnp.zeros((temperature.shape[0], bands, temperature.shape[1]), dtype=temperature.dtype),
                (prefactor, exponent_factor, band_weights),
            )
            admitted = jnp.all(faces_valid, axis=1)
            numerical = jnp.all(jnp.isfinite(flux) & (flux > 0), axis=(1, 2))
            valid = admitted & numerical
            return {
                "surface_flux": jnp.where(valid[:, None, None], flux, jnp.nan),
                "valid_faces": faces_valid,
                "proposal_valid": valid,
                "status": jnp.where(~admitted, 1, jnp.where(numerical, 0, 2)),
            }

        self._integrate = self._jax.jit(integrate)

    @property
    def quadrature_nbytes(self) -> int:
        """Fixed device samples/weights, excluding allocator/compiler workspace."""
        return sum(x.nbytes for x in self._quadrature)

    def block_until_ready(self) -> BlackbodyPassbandFlux:
        """Synchronize one-time fixed quadrature uploads."""
        for x in self._quadrature:
            x.block_until_ready()
        return self

    def __call__(self, temperatures: Any) -> dict[str, Any]:
        """Validate shape, snapshot/upload temperatures and return device flux."""
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        t = np.array(temperatures, dtype=np.float64, copy=True)
        if t.ndim != _FACE_NDIM or not t.shape[0] or not t.shape[1]:
            msg = "Temperatures require a nonempty (proposals,faces) array"
            raise ValueError(msg)
        return self._integrate(self._jax.device_put(t, self.device, may_alias=False), *self._quadrature)


@dataclass(frozen=True)
class PreparedBlackbodyRocheBatch:
    """Thermal proposal snapshot sharing fixed topology and passband quadrature."""

    _owner: Any
    _thermal_batch: Any

    def evaluate(self) -> dict[str, Any]:
        """Evaluate changing geometry, gravity, temperature and band flux on device."""
        owner = self._owner
        owner._geometry._check_precision()
        return owner._evaluate(
            *self._thermal_batch._parameters,
            *owner._geometry._topology,
            *owner._radiance._quadrature,
        )

    def block_until_ready(self) -> PreparedBlackbodyRocheBatch:
        """Synchronize inputs without compiling/evaluating the physics."""
        self._thermal_batch.block_until_ready()
        self._owner._radiance.block_until_ready()
        return self

    @property
    def nbytes(self) -> int:
        """Per-proposal parameter bytes; shared topology/quadrature excluded."""
        return self._thermal_batch.nbytes


class RocheBlackbodyPipeline:
    """Extend a RocheGeometryPipeline with fresh intrinsic passband emission.

    Supplied geometry restrictions/build stage remain unchanged. Passband
    emission is hemispheric surface flux, before limb/optical-depth correction,
    occultation and observer-distance scaling. This does not replace complete
    changing ELISa light curves or table-based atmosphere interpolation.
    """

    def __init__(self, geometry: Any, passbands: Mapping[str, Any], *, chunk_size: int = 32) -> None:
        self._geometry = geometry
        self.device = geometry.device
        self._radiance = BlackbodyPassbandFlux(passbands, device=geometry.device.platform, chunk_size=chunk_size)
        self.passbands = self._radiance.passbands
        self._jax = geometry._jax

        def evaluate(*args: Any) -> dict[str, Any]:
            # Nine proposal parameters, three topology arrays, three quadrature
            # arrays. Retain existing thermal FP64 boundaries and device masks.
            base = self._jax.lax.optimization_barrier(geometry._evaluate_thermal(*args[:-3]))
            emitted = self._radiance._integrate(base["thermal"]["temperatures"], *args[-3:])
            return dict(base, blackbody=emitted, proposal_valid=base["proposal_valid"] & emitted["proposal_valid"])

        self._evaluate = self._jax.jit(evaluate)

    @property
    def topology_nbytes(self) -> int:
        """Shared angular directions/connectivity bytes; quadrature separate."""
        return self._geometry.topology_nbytes

    @property
    def quadrature_nbytes(self) -> int:
        """One-time fixed spectral samples/weights in device memory."""
        return self._radiance.quadrature_nbytes

    def block_until_ready(self) -> RocheBlackbodyPipeline:
        """Synchronize all fixed input uploads without compiling physics."""
        self._geometry.block_until_ready()
        self._radiance.block_until_ready()
        return self

    def prepare(self, *parameters: Any) -> PreparedBlackbodyRocheBatch:
        """Take the same nine physical arrays as geometry.prepare_thermal."""
        return PreparedBlackbodyRocheBatch(self, self._geometry.prepare_thermal(*parameters))
