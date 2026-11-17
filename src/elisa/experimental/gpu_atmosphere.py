"""Fresh Castelli-Kurucz integration/interpolation in complete hybrid curves.

CPU ELISa selects nearest gravity/metallicity, brackets temperature, reads and
revision-checks sources, aligns/strips wavelengths and applies actual responses.
Device work integrates unique prepared spectra with the exact CPU Simpson
functional, gathers each face's brackets, interpolates in T**4, mirrors faces
and divides by the same-band optical-depth LD integral. No fitted surface flux
or LD is cached across proposals. This is deliberately a hybrid table path.
"""

# ruff: noqa: SLF001 - composition with original CPU preparation and device kernels
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import ConvexHull

from elisa import ld
from elisa.atm import atm
from elisa.base.curves.utils import generate_teff_logg_for_ld_cfs
from elisa.experimental.gpu_blackbody import _simpson_weights
from elisa.experimental.gpu_likelihood import _jax_device
from elisa.experimental.gpu_radiance import UnsupportedRadianceError, _normal_intensity

if TYPE_CHECKING:
    from collections.abc import Mapping

_COMPONENTS = ("primary", "secondary")
_MAX_FACES, _MAX_BANDS, _MAX_MODELS, _MAX_WAVES = 8192, 8, 128, 4096
_MAX_SAMPLES = 8 * 1024**2  # 64 MiB of source spectra, excluding allocator/compiler workspace
_MIN_WAVES = 2


@lru_cache(maxsize=8)
def _ld_hull(source: Any) -> np.ndarray:
    return ConvexHull(ld.get_ld_source_interpolator(source).points).equations.copy()


def check_ld_boundaries(system: Any, passbands: Any) -> None:
    """Retain exact CPU domain decisions near a table's convex-hull boundary."""
    for name in _COMPONENTS:
        star = getattr(system, name)
        if star.limb_darkening_coefficients is not None:
            continue
        t, g = generate_teff_logg_for_ld_cfs(star, symmetry_test=True)
        query = np.column_stack((t, g + 2))
        for source in ld.get_ld_source_identities(star.metallicity, passbands):
            hull = _ld_hull(source)
            distance = query @ hull[:, :2].T + hull[:, 2]
            margin = 4 * 3e-13 * (np.abs(query) @ np.abs(hull[:, :2]).T + np.abs(hull[:, 2]) + 1)
            if np.any(np.abs(distance) <= margin):
                msg = "Near an LD convex-hull boundary; use complete CPU domain semantics"
                raise UnsupportedRadianceError(msg)


def check_table_transitions(system: Any) -> None:
    """Use full CPU at numerically ambiguous atlas selection boundaries.

    GPU/CPU state roundoff must not flip nearest-gravity choices or select a
    different temperature file at a grid knot. Four times the existing 3e-13
    state relative gate is a conservative dispatch margin, not a loosened
    numerical acceptance gate or a physical-prior restriction.
    """
    grid_t = np.asarray(atm.atm_file_prefix_to_quantity_list("temperature", "ck04"))
    grid_g = np.asarray(atm.atm_file_prefix_to_quantity_list("gravity", "ck04"))
    middle_g = (grid_g[:-1] + grid_g[1:]) / 2
    for name in _COMPONENTS:
        star = getattr(system, name)
        for values, boundaries in [(star.temperatures, grid_t), (star.log_g + 2, middle_g)]:
            distances = np.abs(np.asarray(values)[:, None] - boundaries)
            if np.any(distances <= 4 * 3e-13 * np.maximum(1, np.abs(boundaries))):
                msg = "Near an atlas temperature/gravity selection boundary; use complete CPU semantics"
                raise UnsupportedRadianceError(msg)


@lru_cache(maxsize=8)
def _weights(wavelength_bytes: bytes) -> np.ndarray:
    """Bounded reuse of fixed quadrature only; exact wavelength revision key."""
    values = _simpson_weights(np.frombuffer(wavelength_bytes, dtype=np.float64))
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class PreparedAtmosphereBatch:
    """Immutable candidate spectra/temperature/LD; evaluation has no transfers."""

    _owner: Any
    _arrays: tuple[Any, ...]
    metadata: dict[str, Any]

    def evaluate(self) -> dict[str, Any]:
        if not self._owner._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        return self._owner._evaluate(*self._arrays)

    def block_until_ready(self) -> PreparedAtmosphereBatch:
        self._owner._jax.block_until_ready(self._arrays)
        return self


class DeviceAtmosphereRadiance:
    """Integrate original CPU-prepared CK04 spectra and fresh T**4 interpolation.

    Source and LD interpolation errors retain their original ELISa classes and
    fitter policy. Resource/numeric masks require whole CPU fallback, never a
    prior exclusion. Both stars must use CK04 here; blackbody and mixed-atlas
    dispatch are separate explicit paths. Optional corrected device temperatures
    match the host values used to select files. Corrected/device-reflected arrays
    remain device inputs; optional CPU post-reflection arrays upload.
    """

    def __init__(self, passbands: tuple[str, ...], law: str, *, device: str = "gpu") -> None:
        if not 1 <= len(passbands) <= _MAX_BANDS:
            msg = "Device atmosphere supports 1 to 8 passbands; use full CPU"
            raise UnsupportedRadianceError(msg)
        if law not in ("linear", "cosine", "logarithmic", "square_root"):
            msg = "Unsupported device optical-depth limb-darkening law"
            raise UnsupportedRadianceError(msg)
        self._jax, jnp, self.device = _jax_device(device)
        self.passbands, self.ncoeff = tuple(passbands), 1 if law in ("linear", "cosine") else 2
        barrier = self._jax.lax.optimization_barrier

        def evaluate(sources: Any, coefficients: Any, active: Any) -> dict[str, Any]:
            flux = jnp.zeros(coefficients.shape[:3], dtype=coefficients.dtype)
            valid = jnp.ones(2, dtype=bool)
            for component, component_sources in enumerate(sources):
                for band, arrays in enumerate(component_sources):
                    spectra, quadrature, bracket, grid_t, face_t, mirror, scale = arrays
                    # CPU quadrature is linear after alignment/response, and
                    # CPU atlas_radiance also integrates UNIQUE spectra first.
                    unique = barrier(jnp.sum(barrier(spectra * quadrature), axis=1))
                    unique = barrier(scale * unique)
                    t4 = barrier(jnp.power(grid_t, 4))
                    bottom, top = t4[bracket[0]], t4[bracket[1]]
                    weight = barrier(barrier(jnp.power(face_t, 4)) - bottom) / barrier(top - bottom)
                    weight = jnp.where(jnp.isnan(weight), 1.0, weight)
                    lower, upper = unique[bracket[0]], unique[bracket[1]]
                    emitted = lower + barrier(weight * barrier(upper - lower))
                    full = emitted[mirror]
                    flux = flux.at[component, band, :full.shape[0]].set(full)
                    admitted = jnp.all(jnp.isfinite(face_t) & (face_t > 0))
                    admitted &= jnp.all(jnp.isfinite(emitted) & (emitted > 0))
                    valid = valid.at[component].set(valid[component] & admitted)
            return _normal_intensity(self._jax, jnp, law, flux, coefficients, active,
                                     valid, jnp.where(valid, 0, 1))

        self._evaluate = self._jax.jit(evaluate)

    def prepare(  # noqa: C901, PLR0915 - explicit original source preparation and shape/resource gates
        self, system: Any, ld_cfs: Mapping[str, Any], *, passband: Mapping[str, Any],
        left_bandwidth: float, right_bandwidth: float, device_temperatures: Mapping[str, Any] | None = None,
    ) -> PreparedAtmosphereBatch:
        """Re-run the original CPU source boundary for every physical candidate."""
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        started = perf_counter()
        stars = [getattr(system, name) for name in _COMPONENTS]
        width = max(len(s.faces) for s in stars)
        if width > _MAX_FACES or tuple(passband) != self.passbands:
            msg = "Unsupported device atmosphere face capacity or passband order"
            raise UnsupportedRadianceError(msg)
        coefficients = np.zeros((2, len(passband), width, self.ncoeff))
        active = np.zeros((2, width), dtype=bool)
        sources, shapes, samples = [], [], 0
        for comp, (name, star) in enumerate(zip(_COMPONENTS, stars, strict=True)):
            if not star.symmetry_test() or star.atmosphere != "ck04":
                msg = "Device atlas emission currently requires two symmetric CK04 components"
                raise UnsupportedRadianceError(msg)
            t = np.array(star.symmetry_faces(star.temperatures), dtype=np.float64, copy=True)
            log_g = star.symmetry_faces(star.log_g)
            # Preserve CPU gravity/metallicity nearest-grid choice, temperature
            # bounds, revisions/errors, wavelength union and response semantics.
            prepared, mapping, flux_mult, wave_mult = atm.NaiveInterpolatedAtm._prepare_atlas_spectra(
                t, log_g, star.metallicity, star.atmosphere, passband=passband,
                left_bandwidth=left_bandwidth, right_bandwidth=right_bandwidth,
            )
            if device_temperatures is None:
                face_t = self._jax.device_put(t, self.device, may_alias=False)
            else:
                all_t = device_temperatures[name]
                if all_t.shape != (len(star.faces),) or all_t.device != self.device or all_t.dtype != np.float64:
                    msg = "Expected same-device FP64 complete corrected temperatures"
                    raise ValueError(msg)
                face_t = all_t[:len(t)]
            mirror = np.array(star.face_symmetry_vector, dtype=np.int64, copy=True)
            if mirror.shape != (len(star.faces),) or np.any(mirror < 0) or np.any(mirror >= len(t)):
                msg = "Unsupported CPU symmetry for device atlas emission"
                raise UnsupportedRadianceError(msg)
            active[comp, :len(star.faces)] = True
            component_sources = []
            for band_index, band in enumerate(passband):
                models = prepared[band]
                wavelength = np.asarray(atm.find_atm_defined_wavelength(models), dtype=np.float64)
                matrix = np.array([a.model.flux for a in models], dtype=np.float64)
                samples += matrix.size
                if (not 1 <= len(models) <= _MAX_MODELS or not _MIN_WAVES <= len(wavelength) <= _MAX_WAVES
                        or samples > _MAX_SAMPLES):
                    msg = "Prepared atlas spectra exceed device model/wavelength/storage capacity; use full CPU"
                    raise UnsupportedRadianceError(msg)
                index = np.empty(2 * len(t), dtype=np.int64)
                for i, model in enumerate(models):
                    index[mapping[model.fpath]] = i
                values = np.array(ld_cfs[name][band], dtype=np.float64, copy=True)
                if values.shape != (len(star.faces), self.ncoeff) or not np.isfinite(values).all():
                    msg = "Unsupported fresh table-LD shape/values; use full CPU"
                    raise UnsupportedRadianceError(msg)
                coefficients[comp, band_index, :len(star.faces)] = values
                host = (matrix, _weights(wavelength.tobytes()), index.reshape((2, len(t))),
                        np.array([m.temperature for m in models]), mirror, np.array(flux_mult * wave_mult))
                arrays = tuple(self._jax.device_put(x, self.device, may_alias=False) for x in host)
                component_sources.append((*arrays[:4], face_t, *arrays[4:]))
                shapes.append({"component": name, "band": band, "unique_spectra": len(models),
                               "wavelength_samples": len(wavelength), "base_faces": len(t)})
            sources.append(tuple(component_sources))
        arrays = (tuple(sources), self._jax.device_put(coefficients, self.device, may_alias=False),
                  self._jax.device_put(active, self.device, may_alias=False))
        return PreparedAtmosphereBatch(self, arrays, {"source_shapes": shapes, "spectral_samples": samples,
                                                      "cpu_prepare_upload_seconds": perf_counter() - started})
