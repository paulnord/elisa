"""Experimental FP64 device kernel for prepared ELISa surface fluxes.

Geometry, radiance/atmosphere interpolation, and eclipse coverage are still CPU
responsibilities. This module ports the actual per-face limb darkening and flux
reduction, not a replacement spherical or analytic eclipse model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa.experimental.gpu_likelihood import _jax_device

_FACE_ARRAY_NDIM = 2


@dataclass(frozen=True)
class PreparedSurfaceBatch:
    """Validated device-resident inputs; evaluate repeatedly without host copies.

    Created by SurfaceFluxIntegrator.prepare. Treat all fields as internal;
    returned JAX arrays are immutable. Evaluation still requires FP64 enabled.
    """

    _owner: Any
    _arrays: tuple[Any, ...]

    def evaluate(self) -> Any:
        """Return asynchronous device face fluxes and totals for this snapshot."""
        if not self._owner._jax.config.x64_enabled:  # noqa: SLF001 - paired implementation
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        return self._owner._integrate(*self._arrays)  # noqa: SLF001 - paired implementation

    def block_until_ready(self) -> PreparedSurfaceBatch:
        """Synchronize input transfers without compiling or evaluating the kernel."""
        for array in self._arrays:
            array.block_until_ready()
        return self

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, padded faces)."""
        return self._arrays[0].shape

    @property
    def nbytes(self) -> int:
        """Input storage only, excluding outputs/compiler/allocator workspace."""
        return sum(x.nbytes for x in self._arrays)


class SurfaceFluxIntegrator:
    """Batched surface integration with explicit device selection, no fallback.

    One row is one star at one phase in one band. Padding is represented by zero
    radiance, coverage and cosine. Returns per-face contributions and row totals
    as device arrays. Coverage is ELISa uncovered surface area (already
    scaled by semi-major axis squared), not a visibility fraction. Normal
    radiance is already optical-depth corrected. Padded memory scales as
    O(rows * maximum selected faces); callers should chunk large batches.
    No normalization, distance scaling or third light is added. Finite inputs
    can still overflow; the caller MUST check device outputs for finiteness
    before using them in inference. This API does not synchronize that check.
    """

    def __init__(self, law: str, *, device: str = "gpu") -> None:
        if law not in ("linear", "cosine", "logarithmic", "square_root"):
            msg = "Unsupported limb-darkening law"
            raise ValueError(msg)
        self._law = law
        self._jax, jnp, self.device = _jax_device(device)
        self._ncoeff = 1 if law in ("linear", "cosine") else 2

        def integrate(radiance: Any, cosine: Any, coverage: Any, coefficients: Any) -> Any:
            mu = jnp.maximum(cosine, 0)
            a = coefficients[..., 0]
            correction = 1 - a + a * mu if law in ("linear", "cosine") else 1 - a * (1 - mu)
            if law == "logarithmic":
                safe_mu = jnp.where(mu > 0, mu, 1)
                correction -= coefficients[..., 1] * mu * jnp.log(safe_mu)
            elif law == "square_root":
                correction -= coefficients[..., 1] * (1 - jnp.sqrt(mu))
            correction = jnp.where(cosine > 0, correction, 0)
            faces = radiance * ((cosine * coverage) * correction)
            return faces, jnp.sum(faces, axis=1)

        self._integrate = self._jax.jit(integrate)

    def __call__(self, radiance: Any, cosine: Any, coverage: Any, coefficients: Any) -> Any:
        """Validate, transfer and evaluate a batch (not device-resident timing)."""
        return self.prepare(radiance, cosine, coverage, coefficients).evaluate()

    def prepare(self, radiance: Any, cosine: Any, coverage: Any, coefficients: Any) -> PreparedSurfaceBatch:
        """Validate host inputs and transfer a reusable device snapshot.

        Shapes are (rows, faces) and (rows, faces, coefficients). LD coefficients
        are unrestricted only when prepared radiance/coverage are nonnegative.
        This is stricter than upstream for nonphysical configurations that
        produce negative radiance. Finite negative LD factors are not clipped.
        Cosines must be in [-1,1] (roundoff tolerance 1e-12).
        """
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        arrays = [np.array(x, dtype=np.float64, copy=True) for x in (radiance, cosine, coverage, coefficients)]
        r, mu, cov, coeff = arrays
        if r.ndim != _FACE_ARRAY_NDIM or r.shape[0] == 0 or mu.shape != r.shape or cov.shape != r.shape:
            msg = "Surface inputs must have matching (rows, faces) shapes"
            raise ValueError(msg)
        if coeff.shape != (*r.shape, self._ncoeff):
            msg = "Coefficient shape does not match faces and limb-darkening law"
            raise ValueError(msg)
        if not all(np.isfinite(x).all() for x in arrays):
            msg = "Surface inputs must be finite"
            raise ValueError(msg)
        if np.any(r < 0) or np.any(cov < 0) or np.any(np.abs(mu) > 1 + 1e-12):
            msg = "Radiance/coverage must be nonnegative and cosines in [-1,1]"
            raise ValueError(msg)
        return PreparedSurfaceBatch(self, tuple(self._jax.device_put(x, self.device, may_alias=False) for x in arrays))

    def from_containers(self, band: str, stars: Any, *, pad_to: int | None = None) -> Any:
        """Snapshot selected faces of prepared containers, pad, and integrate.

        Input order is preserved, including empty selections and zero-coverage occulted faces.
        Caller MUST choose the same LD law used to prepare these containers:
        normal radiance already includes that law's optical-depth correction.
        The containers carry no authoritative LD-law provenance tag.
        Containers must not be mutated concurrently while they are copied.
        pad_to optionally retains a fixed row width across changing phase
        selections, avoiding recompilation. It must accommodate every selection.
        """
        rows = []
        for star in stars:
            ix = star.indices
            rows.append(
                tuple(
                    np.array(x[ix], dtype=np.float64, copy=True)
                    for x in (
                        star.normal_radiance[band],
                        star.los_cosines,
                        star.coverage,
                        star.ld_cfs[band],
                    )
                ),
            )
        if not rows:
            msg = "At least one star snapshot is required"
            raise ValueError(msg)
        width = max(len(row[0]) for row in rows)
        if pad_to is not None:
            if not isinstance(pad_to, int) or isinstance(pad_to, bool) or pad_to < width:
                msg = "pad_to must be an integer accommodating every selected face"
                raise ValueError(msg)
            width = pad_to
        shape = (len(rows), width)
        r, mu, cov = (np.zeros(shape) for _ in range(3))
        coeff = np.zeros((*shape, self._ncoeff))
        for i, (rr, mm, cc, ll) in enumerate(rows):
            n = len(rr)
            if rr.shape != (n,) or mm.shape != (n,) or cc.shape != (n,) or ll.shape != (n, self._ncoeff):
                msg = "Malformed prepared surface arrays"
                raise ValueError(msg)
            r[i, :n], mu[i, :n], cov[i, :n], coeff[i, :n] = rr, mm, cc, ll
        return self(r, mu, cov, coeff)
