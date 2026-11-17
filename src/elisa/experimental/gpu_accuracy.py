"""Resolution diagnostics, separate from implementation parity and priors.

Finite mesh/grid comparisons are empirical changes, not certified continuum
error bounds. CPU defaults and synthesis are untouched. The spectral observer
is a CPU diagnostic reference for blackbodies, never a GPU speed measurement.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from elisa import const, ld, settings
from elisa.atm.atm import NaiveInterpolatedAtm
from elisa.binary_system.curves.lc_point import compute_lc_on_pos
from elisa.binary_system.system import BinarySystem
from elisa.experimental.gpu_native import _RoutedBinary

_MIN_ORDER, _MAX_ORDER = 2, 64


def resolution_error(
    prediction: Any,
    reference: Any,
    sigma: Any,
    *,
    observed: Any | None = None,
    ln_f: float = -np.inf,
    max_distance_sigma: float = 0.1,
) -> dict[str, Any]:
    """Noise-weighted finite model change with the ORIGINAL fixed-noise law.

    Supply predictions after the original observation mapping/normalization.
    The default diagnostic budget ||delta/sigma_eff|| <=0.1 corresponds to
    <=0.005 expected log-likelihood loss when reference generates the data.
    It is not a prior, fallback decision, new parity gate or convergence proof.
    Delta chi2/logL include the data residual cross term when observations are
    supplied. Covariance, priors and parameter-dependent noise are not added.
    """
    a, b = np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    error = np.asarray(sigma, dtype=float)
    data = b if observed is None else np.asarray(observed, dtype=float)
    if (
        a.shape != b.shape
        or data.shape != b.shape
        or not b.size
        or error.shape not in ((), b.shape)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(data).all()
        or not np.isfinite(error).all()
        or np.any(error <= 0)
    ):
        msg = "Resolution comparison needs matching finite arrays and positive aligned/scalar errors"
        raise ValueError(msg)
    if np.isnan(ln_f) or np.isposinf(ln_f) or not np.isfinite(max_distance_sigma) or max_distance_sigma <= 0:
        msg = "Supply finite ln_f (or -inf) and a positive finite diagnostic budget"
        raise ValueError(msg)
    # logaddexp avoids overflow and retains the existing sigma²*(1+exp(2ln_f)).
    scale = error * np.exp(0.5 * np.logaddexp(0.0, 2 * ln_f))
    if not np.isfinite(scale).all():
        msg = "Effective observational errors overflowed"
        raise ValueError(msg)
    delta, residual = (a - b) / scale, (data - b) / scale
    distance2 = float(np.sum(delta**2))
    delta_chi2 = float(np.sum(delta**2 - 2 * residual * delta))
    return {
        "elements": b.size,
        "max_abs_flux": float(np.max(np.abs(a - b))),
        "max_abs_sigma": float(np.max(np.abs(delta))),
        "rms_sigma": float(np.sqrt(distance2 / b.size)),
        "distance_sigma": float(np.sqrt(distance2)),
        "distance_squared": distance2,
        "expected_delta_log_likelihood": -0.5 * distance2,
        "delta_chi_squared": delta_chi2,
        "delta_log_likelihood": -0.5 * delta_chi2,
        "max_distance_sigma": max_distance_sigma,
        "within_diagnostic_budget": distance2 <= max_distance_sigma**2,
        "reference_status": "finite-resolution comparison; no continuum certification",
    }


class BlackbodySpectralProbe:
    """Independent Gauss-Legendre quadrature of the actual clipped response.

    Integrates in log(wavelength) over native throughput knots, logarithmic
    subintervals and available Akima zero crossings. Increasing order tests
    quadrature stability, not missing spectral information in atmosphere tables.
    Fixed limits equal the CPU passband limits; bolometric truncation is kept.
    """

    def __init__(self, passbands: Any, *, order: int = 16) -> None:
        if not isinstance(order, int) or not _MIN_ORDER <= order <= _MAX_ORDER:
            msg = "Spectral diagnostic order must be an integer from 2 to 64"
            raise ValueError(msg)
        self.passbands = deepcopy(passbands)
        self.order = order
        nodes, weights = leggauss(order)
        self.quadrature = {}
        self._response_checks = {}
        for name, band in self.passbands.items():
            wave = np.asarray(band.table[settings.PASSBAND_DATAFRAME_WAVE], dtype=float)
            lo, hi = float(band.left_bandwidth), float(band.right_bandwidth)
            if not 0 < lo < hi or not np.isfinite([lo, hi]).all():
                msg = "Spectral diagnostics require finite positive wavelength limits"
                raise ValueError(msg)
            boundaries = [np.log(wave[(wave >= lo) & (wave <= hi)]), np.linspace(np.log(lo), np.log(hi), 33)]
            if hasattr(band.akima, "roots"):
                roots = band.akima.roots(extrapolate=False)
                boundaries.append(np.log(roots[(roots > lo) & (roots < hi)]))
            edges = np.unique(np.concatenate(boundaries))
            half = np.diff(edges) / 2
            logwave = ((edges[:-1] + edges[1:]) / 2)[:, None] + half[:, None] * nodes
            wavelength = np.exp(logwave).ravel()
            response = np.maximum(np.asarray(band.akima(wavelength)), 0)
            if not np.isfinite(response).all():
                msg = "Nonfinite diagnostic passband response"
                raise ValueError(msg)
            # d(lambda_m)=lambda_m*d(log lambda), then hemispheric pi.
            weighted = (half[:, None] * weights).ravel() * wavelength * 1e-10 * response * const.PI
            self.quadrature[name] = (wavelength * 1e-10, weighted)
            self._response_checks[name] = (wavelength.copy(), response.copy(), (lo, hi))

    def surface_flux(self, temperatures: Any) -> dict[str, np.ndarray]:
        """Continuous-temperature hemispheric flux in W/m²; bounded chunks."""
        t = np.asarray(temperatures, dtype=float)
        if t.ndim != 1 or not t.size or not np.isfinite(t).all() or np.any(t <= 0):
            msg = "Spectral diagnostics require a finite positive temperature vector"
            raise ValueError(msg)
        result = {}
        for name, (wave, weights) in self.quadrature.items():
            total = np.zeros(t.shape)
            for first in range(0, len(wave), 32):
                w = wave[first : first + 32]
                x = const.PLANCK_CONST * const.C / (const.BOLTZMAN_CONST * t[:, None] * w)
                planck = (2 * const.PLANCK_CONST * const.C**2 / w**5) * np.exp(-x) / -np.expm1(-x)
                total += planck @ weights[first : first + 32]
            result[name] = total
        return result

    def observe(self, observer: Any, phases: Any, *, native_grid_control: bool = False) -> dict[str, Any]:
        """Fresh original CPU physics/coverage with diagnostic spectral emission.

        This owns its Observer/model and replaces only the existing flux
        callback; no global CPU function is patched. It still incurs original
        radiance preparation before replacement. Within-curve reuse of constant
        unspotted radiance is local; no physical cache survives this call.
        """
        owned = deepcopy(observer)
        binary = owned._system  # noqa: SLF001
        if (
            type(binary) is not BinarySystem
            or binary.eccentricity != 0
            or binary.has_spots()
            or binary.has_pulsations()
            or any(getattr(binary, s).atmosphere != "bb" for s in ("primary", "secondary"))
        ):
            msg = "Spectral observer diagnostic supports original circular unspotted/nonpulsating blackbody binaries"
            raise ValueError(msg)
        if settings.NUMBER_OF_PROCESSES != 1 or tuple(owned.passband) != tuple(self.passbands):
            msg = "Spectral diagnostic requires serial matching passbands"
            raise ValueError(msg)
        for name, (wave, expected, limits) in self._response_checks.items():
            band = owned.passband[name]
            actual = np.maximum(np.asarray(band.akima(wave)), 0)
            if (band.left_bandwidth, band.right_bandwidth) != limits or not np.array_equal(actual, expected):
                msg = "Spectral snapshot response changed; construct a fresh diagnostic probe"
                raise ValueError(msg)
        intensity = {}

        def flux(curves: Any, index: int, labels: Any, system: Any) -> Any:
            for name in ("primary", "secondary"):
                star = getattr(system, name)
                if name not in intensity:
                    surface = (
                        NaiveInterpolatedAtm.black_body_radiance(star.temperatures, passband=self.passbands)
                        if native_grid_control
                        else self.surface_flux(star.temperatures)
                    )
                    intensity[name] = {
                        b: surface[b]
                        / ld.calculate_integrated_limb_darkening_factor(
                            settings.LIMB_DARKENING_LAW,
                            star.ld_cfs[b].T,
                        )
                        for b in labels
                    }
                star.normal_radiance = {b: v.copy() for b, v in intensity[name].items()}
            return compute_lc_on_pos(curves, index, labels, system)

        owned._system = _RoutedBinary(binary, flux)  # noqa: SLF001
        try:
            returned, curves = owned.lc(phases=np.asarray(phases), normalize=True)
        finally:
            owned._system = binary  # noqa: SLF001
        return {
            "phases": returned,
            "fluxes": curves,
            "contract": "CPU diagnostic spectrum replacement; original physics/coverage/Observer semantics",
        }
