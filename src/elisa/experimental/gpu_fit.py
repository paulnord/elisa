"""Opt-in HYBRID physical fitting using the original ELISa fitting callbacks.

Only the synthesis hook changes. Parameter/constraint serialization, support
selection, JD/phase mapping, interpolation, per-band observation-mean
normalization, residuals, noise likelihood, priors and failure policy are the
existing CPU implementations. Physical curves are rebuilt for every proposal;
this is neither device-resident sampling nor a production backend registration.
"""

from __future__ import annotations

from inspect import unwrap
from time import perf_counter
from typing import TYPE_CHECKING, Any

from elisa import settings, units
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.models import lc
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_likelihood import _jax_device

if TYPE_CHECKING:
    import numpy as np


class HybridBinarySynthesis:
    """Adapt fresh hybrid curves to lc.synthetic_binary's exact output contract.

    Uses the original prepare_binary serializer for EVERY call, then Observer
    maximum normalization on the fitter's support. The unchanged prediction
    layer subsequently maps/interpolates to each band's observations and mean
    normalizes. Those two normalization stages are distinct. Unsupported device
    model/backend/resource states always use the complete CPU Observer; they
    are never turned into sampler prior rejections. Missing FP64/device raises.

    Engine reuse caches only compiled shapes/code, never a physical prediction.
    last_evaluation is small execution metadata, not a curve cache. Instances
    and the associated fitter/Observer are single-process, single-caller objects.
    """

    def __init__(
        self, *, device: str = "gpu", coverage: str = "device", radiance: str = "device", general: str = "cpu",
    ) -> None:
        self._jax, _, self.device = _jax_device(device)
        if coverage not in ("cpu", "device"):
            msg = "coverage must be 'cpu' or 'device'"
            raise ValueError(msg)
        self.coverage = coverage
        if radiance not in ("cpu", "device"):
            msg = "radiance must be 'cpu' or 'device'"
            raise ValueError(msg)
        self.radiance = radiance
        if general not in ("cpu", "device_integration"):
            msg = "general must be 'cpu' or 'device_integration'"
            raise ValueError(msg)
        self.general = general
        self._engines: dict[tuple[str, ...], HybridBinaryLightCurve] = {}
        self.calls = 0
        self.hybrid_calls = 0
        self.cpu_fallback_calls = 0
        self.last_evaluation: dict[str, Any] | None = None

    def check_runtime(self) -> None:
        """Reject a disabled FP64 runtime outside physical-candidate penalties."""
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled for hybrid fitting"
            raise RuntimeError(msg)
        if self.general == "device_integration" and settings.NUMBER_OF_PROCESSES != 1:
            msg = "Native device integration requires NUMBER_OF_PROCESSES=1"
            raise RuntimeError(msg)

    def __call__(self, phases: Any, discretization: float, observer: Any, **physical: Any) -> dict[str, np.ndarray]:
        self.check_runtime()
        started = perf_counter()
        self.last_evaluation = None
        self.calls += 1
        binary = lc.prepare_binary(discretization=discretization, **physical)
        constructed = perf_counter()
        observer._system = binary  # noqa: SLF001 - same per-instance contract as lc.synthetic_binary
        labels = tuple(observer.passband)
        if labels not in self._engines:
            self._engines[labels] = HybridBinaryLightCurve(
                list(labels), device=self.device.platform, fallback="cpu", coverage=self.coverage,
                radiance=self.radiance, general=self.general,
            )
        result = self._engines[labels](binary, phases, normalize=True, observer=observer)
        # Retain synthetic_binary's observable per-instance Observer state too.
        observer.phases = result.phases.copy()
        observer.fluxes = {band: curve.copy() for band, curve in result.fluxes.items()}
        observer.flux_unit = units.dimensionless_unscaled
        self.hybrid_calls += result.backend in ("hybrid", "hybrid_native")
        self.cpu_fallback_calls += result.backend == "cpu_fallback"
        self.last_evaluation = {
            "backend": result.backend, "fallback_reason": result.fallback_reason,
            "radiance_mode": self.radiance, "coverage_mode": self.coverage,
            "general_mode": self.general,
            "candidate_construction_seconds": constructed - started,
            "curve_stage_seconds": result.timings, "total_synthesis_seconds": perf_counter() - started,
        }
        return result.fluxes


class _HybridPredictionMixin:
    """Replace only the optional synthesis argument of the existing fitters."""

    def __init__(
        self, *, device: str = "gpu", coverage: str = "device", radiance: str = "device", general: str = "cpu",
    ) -> None:
        super().__init__()
        self.hybrid_synthesis = HybridBinarySynthesis(device=device, coverage=coverage,
                                                      radiance=radiance, general=general)

    def predict_light_curves(self, model_parameters: dict[str, Any], **kwargs: Any) -> dict[str, np.ndarray]:
        synthesis = kwargs.pop("synthesis", None)
        # LSQ supplies its logging-wrapped CPU synthesis explicitly. Recognize
        # that hook, while refusing to silently replace a caller's custom model.
        if synthesis is not None and unwrap(synthesis) is not lc.synthetic_binary:
            msg = "Hybrid fitter owns its synthesis hook; use the CPU fitter for a custom synthesis callable"
            raise ValueError(msg)
        return super().predict_light_curves(model_parameters, synthesis=self.hybrid_synthesis, **kwargs)


class _HybridLeastSquares(_HybridPredictionMixin):
    """Experimental serial LSQ: original residual/optimizer, fresh hybrid physics.

    set_up and fit retain original argument/normalization semantics. The actual
    CPU serializer, constraints and prediction layer are inherited unchanged.
    No analytic/autodiff Jacobian or parallel GPU worker support is claimed.
    """

    def model_to_fit(self, xn: Any) -> np.ndarray:
        self.hybrid_synthesis.check_runtime()
        return super().model_to_fit(xn)

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("workers") not in (None, 1):
            msg = "Experimental hybrid LSQ supports only serial workers"
            raise ValueError(msg)
        return super().fit(*args, **kwargs)


class _HybridMCMC(_HybridPredictionMixin):
    """Experimental serial MCMC with original likelihood/prior/failure semantics.

    Proposals and emcee moves remain on CPU. Each physical callback constructs a
    new BinarySystem and transfers/recomputes affected state. No convergence,
    effective-sample or device-resident MCMC claim follows from short smoke runs.
    Use FixedBinaryLikelihood's cached curve for the l3/ln_f-only pilot instead.
    """

    def ln_probability(self, xn: Any) -> float:
        self.hybrid_synthesis.check_runtime()
        return super().ln_probability(xn)

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        if settings.NUMBER_OF_MCMC_PROCESSES != 1:
            msg = "Experimental hybrid MCMC requires NUMBER_OF_MCMC_PROCESSES=1"
            raise ValueError(msg)
        return super().fit(*args, **kwargs)

    def make_sampler(self, nwalkers: int, **kwargs: Any) -> Any:
        """Create serial emcee with the original ln_probability callback.

        Call set_up first and initialize norm_init_vector/sigmas using the
        existing vector_normalizer/normalized_sigma methods, as fit() does.
        This convenience does not implement burn-in, convergence or storage.
        """
        import emcee  # noqa: PLC0415 - optional sampler use

        ndim = len(self.fitable)
        if not isinstance(nwalkers, int) or nwalkers < 2 * ndim:
            msg = "At least twice as many walkers as free parameters are required"
            raise ValueError(msg)
        if kwargs.get("pool") is not None or kwargs.get("vectorize", False):
            msg = "Experimental physical-model sampler is serial, without a pool or vectorization"
            raise ValueError(msg)
        return emcee.EnsembleSampler(nwalkers, ndim, self.ln_probability, **kwargs)


class HybridDetachedLeastSquaresFit(_HybridLeastSquares, least_squares.DetachedLightCurveFit):
    """Original detached public fitter with opt-in hybrid synthesis only."""


class HybridOvercontactLeastSquaresFit(_HybridLeastSquares, least_squares.OvercontactLightCurveFit):
    """Original overcontact constraints/fitter with opt-in hybrid synthesis only."""


class HybridDetachedMCMCFit(_HybridMCMC, mcmc.DetachedLightCurveFit):
    """Original detached posterior/sampler lifecycle with hybrid synthesis only."""


class HybridOvercontactMCMCFit(_HybridMCMC, mcmc.OvercontactLightCurveFit):
    """Original overcontact posterior/constraints with hybrid synthesis only."""
