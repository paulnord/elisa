"""First device-integration pilot, not a GPU stellar-physics engine.

Only additional light and the existing ln_f noise parameter vary. The entire
stellar model is synthesized once by ELISa on the CPU, at the original single-
band observation phases. No interpolation or
physical parameter changes are permitted. JAX is optional and imported lazily.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import emcee
    import jax
    from numpy.typing import ArrayLike, NDArray

    from elisa.binary_system.system import BinarySystem

_NDIM = 2
_MIN_WALKERS = 4


class FixedBinaryLikelihood:
    """Batched FP64 posterior for a fixed binary and two nuisance parameters.

    Parameter order is ``system@additional_light, nuisance@ln_f``. The prior is
    uniform inside explicit bounds. This is a restricted experimental API,
    deliberately not registered as a general ``LCBinaryAnalyticsTask`` backend.
    All observation values/errors must already be in mean-normalized flux units.
    """

    parameter_names = ("system@additional_light", "nuisance@ln_f")

    def __init__(
        self,
        baseline: ArrayLike,
        observed: ArrayLike,
        errors: ArrayLike,
        *,
        device: str = "gpu",
        bounds: ArrayLike = ((0.0, 0.95), (-10.0, 2.0)),
        support_mean: float | None = None,
    ) -> None:
        baseline, observed, errors, bounds, self.mean_ratio = _validate_inputs(
            baseline,
            observed,
            errors,
            bounds,
            support_mean,
        )
        jax, jnp, self.device = _jax_device(device)
        self._jax = jax
        self.baseline = baseline / baseline.mean()
        self.observed, self.errors, self.bounds = observed, errors, bounds
        for a in (self.baseline, self.observed, self.errors, self.bounds):
            a.setflags(write=False)
        base, obs, logs, limits = [
            jax.device_put(a, self.device) for a in (self.baseline, observed, np.log(errors), bounds)
        ]

        ratio = self.mean_ratio

        def prediction(theta: jax.Array) -> jax.Array:
            l3 = theta[:, 0, None]
            return ((1 - l3) * base[None, :] + l3 * ratio) / ((1 - l3) + l3 * ratio)

        def posterior(theta: jax.Array) -> jax.Array:
            valid = jnp.all(jnp.isfinite(theta), axis=1)
            valid &= jnp.all((theta >= limits[:, 0]) & (theta <= limits[:, 1]), axis=1)
            # Replace invalid proposals before arithmetic; they still return -inf.
            safe = jnp.where(valid[:, None], theta, jnp.mean(limits, axis=1))
            ln_f = safe[:, 1, None]
            predicted = prediction(safe)
            log_scale = jnp.maximum(0.0, ln_f) + 0.5 * jnp.log1p(jnp.exp(-2 * jnp.abs(ln_f)))
            log_sigma = logs[None, :] + log_scale
            # Match ELISa's relative-error inflation, not model-dependent jitter.
            log_residual = jnp.log(jnp.abs(obs[None, :] - predicted))
            half_square = jnp.exp(2 * (log_residual - log_sigma) - jnp.log(2.0))
            result = -jnp.sum(half_square + log_sigma + 0.5 * jnp.log(2 * jnp.pi), axis=1)
            return jnp.where(valid & jnp.isfinite(result), result, -jnp.inf)

        self._compiled = jax.jit(posterior)
        self._prediction = jax.jit(prediction)
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Do not allow host metadata to diverge from compiled device closures."""
        if getattr(self, "_frozen", False):
            msg = "This likelihood is an immutable snapshot; construct a new instance"
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    @classmethod
    def from_binary(
        cls,
        binary: BinarySystem,
        phases: ArrayLike,
        band: str,
        observed: ArrayLike,
        errors: ArrayLike,
        **kwargs: Any,
    ) -> FixedBinaryLikelihood:
        """Snapshot a real ELISa circular synchronous unspotted detached binary.

        The copy prevents l3=0 preparation from changing the caller's system.
        Unique original phases are passed to Observer, then mapped back to
        observations. The support/observation mean ratio preserves duplicates.
        This matches the direct-phase fitter path, without interpolation.
        Arbitrary raw arrays are supported by the constructor only for testing.
        """
        from elisa import settings  # noqa: PLC0415
        from elisa.analytics.tools.utils import time_layer_resolver  # noqa: PLC0415
        from elisa.observer.observer import Observer  # noqa: PLC0415 - isolate optional pilot imports

        phases = np.asarray(phases, dtype=float)
        if phases.ndim != 1 or phases.size == 0 or not np.all(np.isfinite(phases)):
            msg = "phases must be a nonempty finite vector"
            raise ValueError(msg)
        if phases.shape != np.shape(observed):
            msg = "phases must match the observation vector"
            raise ValueError(msg)
        if not isinstance(band, str) or not band:
            msg = "one passband name is required"
            raise ValueError(msg)
        if binary.eccentricity != 0 or binary.morphology != "detached":
            msg = "pilot supports only circular detached binaries"
            raise ValueError(msg)
        for star in (binary.primary, binary.secondary):
            if star.synchronicity != 1 or star.has_spots() or star.has_pulsations():
                msg = "pilot requires synchronous unspotted, nonpulsating stars"
                raise ValueError(msg)
        model = deepcopy(binary)
        model.additional_light = 0.0
        observer = Observer(passband=[band], system=model)
        support, inverse = np.unique(phases, return_inverse=True)
        resolved, _ = time_layer_resolver(support, preserve_phases=settings.PRESERVE_INPUT_PHASES)
        _, curves = observer.lc(phases=resolved, normalize=True)
        return cls(curves[band][inverse], observed, errors, support_mean=curves[band].mean(), **kwargs)

    def device_log_probability(self, theta: ArrayLike) -> jax.Array:
        """Return a device array; call block_until_ready() when benchmarking."""
        if not self._jax.config.x64_enabled:
            msg = "FP64 must remain enabled while evaluating this backend"
            raise RuntimeError(msg)
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != _NDIM or theta.shape[1] != _NDIM or theta.shape[0] == 0:
            msg = "theta must have shape (nonzero batch, 2)"
            raise ValueError(msg)
        return self._compiled(self._jax.device_put(theta, self.device))

    def predict(self, theta: ArrayLike) -> NDArray[np.float64]:
        """Return pointwise predictions for valid two-parameter batches."""
        if not self._jax.config.x64_enabled:
            msg = "FP64 must remain enabled while evaluating this backend"
            raise RuntimeError(msg)
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != _NDIM or theta.shape[1] != _NDIM or not theta.shape[0]:
            msg = "theta must have shape (nonzero batch, 2)"
            raise ValueError(msg)
        if not np.all(np.isfinite(theta)) or np.any(theta < self.bounds[:, 0]) or np.any(theta > self.bounds[:, 1]):
            msg = "prediction parameters must be finite and inside bounds"
            raise ValueError(msg)
        return np.asarray(self._prediction(self._jax.device_put(theta, self.device)))

    def __call__(self, theta: ArrayLike) -> NDArray[np.float64]:
        """Emcee vectorized callback; synchronizes/transfers batch log densities."""
        return np.asarray(self.device_log_probability(theta))

    def numpy_reference(self, theta: ArrayLike) -> NDArray[np.float64]:
        """Independent existing ELISa cost-function oracle for test batches."""
        from elisa.analytics.models.cost_fns import prepare_observations, prepared_likelihood  # noqa: PLC0415

        prepared = prepare_observations({"band": self.observed}, {"band": self.errors})
        results = []
        for row in np.asarray(theta):
            if not np.all(np.isfinite(row)) or np.any(row < self.bounds[:, 0]) or np.any(row > self.bounds[:, 1]):
                results.append(-np.inf)
                continue
            l3, ln_f = row
            curve = ((1 - l3) * self.baseline + l3 * self.mean_ratio) / ((1 - l3) + l3 * self.mean_ratio)
            results.append(prepared_likelihood(prepared, {"band": curve}, ln_f))
        return np.asarray(results)

    def make_sampler(self, nwalkers: int) -> emcee.EnsembleSampler:
        """Use emcee's existing ensemble move, not a new statistical algorithm.

        emcee runs on the CPU; only batched likelihoods are compiled for device.
        No multiprocessing pool is used. This is not device-resident MCMC.
        """
        import emcee  # noqa: PLC0415 - only needed when sampling

        if not isinstance(nwalkers, int) or nwalkers < _MIN_WALKERS:
            msg = "at least four walkers are required for two parameters"
            raise ValueError(msg)
        return emcee.EnsembleSampler(nwalkers, 2, self, vectorize=True)


def _jax_device(device: str) -> tuple[Any, Any, Any]:
    """Load optional JAX and resolve an explicit FP64 device."""
    if device not in ("cpu", "gpu"):
        msg = "device must be 'cpu' or 'gpu'; there is no automatic fallback"
        raise ValueError(msg)
    try:
        import jax  # noqa: PLC0415 - optional dependency
        import jax.numpy as jnp  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        msg = "Install the optional JAX dependency for this experimental backend"
        raise RuntimeError(msg) from exc
    if not jax.config.x64_enabled:
        msg = "FP64 is required: set JAX_ENABLE_X64=True before starting Python"
        raise RuntimeError(msg)
    try:
        devices = jax.devices(device)
    except RuntimeError as exc:
        msg = f"Requested JAX {device} device is unavailable; no CPU fallback"
        raise RuntimeError(msg) from exc
    if not devices:
        msg = f"Requested JAX {device} device is unavailable; no CPU fallback"
        raise RuntimeError(msg)
    return jax, jnp, devices[0]


def _validate_inputs(
    baseline: ArrayLike,
    observed: ArrayLike,
    errors: ArrayLike,
    bounds: ArrayLike,
    support_mean: float | None,
) -> tuple[Any, Any, Any, Any, float]:
    """Copy and validate immutable host-side inputs before device allocation."""
    arrays = [np.array(a, dtype=np.float64, copy=True) for a in (baseline, observed, errors)]
    baseline, observed, errors = arrays
    if baseline.ndim != 1 or baseline.size == 0 or any(a.shape != baseline.shape for a in arrays):
        msg = "baseline, observed and errors must be matching nonempty vectors"
        raise ValueError(msg)
    if any(not np.all(np.isfinite(a)) for a in arrays) or np.any(errors <= 0):
        msg = "data must be finite and errors strictly positive"
        raise ValueError(msg)
    if np.any(baseline < 0) or not np.isfinite(baseline.mean()) or baseline.mean() <= 0:
        msg = "baseline must be nonnegative with a positive mean"
        raise ValueError(msg)
    bounds = np.array(bounds, dtype=float, copy=True)
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        msg = "bounds must contain two finite increasing intervals"
        raise ValueError(msg)
    if not (0 <= bounds[0, 0] < bounds[0, 1] < 1):
        msg = "additional-light bounds must lie in [0, 1)"
        raise ValueError(msg)
    support_mean = baseline.mean() if support_mean is None else float(support_mean)
    if not np.isfinite(support_mean) or support_mean <= 0:
        msg = "support_mean must be finite and positive"
        raise ValueError(msg)
    ratio = support_mean / baseline.mean()
    if not np.isfinite(ratio) or ratio <= 0:
        msg = "support/observation mean ratio must be finite and positive"
        raise ValueError(msg)
    return baseline, observed, errors, bounds, ratio
