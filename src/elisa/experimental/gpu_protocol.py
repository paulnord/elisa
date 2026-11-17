"""Opt-in observation-phase prediction and empirical mesh-qualified fitting.

CPU engines/defaults are unchanged. Qualification executes before the original
candidate-penalty boundary. A failed finite-resolution check aborts evaluation;
it never becomes a prior exclusion, large residual or automatic same-grid CPU
fallback. Finite comparisons do not certify continuum accuracy.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from elisa import settings
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.lc_prediction import predict_light_curves
from elisa.analytics.models import lc
from elisa.analytics.params import parameters
from elisa.experimental.gpu_accuracy import resolution_error

_REFERENCE_COUNT, _MIN_REFERENCE_DEGREES = 2, 2.0
_INFERRED_SECONDARY_FLOOR = 3.0


@dataclass(frozen=True)
class MeshResolutionCheck:
    """Two strictly finer CPU references and a joint observational-noise budget.

    Both target-to-finest and middle-to-finest distances must meet the budget.
    This is a fresh empirical check at each candidate and actual observation
    grid, not a continuum bound or domain-wide cached certificate. Bounds cap
    reference work to the exercised >=2-degree original CPU meshes. Finer than
    3 degrees requires explicit paired mesh mode to avoid the inferred-secondary floor.
    """

    refinement_degrees: tuple[float, float] = (8.0, 6.0)
    max_distance_sigma: float = 0.1

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in self.refinement_degrees)
        if (len(values) != _REFERENCE_COUNT or not np.isfinite(values).all()
                or not values[0] > values[1] >= _MIN_REFERENCE_DEGREES
                or not np.isfinite(self.max_distance_sigma) or self.max_distance_sigma <= 0):
            msg = "Supply two decreasing reference degrees >=2 and a positive finite mesh budget"
            raise ValueError(msg)
        object.__setattr__(self, "refinement_degrees", values)


class ResolutionNotQualifiedError(RuntimeError):
    """The configured finite mesh accuracy is unestablished; abort the fit/run."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = deepcopy(report)


class ProtocolPredictionMixin:
    """Cooperative mixin; native defaults preserve original prediction exactly."""

    def __init__(
        self, *, phase_mode: str = "native", resolution_check: MeshResolutionCheck | None = None,
        mesh_mode: str = "native",
    ) -> None:
        if phase_mode not in ("native", "observations"):
            msg = "phase_mode must be 'native' or 'observations'"
            raise ValueError(msg)
        if resolution_check is not None and (not isinstance(resolution_check, MeshResolutionCheck)
                                             or phase_mode != "observations"):
            msg = "MeshResolutionCheck requires phase_mode='observations'"
            raise ValueError(msg)
        if mesh_mode not in ("native", "paired") or (mesh_mode == "paired" and phase_mode != "observations"):
            msg = "mesh_mode must be 'native', or 'paired' with observation phases"
            raise ValueError(msg)
        if (resolution_check is not None and min(resolution_check.refinement_degrees) < _INFERRED_SECONDARY_FLOOR
                and mesh_mode != "paired"):
            msg = "Reference meshes below 3 degrees require explicit paired mesh mode"
            raise ValueError(msg)
        self._phase_mode, self._resolution_check = phase_mode, resolution_check
        self._mesh_mode = mesh_mode
        self._protocol_active, self._protocol_target_error = False, None
        self.last_resolution_check: dict[str, Any] | None = None
        super().__init__()

    @property
    def phase_mode(self) -> str:
        return self._phase_mode

    @property
    def resolution_check(self) -> MeshResolutionCheck | None:
        return self._resolution_check

    @property
    def mesh_mode(self) -> str:
        return self._mesh_mode

    def check_protocol_runtime(self) -> None:
        if self.phase_mode == "observations" and settings.NUMBER_OF_PROCESSES != 1:
            msg = "Experimental observation-phase protocol requires NUMBER_OF_PROCESSES=1"
            raise RuntimeError(msg)
        if self.resolution_check is not None and not self.discretization > self.resolution_check.refinement_degrees[0]:
            msg = "Both qualification reference meshes must be strictly finer than the evaluation mesh"
            raise ValueError(msg)

    def generate_sample_phases(self, samples: Any) -> Any:
        # Explicit observation mode bypasses unused interpolation/adaptive setup.
        # Dynamic JD/phase conversion remains in the original predictor per call.
        return None if self.phase_mode == "observations" else super().generate_sample_phases(samples)

    def _direct_prediction(
        self, model_parameters: dict[str, Any], discretization: float, synthesis: Any, *, observer: Any,
    ) -> dict[str, np.ndarray]:
        if self.mesh_mode == "paired":
            model_parameters = dict(model_parameters, **{"secondary@discretization_factor": discretization})
        return predict_light_curves(
            self.x_data_reduced, self.y_data, self.x_data_reducer, discretization, observer,
            len(self.x_data_reduced)+1, model_parameters, fit_xs=None,
            x_data=self.x_data, y_err=self.y_err, _validated_reducers=self._validated_reducers,
            atmosphere_model=self.atmosphere_model, limb_darkening_coefficients=self.limb_darkening_coefficients,
            synthesis=synthesis,
        )

    def _qualify(self, props: dict[str, Any]) -> None:
        check = self.resolution_check
        if check is None:
            return
        self.check_protocol_runtime()
        started = perf_counter()
        report = {"status": "checking", "degrees": (self.discretization, *check.refinement_degrees),
                  "phase_mode": self.phase_mode, "mesh_mode": self.mesh_mode, "reference_backend": "original_cpu",
                  "noise_reference": "original normalized observational errors, independent of fitted ln_f",
                  "max_distance_sigma": check.max_distance_sigma,
                  "continuum_certified": False}
        self.last_resolution_check = report
        predictions = []
        for index, disc in enumerate(report["degrees"]):
            try:
                predictions.append(self._direct_prediction(props, disc, lc.synthetic_binary,
                                                           observer=deepcopy(self.observer)))
            except Exception as exc:
                if index == 0:
                    # Original target physics failed. The callback wrapper will
                    # relay this exact error to the original penalty handler.
                    report.update(status="target_physics_failed", error_type=type(exc).__name__,
                                  total_seconds=perf_counter()-started)
                    raise
                report.update(status="reference_failed", error_type=type(exc).__name__,
                              total_seconds=perf_counter()-started)
                msg = "A CPU refinement failed; mesh accuracy is unestablished, not a physical-prior rejection"
                raise ResolutionNotQualifiedError(msg, report) from exc
        labels = list(self.y_data)
        sigma = np.concatenate([np.broadcast_to(self.y_err[b], np.shape(predictions[0][b])) for b in labels])
        finest = np.concatenate([predictions[-1][b] for b in labels])
        comparisons = [resolution_error(np.concatenate([p[b] for b in labels]), finest, sigma,
                                        max_distance_sigma=check.max_distance_sigma) for p in predictions[:-1]]
        qualified = all(c["within_diagnostic_budget"] for c in comparisons)
        report.update(status="qualified" if qualified else "budget_failed", comparisons=comparisons,
                      total_seconds=perf_counter()-started)
        if not qualified:
            msg = "Finite mesh changes exceed the configured observational-noise budget; abort evaluation"
            raise ResolutionNotQualifiedError(msg, report)

    @contextmanager
    def _qualified_callback(self, props: dict[str, Any] | None, preparation_error: Exception | None = None) -> Any:
        self._protocol_target_error = preparation_error
        try:
            if preparation_error is None:
                try:
                    self._qualify(props)
                except ResolutionNotQualifiedError:
                    raise  # outside the original catch-all candidate policy
                except Exception as exc:
                    # Only a failed original target is eligible for its original
                    # rejection semantics. Other protocol/infrastructure errors escape.
                    if (self.last_resolution_check is None
                            or self.last_resolution_check["status"] != "target_physics_failed"):
                        raise
                    self._protocol_target_error = exc
            self._protocol_active = True
            yield
        finally:
            self._protocol_active, self._protocol_target_error = False, None

    def predict_light_curves(self, model_parameters: dict[str, Any], **kwargs: Any) -> dict[str, np.ndarray]:
        self.check_protocol_runtime()
        if self._protocol_target_error is not None:
            raise self._protocol_target_error
        if self.resolution_check is not None and not self._protocol_active:
            self._qualify(model_parameters)
        if self.phase_mode == "native":
            return super().predict_light_curves(model_parameters, **kwargs)
        synthesis = kwargs.pop("synthesis", lc.synthetic_binary)
        if kwargs:
            msg = "Observation-phase prediction accepts only an optional synthesis hook"
            raise TypeError(msg)
        return self._direct_prediction(model_parameters, self.discretization, synthesis, observer=self.observer)


class ProtocolLeastSquaresMixin(ProtocolPredictionMixin):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        if self.phase_mode == "observations" and kwargs.get("workers") not in (None, 1):
            msg = "Experimental observation-phase LSQ supports only serial workers"
            raise ValueError(msg)
        return super().fit(*args, **kwargs)

    def model_to_fit(self, xn: Any) -> Any:
        self.check_protocol_runtime()
        if self.resolution_check is None:
            return super().model_to_fit(xn)
        props, error = None, None
        try:
            values = parameters.vector_renormalizer(xn, self.fitable.keys(), self.normalization)
            props = parameters.prepare_properties_set(np.asarray(values), self.fitable.keys(), self.constrained,
                                                      self.fixed, raise_initial=False)
        except Exception as exc:  # noqa: BLE001 - relay unchanged to original callback policy
            error = exc
        with self._qualified_callback(props, error):
            return super().model_to_fit(xn)


class ProtocolMCMCMixin(ProtocolPredictionMixin):
    def ln_probability(self, xn: Any) -> float:
        self.check_protocol_runtime()
        if self.resolution_check is None:
            return super().ln_probability(xn)
        try:
            prior = self.ln_prior(xn, self.norm_init_vector, self.sigmas)
        except Exception:  # noqa: BLE001 - delegate all prior errors to original policy
            return super().ln_probability(xn)
        if not np.isfinite(prior):
            return super().ln_probability(xn)  # original prior/nonfinite-prior handling
        props, error = None, None
        try:
            _, props = self._prepare_likelihood_parameters(xn)
        except Exception as exc:  # noqa: BLE001 - relay unchanged to original callback policy
            error = exc
        with self._qualified_callback(props, error):
            return super().ln_probability(xn)

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        if self.phase_mode == "observations" and settings.NUMBER_OF_MCMC_PROCESSES != 1:
            msg = "Experimental observation-phase MCMC is serial"
            raise RuntimeError(msg)
        return super().fit(*args, **kwargs)


class ProtocolDetachedLeastSquaresFit(ProtocolLeastSquaresMixin, least_squares.DetachedLightCurveFit):
    """Original CPU detached fitter with explicit opt-in phase/accuracy protocol."""


class ProtocolOvercontactLeastSquaresFit(ProtocolLeastSquaresMixin, least_squares.OvercontactLightCurveFit):
    """Original CPU overcontact fitter and constraints with the opt-in protocol."""


class ProtocolDetachedMCMCFit(ProtocolMCMCMixin, mcmc.DetachedLightCurveFit):
    """Original CPU detached posterior with explicit opt-in phase/accuracy protocol."""


class ProtocolOvercontactMCMCFit(ProtocolMCMCMixin, mcmc.OvercontactLightCurveFit):
    """Original CPU overcontact posterior and constraints with the opt-in protocol."""
