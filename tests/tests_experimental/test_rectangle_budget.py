"""Analytic mixed differences and Gaussian bounds beyond independent axes."""

from dataclasses import replace

import numpy as np
import pytest

from elisa import settings
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_curved import CurvedCurveResult
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_rectangle_budget import RectangleInferenceBudget


def make_rectangle() -> tuple:
    base = np.array([[0.8, 0.2], [0.7, 0.2], [0.6, 0.2]])
    a = CurvedCurveResult(np.array([0.0, 0.1, 0.2]), {}, {}, {"b": base}, "fixed", {"depth": 3}, 3, 0.2, {}, {})
    f = replace(a, field_levels=2)
    d = replace(a, rule={"depth": 2})
    c = replace(f, rule={"depth": 2}, component_fluxes={"b": base + np.array([[0.0, 0.0], [-1e-6, 0.0], [-1e-6, 0.0]])})
    return a, f, d, c


@pytest.mark.parametrize("mean", [False, True])
def test_interaction_corner_can_fail_when_individual_axes_are_identical(mean: bool) -> None:
    curves = make_rectangle()
    a, _f, _d, c = curves
    reducers = {"b": np.array([2, 1, 2, 0, 1])}
    observed = {"b": np.array([0.7, 0.8, 0.7, 0.9, 0.8])}
    sigma = 1e-4
    report = RectangleInferenceBudget().compare(
        *curves,
        sigma,
        observed=observed,
        reducers=reducers,
        mean_normalize=mean,
        require_budget=False,
    )
    assert report["linear_axes_control"]["within_finite_budget"]
    assert not report["within_finite_budget"]
    assert report["measured_interaction_distance_sigma"] > 0
    assert report["computed_combination_enclosed"]
    assert not report["continuum_certified"]
    assert not report["uncomputed_interactions_certified"]

    def predict(x: CurvedCurveResult) -> np.ndarray:
        y = HybridBinaryLightCurve._observed_fluxes(x.component_fluxes, x, normalize=True)["b"][reducers["b"]]
        return y / y.mean() if mean else y

    pa, pc = predict(a), predict(c)
    actual = abs(-0.5 * np.sum(((observed["b"] - pc) / sigma) ** 2) + 0.5 * np.sum(((observed["b"] - pa) / sigma) ** 2))
    assert actual <= report["computed_combination_log_likelihood_bound"]
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError):
        RectangleInferenceBudget().compare(*curves, sigma, observed=observed, reducers=reducers, mean_normalize=mean)


def test_zero_rectangle_passes_and_mismatches_raise() -> None:
    a, f, d, _ = make_rectangle()
    c = replace(f, rule=d.rule)
    report = RectangleInferenceBudget().compare(a, f, d, c, 1e-4, observed={"b": np.ones(3)})
    assert report["within_finite_budget"]
    assert report["measured_interaction_distance_sigma"] == 0
    with pytest.raises(ValueError, match="rectangle"):
        RectangleInferenceBudget().compare(a, a, d, c, 1e-4, observed={"b": np.ones(3)})
    with pytest.raises(ValueError, match="matching"):
        RectangleInferenceBudget().compare(a, f, d, replace(c, signature="other"), 1e-4, observed={"b": np.ones(3)})
