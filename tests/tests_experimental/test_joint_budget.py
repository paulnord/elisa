"""Analytic Gaussian controls, observation mapping and nominal accuracy stops."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_curved import CurvedCurveResult
from elisa.experimental.gpu_curved_edges import InferenceBudget
from elisa.experimental.gpu_joint_budget import JointInferenceBudget
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError


def curve(component: np.ndarray, **kwargs) -> CurvedCurveResult:
    return CurvedCurveResult(np.array([0.0, 0.1, 0.2]), {}, {}, {"b": component}, "fixed", {}, 0, 0.0, {}, {}, **kwargs)


def test_individual_axes_can_pass_while_joint_likelihood_fails() -> None:
    base = curve(np.array([[0.8, 0.2], [0.7, 0.2], [0.6, 0.2]]))
    change = np.array([[0.0, 0.0], [-6e-6, 0.0], [-6e-6, 0.0]])
    finer = replace(base, component_fluxes={"b": base.component_fluxes["b"] + change})
    observed = {"b": np.array([1.0, 0.9, 0.8])}
    one = InferenceBudget().compare(base, finer, 1e-4, observed=observed)
    assert one["within_finite_budget"]
    joint = JointInferenceBudget().compare(
        base,
        {"field": finer, "boundary": finer},
        1e-4,
        observed=observed,
        require_budget=False,
    )
    assert joint["distance_sigma_bound"] > 0.1
    assert joint["finite_log_likelihood_absolute_bound"] > 0.01
    actual = 0.5 * np.sum((2 * change.sum(axis=1) / 1e-4) ** 2)
    assert joint["finite_log_likelihood_absolute_bound"] >= actual
    assert joint["quadratic_cross_terms"] > 0
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError):
        JointInferenceBudget().compare(base, {"field": finer, "boundary": finer}, 1e-4, observed=observed)


@pytest.mark.parametrize("mean", [False, True])
def test_common_anchor_observation_reducers_third_light_and_gaussian_bound(mean: bool) -> None:
    component = np.array([[0.8, 0.2], [0.7, 0.2], [0.6, 0.2]])
    anchor = replace(curve(component), additional_light=0.2)
    alternatives = {
        "field": replace(anchor, component_fluxes={"b": component + np.array([[0, 0], [1e-8, -2e-8], [-1e-8, 0]])}),
        "boundary": replace(anchor, component_fluxes={"b": component + np.array([[0, 0], [2e-8, 0], [-1e-8, 1e-8]])}),
    }
    reducer = {"b": np.array([2, 1, 2, 0, 1])}
    data = {"b": np.array([0.91, 0.97, 0.90, 1.09, 0.97])}
    sigma = {"b": np.full(5, 1e-4)}
    report = JointInferenceBudget().compare(
        anchor,
        alternatives,
        sigma,
        observed=data,
        reducers=reducer,
        mean_normalize=mean,
        require_budget=False,
    )

    def predict(c: CurvedCurveResult) -> np.ndarray:
        y = HybridBinaryLightCurve._observed_fluxes(c.component_fluxes, c, normalize=True)["b"][reducer["b"]]
        return y / y.mean() if mean else y

    y = predict(anchor)
    delta = sum(predict(c) - y for c in alternatives.values())
    actual = -0.5 * np.sum(((data["b"] - y - delta) / sigma["b"]) ** 2) + 0.5 * np.sum(
        ((data["b"] - y) / sigma["b"]) ** 2,
    )
    assert abs(actual) <= report["finite_log_likelihood_absolute_bound"]
    assert np.linalg.norm(delta / sigma["b"]) <= report["distance_sigma_bound"]
    assert not report["nuisance_noise_used_for_gate"]
    assert not report["continuum_certified"]
    assert not report["uncomputed_interactions_certified"]
    single = JointInferenceBudget().compare(
        anchor,
        {"field": alternatives["field"]},
        sigma,
        observed=data,
        reducers=reducer,
        mean_normalize=mean,
        require_budget=False,
    )
    pair = InferenceBudget().compare(
        anchor,
        alternatives["field"],
        sigma,
        observed=data,
        reducers=reducer,
        mean_normalize=mean,
        require_budget=False,
    )
    assert_allclose(
        single["finite_log_likelihood_absolute_bound"],
        pair["nominal_report"]["finite_log_likelihood_absolute_bound"],
        rtol=1e-13,
    )


def test_mismatched_snapshot_empty_axis_and_invalid_budget_raise() -> None:
    anchor = curve(np.array([[0.8, 0.2], [0.7, 0.2], [0.6, 0.2]]))
    data = {"b": np.ones(3)}
    for options in [{"max_distance_sigma": 0}, {"max_log_likelihood": np.nan}]:
        with pytest.raises(ValueError, match=r"finite|matching|axes"):
            JointInferenceBudget(**options)
    for axes in [{}, {" ": anchor}, {"field": replace(anchor, signature="different")}]:
        with pytest.raises(ValueError, match=r"finite|matching|axes"):
            JointInferenceBudget().compare(anchor, axes, 1e-4, observed=data)
