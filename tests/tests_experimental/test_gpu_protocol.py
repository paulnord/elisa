"""Opt-in cadence semantics and accuracy errors outside candidate penalties."""

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa.experimental.gpu_protocol import (
    MeshResolutionCheck,
    ProtocolDetachedMCMCFit,
    ResolutionNotQualifiedError,
)
from scripts.validate_gpu_fit import make_problem, normalized_vector, physical, prepare_fit
from scripts.validate_gpu_protocol import protocol_problem

from .test_gpu_binary import require_device
from .test_gpu_likelihood import fp64  # noqa: F401


@pytest.fixture(autouse=True)
def protocol_settings() -> Iterator[None]:
    with settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
                           LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1,
                           NUMBER_OF_MCMC_PROCESSES=1, SUPPRESS_WARNINGS=True):
        yield


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(("jd", "preserve", "sampling"), [(False, False, "uniform"),
                                                          (True, False, "manual_equal_length"),
                                                          (False, True, "uniform")])
def test_direct_observation_mapping_and_changing_support(
    device: str, jd: bool, preserve: bool, sampling: str,
) -> None:
    require_device(device)
    with settings.override(PRESERVE_INPUT_PHASES=preserve):
        problem = make_problem(jd=jd, sampling=sampling, free_nuisance=True, shape_parameters=True)
        original = prepare_fit(problem)
        cpu = prepare_fit(problem, phase_mode="observations")
        hybrid = prepare_fit(problem, phase_mode="observations", device=device)
        assert cpu.fit_xs is None
        assert hybrid.fit_xs is None
        initial_observer = deepcopy(cpu.observer)
        for vector in [normalized_vector(cpu), np.linspace(.25, .7, len(cpu.fitable))]:
            props = physical(cpu, vector)
            native = original.predict_light_curves(props)
            # Explicit original direct protocol is the oracle, not interpolation.
            original.fit_xs, original.interp_treshold = None, len(original.x_data_reduced)+1
            expected = original.predict_light_curves(props)
            for fit in [cpu, hybrid]:
                actual = fit.predict_light_curves(props)
                for band in expected:
                    assert_allclose(actual[band], expected[band], rtol=3e-13, atol=2e-12)
                    assert_allclose(actual[band].mean(), 1, rtol=0, atol=3e-15)
            assert_allclose(hybrid.ln_probability(vector), original.ln_probability(vector), rtol=0, atol=1e-6)
            if np.array_equal(vector, normalized_vector(cpu)):
                assert any(np.max(np.abs(native[b]-expected[b])) > 1e-6 for b in expected)
        assert initial_observer.passband.keys() == cpu.observer.passband.keys()
        assert hybrid.hybrid_synthesis.last_evaluation["backend"] == "hybrid"


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_fresh_qualified_predictions_keep_noise_law_and_observer(device: str) -> None:
    require_device(device)
    problem = protocol_problem()
    problem.config["nuisance"]["ln_f"]["fixed"] = False
    original = prepare_fit(problem, phase_mode="observations")
    fit = prepare_fit(problem, device=device, phase_mode="observations", resolution_check=MeshResolutionCheck())
    before = deepcopy(problem.config)
    vector = normalized_vector(fit)
    reports, densities = [], []
    for ln_f in [-4, 0]:
        index = list(fit.fitable).index("nuisance@ln_f")
        vector[index] = (ln_f+5)/6
        props = physical(fit, vector)
        expected = original.predict_light_curves(props)
        actual = fit.predict_light_curves(props)
        report = fit.last_resolution_check
        assert report["status"] == "qualified"
        assert not report["continuum_certified"]
        reports.append(deepcopy(report))
        for band in actual:
            assert_allclose(actual[band], expected[band], rtol=3e-13, atol=2e-12)
            assert_allclose(fit.observer.fluxes[band], original.observer.fluxes[band], rtol=3e-13, atol=2e-12)
        densities.append(fit.ln_probability(vector))
        assert_allclose(densities[-1], original.ln_probability(vector), rtol=0, atol=1e-6)
        assert fit.last_resolution_check is not report  # fresh check, no certificate cache
    assert reports[0]["comparisons"] == reports[1]["comparisons"]
    assert densities[0] != densities[1]
    assert problem.config == before


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("kind", ["lsq", "mcmc"])
def test_resolution_failure_aborts_before_emergency_penalties(device: str, kind: str) -> None:
    require_device(device)
    problem = make_problem(sampling="uniform")
    for backend in [None, device]:
        fit = prepare_fit(problem, kind=kind, device=backend, phase_mode="observations",
                          resolution_check=MeshResolutionCheck())
        vector = normalized_vector(fit)
        callback = fit.model_to_fit if kind == "lsq" else fit.ln_probability
        with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError) as caught:
            callback(vector)
        assert caught.value.report["status"] == "budget_failed"
        assert caught.value.report["comparisons"][0]["distance_sigma"] > .1
        assert not fit._protocol_active
        assert fit._protocol_target_error is None
        if kind == "mcmc":
            assert fit._candidate_failure_diagnostics.total == 0
            assert np.isneginf(fit.ln_probability(np.full(vector.shape, -.1)))
            assert fit._candidate_failure_diagnostics.total == 0
        if backend:
            assert fit.hybrid_synthesis.calls == 0


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("kind", ["lsq", "mcmc"])
def test_original_invalid_physical_target_retains_rejection_policy(device: str, kind: str) -> None:
    require_device(device)
    problem = make_problem()
    # A real physical invalidity: the secondary's constrained temperature is negative.
    problem.config["primary"]["t_eff"]["min"] = 1000
    original = prepare_fit(problem, kind=kind)
    fit = prepare_fit(problem, kind=kind, device=device, phase_mode="observations",
                      resolution_check=MeshResolutionCheck())
    vector = normalized_vector(fit)
    vector[list(fit.fitable).index("primary@t_eff")] = 0
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True):
        if kind == "mcmc":
            assert np.isneginf(fit.ln_probability(vector))
            assert np.isneginf(original.ln_probability(vector))
            assert fit._candidate_failure_diagnostics.total == original._candidate_failure_diagnostics.total == 1
        else:
            assert_allclose(fit.model_to_fit(vector), original.model_to_fit(vector), rtol=0, atol=0)
    assert fit.last_resolution_check["status"] == "target_physics_failed"
    assert not fit._protocol_active
    assert fit._protocol_target_error is None


@pytest.mark.parametrize("kind", ["lsq", "mcmc"])
def test_failed_refinement_is_fatal_outside_original_penalty_handler(
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    fit = prepare_fit(protocol_problem(), kind=kind, phase_mode="observations",
                      resolution_check=MeshResolutionCheck())
    original_diagnostic = fit._direct_prediction

    def diagnostic(props: Any, disc: float, synthesis: Any, **kwargs: Any) -> Any:
        if disc == 8:
            raise RuntimeError("reference resource failure")
        return original_diagnostic(props, disc, synthesis, **kwargs)

    monkeypatch.setattr(fit, "_direct_prediction", diagnostic)
    callback = fit.model_to_fit if kind == "lsq" else fit.ln_probability
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError) as caught:
        callback(normalized_vector(fit))
    assert caught.value.report["status"] == "reference_failed"
    assert isinstance(caught.value.__cause__, RuntimeError)
    if kind == "mcmc":
        assert fit._candidate_failure_diagnostics.total == 0
    assert not fit._protocol_active


def test_nonmonotone_reference_cancellation_is_not_qualified(monkeypatch: pytest.MonkeyPatch) -> None:
    fit = prepare_fit(protocol_problem(), phase_mode="observations", resolution_check=MeshResolutionCheck())

    def diagnostic(_props: Any, disc: float, _synthesis: Any, **_kwargs: Any) -> Any:
        # Isolate diagnostic logic: same target/finest, different middle reference.
        return {b: np.ones(len(y))+(disc == 8)*.01 for b, y in fit.y_data.items()}

    monkeypatch.setattr(fit, "_direct_prediction", diagnostic)
    with pytest.raises(ResolutionNotQualifiedError) as caught:
        fit.ln_probability(normalized_vector(fit))
    a, b = caught.value.report["comparisons"]
    assert a["within_diagnostic_budget"]
    assert not b["within_diagnostic_budget"]


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_direct_unsupported_device_case_is_complete_cpu_fallback(device: str) -> None:
    require_device(device)
    problem = make_problem(mass_ratio=.4)
    cpu = prepare_fit(problem, phase_mode="observations")
    fit = prepare_fit(problem, phase_mode="observations", device=device)
    vector = normalized_vector(cpu)
    assert_allclose(fit.ln_probability(vector), cpu.ln_probability(vector), rtol=0, atol=0)
    assert fit.hybrid_synthesis.last_evaluation["backend"] == "cpu_fallback"


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_overcontact_direct_protocol_preserves_original_constraints(device: str) -> None:
    from scripts.validate_gpu_native import fit_problem

    require_device(device)
    problem = fit_problem("overcontact")
    cpu = prepare_fit(problem, morphology="overcontact", phase_mode="observations")
    fit = prepare_fit(problem, morphology="overcontact", phase_mode="observations",
                      device=device, general="device_integration")
    vector = normalized_vector(cpu)
    assert_allclose(fit.ln_probability(vector), cpu.ln_probability(vector), rtol=0, atol=1e-6)
    assert fit.hybrid_synthesis.last_evaluation["backend"] == "hybrid_native"


@pytest.mark.parametrize("references", [(6, 8), (8, 8), (4, 1.5), (8,), (8, np.nan)])
def test_invalid_reference_configuration_fails_explicitly(references: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="reference degrees"):
        MeshResolutionCheck(refinement_degrees=references)


def test_opt_in_configuration_and_runtime_fail_outside_penalties() -> None:
    with pytest.raises(ValueError, match="phase_mode"):
        ProtocolDetachedMCMCFit(phase_mode="adaptive")
    with pytest.raises(ValueError, match="requires phase_mode"):
        ProtocolDetachedMCMCFit(resolution_check=MeshResolutionCheck())
    lsq = prepare_fit(protocol_problem(), kind="lsq", phase_mode="observations")
    with pytest.raises(ValueError, match="serial workers"):
        lsq.fit(workers=2)
    fit = prepare_fit(protocol_problem(), phase_mode="observations", resolution_check=MeshResolutionCheck())
    vector = normalized_vector(fit)
    with settings.override(NUMBER_OF_PROCESSES=2, PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(RuntimeError):
        fit.ln_probability(vector)
    with settings.override(NUMBER_OF_MCMC_PROCESSES=2), pytest.raises(RuntimeError, match="serial"):
        fit.fit()
    fit.discretization = 8
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ValueError, match="strictly finer"):
        fit.ln_probability(vector)
