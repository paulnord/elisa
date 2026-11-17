"""Actual fitter serialization/mapping/costs, fits and short physical sampler replay."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_fit import HybridDetachedLeastSquaresFit, HybridDetachedMCMCFit  # noqa: E402
from scripts.validate_gpu_fit import (  # noqa: E402
    BANDS,
    fit_assessment,
    make_problem,
    normalized_vector,
    physical,
    prepare_fit,
)

from .test_gpu_binary import require_device  # noqa: E402
from .test_gpu_likelihood import fp64  # noqa: E402, F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(("mode", "jd", "standard"), [("direct", False, False), ("direct", True, False),
                                                     ("manual_equal_length", False, False),
                                                     ("uniform", False, False), ("direct", False, True)])
def test_original_mapping_likelihood_priors_and_fresh_physics(
    device: str, mode: str, jd: bool, standard: bool,
) -> None:
    from elisa import settings
    from elisa.analytics.models import lc
    from elisa.analytics.params import parameters
    from elisa.analytics.tools.utils import time_layer_resolver

    require_device(device)
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1):
        problem = make_problem(jd=jd, sampling=mode, free_nuisance=True, shape_parameters=True, standard=standard)
        original, hybrid = prepare_fit(problem), prepare_fit(problem, device=device)
        vector = normalized_vector(original)
        predictions = []
        for current in (vector, np.linspace(0.25, 0.7, len(vector))):
            props = physical(original, current)
            cpu, gpu = original.predict_light_curves(props), hybrid.predict_light_curves(props)
            assert props["secondary@t_eff"] == props["primary@t_eff"] - 1500
            for band in BANDS:
                assert_allclose(gpu[band], cpu[band], rtol=3e-13, atol=2e-12)
                assert_allclose(gpu[band].mean(), 1, rtol=0, atol=3e-15)
            assert_allclose(hybrid.likelihood(current), original.likelihood(current), rtol=0, atol=1e-6)
            assert_allclose(hybrid.ln_probability(current), original.ln_probability(current), rtol=0, atol=1e-6)
            assert hybrid.hybrid_synthesis.last_evaluation["backend"] == "hybrid"
            predictions.append(gpu)
        assert not np.allclose(predictions[0][BANDS[0]], predictions[1][BANDS[0]], rtol=1e-8, atol=1e-8)
        # Independently map CPU raw support, including equal-length DIFFERENT
        # grids, then normalize each band's actual observation multiplicities.
        phases, kwargs = time_layer_resolver(original.x_data_reduced, **props)
        support = phases if original.fit_xs is None else original.fit_xs
        if mode == "manual_equal_length":
            assert len(support) == len(phases)
            assert not np.array_equal(support, phases)
        kwargs = parameters.extend_json_with_atm_params(kwargs, atmosphere_model=problem.atmosphere,
                                                       limb_darkening_coefficients=problem.ld)
        raw = lc.synthetic_binary(support, original.discretization, original.observer, **kwargs)
        for band in BANDS:
            target = phases[original.x_data_reducer[band]]
            mapped = (raw[band][original.x_data_reducer[band]] if np.array_equal(support, phases)
                      else interp1d(support, raw[band], kind="cubic")(target))
            assert_allclose(predictions[-1][band], mapped / mapped.mean(), rtol=3e-13, atol=2e-12)
        calls = hybrid.hybrid_synthesis.calls
        assert np.isneginf(hybrid.ln_probability(np.full(vector.shape, -0.1)))
        assert hybrid.hybrid_synthesis.calls == calls
        original.sigmas[0] = hybrid.sigmas[0] = 0.2
        assert_allclose(hybrid.ln_probability(vector), original.ln_probability(vector), rtol=0, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_original_lsq_residual_and_reflection_callbacks(device: str) -> None:
    from elisa import settings
    from elisa.analytics.models import cost_fns

    require_device(device)
    with settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1):
        problem = make_problem()
        cpu, gpu = prepare_fit(problem, kind="lsq"), prepare_fit(problem, kind="lsq", device=device)
        vector = normalized_vector(cpu)
        actual, expected = gpu.model_to_fit(vector), cpu.model_to_fit(vector)
        assert_allclose(actual, expected, rtol=3e-13, atol=2e-8)
        props = physical(cpu, vector)
        direct = gpu.predict_light_curves(props)
        independent = cost_fns.standardized_residuals(gpu.y_data, gpu.y_err, direct)
        assert_allclose(actual, independent, rtol=3e-13, atol=2e-8)
        assert_allclose(gpu.prediction_r_squared(props), cpu.prediction_r_squared(props), rtol=3e-13, atol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_unsupported_device_model_uses_finite_cpu_posterior(device: str) -> None:
    from elisa import settings

    require_device(device)
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear", PENALIZE_ALL_EXCEPTIONS=True):
        problem = make_problem(mass_ratio=0.4)
        cpu, gpu = prepare_fit(problem), prepare_fit(problem, device=device)
        vector = normalized_vector(cpu)
        actual, expected = gpu.ln_probability(vector), cpu.ln_probability(vector)
        assert np.isfinite(actual)
        assert_allclose(actual, expected, rtol=0, atol=0)
        assert gpu.hybrid_synthesis.last_evaluation["backend"] == "cpu_fallback"
        assert gpu.hybrid_synthesis.cpu_fallback_calls == 1
        # A disabled device precision mode is infrastructure, outside the
        # original emergency physical-candidate penalty policy.
        with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain enabled"):
            gpu.ln_probability(vector)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_complete_public_fit_and_short_sampler_lifecycles(device: str) -> None:
    from elisa import settings

    require_device(device)
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1,
                           NUMBER_OF_MCMC_PROCESSES=1, SUPPRESS_WARNINGS=True):
        result = fit_assessment(device, steps=2)
    assert result["chain_parity"]["max_abs"] == 0
    assert result["chain_density_parity"]["max_abs"] <= 1e-6
    assert "no convergence" in result["sampler_limitations"]


def test_explicit_serial_runtime_and_synthesis_contracts() -> None:
    from elisa import settings
    from elisa.experimental.gpu_fit import HybridBinarySynthesis

    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear"):
        problem = make_problem()
        original = deepcopy(problem.config)
        fit = prepare_fit(problem, device="cpu")
        fit.ln_probability(normalized_vector(fit))
        assert original == problem.config
        props = physical(fit, normalized_vector(fit))
        with pytest.raises(ValueError, match="owns its synthesis"):
            fit.predict_light_curves(props, synthesis=lambda *_args, **_kwargs: {})
        with settings.override(NUMBER_OF_MCMC_PROCESSES=2), pytest.raises(ValueError, match="PROCESSES=1"):
            fit.fit()
    with pytest.raises(ValueError, match="serial workers"):
        HybridDetachedLeastSquaresFit(device="cpu").fit(workers=2)
    with pytest.raises(ValueError, match="coverage"):
        HybridBinarySynthesis(device="cpu", coverage="stale")
    assert isinstance(fit, HybridDetachedMCMCFit)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_prepared_observer_response_and_raw_state_are_preserved(device: str) -> None:
    from elisa import settings, units

    require_device(device)
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear"):
        problem = make_problem()
        original, hybrid = prepare_fit(problem), prepare_fit(problem, device=device)
        props = physical(original, normalized_vector(original))
        standard = original.predict_light_curves(props)
        for fit in (original, hybrid):
            table = fit.observer.passband[BANDS[0]].table.copy(deep=True)
            table[settings.PASSBAND_DATAFRAME_THROUGHPUT] *= np.linspace(0.3, 1.7, len(table))
            fit.observer.passband[BANDS[0]].table = table
        expected, actual = original.predict_light_curves(props), hybrid.predict_light_curves(props)
        assert not np.allclose(expected[BANDS[0]], standard[BANDS[0]], rtol=1e-8, atol=1e-8)
        for band in BANDS:
            assert_allclose(actual[band], expected[band], rtol=3e-13, atol=2e-12)
            assert_allclose(hybrid.observer.fluxes[band], original.observer.fluxes[band], rtol=3e-13, atol=2e-12)
        assert_allclose(hybrid.observer.phases, original.observer.phases, rtol=0, atol=0)
        assert hybrid.observer.flux_unit == units.dimensionless_unscaled
