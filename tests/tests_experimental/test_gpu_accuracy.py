"""Independent noise-error algebra, spectral quadrature and resolution/parity separation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from elisa.experimental.gpu_accuracy import BlackbodySpectralProbe, resolution_error

from .test_gpu_binary import require_device
from .test_gpu_likelihood import fp64  # noqa: F401


def test_original_noise_log_likelihood_and_diagnostic_budget() -> None:
    from elisa.analytics.models import cost_fns

    original = {"v": np.array([1.0, 0.8, 0.7])}
    changed = {"v": np.array([1.0001, 0.7998, 0.7002])}
    data = {"v": np.array([0.9999, 0.8001, 0.7003])}
    errors = {"v": np.array([1e-4, 2e-4, 3e-4])}
    result = resolution_error(changed["v"], original["v"], errors["v"], observed=data["v"], ln_f=-2)
    expected = cost_fns.likelihood_fn(data, errors, changed, -2) - cost_fns.likelihood_fn(data, errors, original, -2)
    assert_allclose(result["delta_log_likelihood"], expected, rtol=1e-12, atol=1e-12)
    assert not result["within_diagnostic_budget"]
    exact = resolution_error(original["v"], original["v"], errors["v"])
    assert exact["within_diagnostic_budget"]
    assert exact["distance_sigma"] == 0
    noisy = resolution_error(changed["v"], original["v"], errors["v"], ln_f=0)
    clean = resolution_error(changed["v"], original["v"], errors["v"])
    assert_allclose(noisy["distance_squared"], clean["distance_squared"] / 2, rtol=1e-14, atol=0)
    assert_allclose(clean["expected_delta_log_likelihood"], -0.5 * clean["distance_squared"], rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["shape", "zero_sigma", "negative_sigma", "nan", "empty", "ln_f", "budget"])
def test_accuracy_inputs_fail_explicitly(bad: str) -> None:
    a, b, sigma, options = np.ones(3), np.ones(3), np.ones(3), {}
    if bad == "shape":
        a = np.ones(2)
    elif bad == "zero_sigma":
        sigma[0] = 0
    elif bad == "negative_sigma":
        sigma[0] = -1
    elif bad == "nan":
        a[0] = np.nan
    elif bad == "empty":
        a, b, sigma = np.array([]), np.array([]), np.array([])
    elif bad == "ln_f":
        options["ln_f"] = np.nan
    else:
        options["max_distance_sigma"] = 0
    with pytest.raises(ValueError, match=r"Resolution|Supply"):
        resolution_error(a, b, sigma, **options)


def test_log_wavelength_quadrature_against_independent_adaptive_integrals() -> None:
    from elisa import Observer, const
    from elisa.atm.atm import planck_function

    observer = Observer(passband=["Generic.Bessell.V", "bolometric"])
    probe = BlackbodySpectralProbe(observer.passband, order=16)
    temperatures = np.array([3000.0, 6500.0, 30000.0])
    flux = probe.surface_flux(temperatures)
    band = observer.passband["Generic.Bessell.V"]
    for i, temp in enumerate(temperatures):
        expected, _ = quad(
            lambda w, temp=temp: const.PI * 1e-10 * max(float(band.akima(w)), 0)
            * float(planck_function(w * 1e-10, temp)),
            band.left_bandwidth,
            band.right_bandwidth,
            epsrel=1e-11,
            points=np.asarray(band.table["wavelength"])[1:-1],
            limit=2000,
        )
        assert_allclose(flux["Generic.Bessell.V"][i], expected, rtol=1e-10, atol=0)
        bolo = observer.passband["bolometric"]
        factor = const.PLANCK_CONST * const.C / (const.BOLTZMAN_CONST * temp)
        value, _ = quad(
            lambda x: x**3 * np.exp(-x) / -np.expm1(-x),
            factor / (bolo.right_bandwidth * 1e-10),
            factor / (bolo.left_bandwidth * 1e-10),
            epsrel=1e-11,
        )
        expected = 2 * const.PI * (const.BOLTZMAN_CONST * temp) ** 4 / (const.PLANCK_CONST**3 * const.C**2) * value
        assert_allclose(flux["bolometric"][i], expected, rtol=1e-10, atol=0)


def test_spectral_observer_native_grid_control_and_owned_state() -> None:
    from elisa import BinarySystem, Observer, settings
    from scripts.validate_gpu_binary import fixture

    with settings.override(REFLECTION_EFFECT=True, NUMBER_OF_PROCESSES=1):
        binary = fixture({"inclination": 85, "discretization": 12})
        observer = Observer(passband=["Generic.Bessell.V", "bolometric"], system=binary)
        phases = np.array([0.5, 0.49, 0, 0.01, 0.25, 1.01])
        expected = observer.lc(phases=phases, normalize=True)[1]
        original = BinarySystem.compute_lightcurve
        probe = BlackbodySpectralProbe(observer.passband)
        actual = probe.observe(observer, phases, native_grid_control=True)["fluxes"]
        for band in expected:
            assert_allclose(actual[band], expected[band], rtol=3e-13, atol=2e-12)
        assert observer._system is binary
        assert BinarySystem.compute_lightcurve is original
        assert binary.primary.t_eff == 6500
        observer.passband["Generic.Bessell.V"].akima = np.ones_like
        with pytest.raises(ValueError, match="response changed"):
            probe.observe(observer, phases)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_mesh_resolution_error_is_separate_from_implementation_parity(device: str) -> None:
    from elisa import Observer, settings
    from elisa.experimental.gpu_binary import HybridBinaryLightCurve
    from scripts.validate_gpu_binary import fixture

    require_device(device)
    engine = HybridBinaryLightCurve(["Generic.Bessell.V", "bolometric"], device=device)
    results = []
    with settings.override(REFLECTION_EFFECT=True, NUMBER_OF_PROCESSES=1):
        for disc in [12, 4]:
            binary = fixture({"inclination": 85, "discretization": disc})
            phases = np.array([0, 0.01, 0.025, 0.25, 0.5, 0.515]) + 0.017
            expected = Observer(passband=list(engine.passbands), system=binary).lc(phases=phases, normalize=True)[1]
            actual = engine(binary, phases, normalize=True)
            for band in expected:
                assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)
            if disc == 4:
                assert actual.backend == "cpu_fallback"
            results.append(expected["Generic.Bessell.V"])
    change = resolution_error(results[0], results[1], 1e-4)
    assert change["max_abs_sigma"] > 1
    assert not change["within_diagnostic_budget"]
