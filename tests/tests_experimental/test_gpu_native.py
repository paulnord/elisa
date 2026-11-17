"""Broader original physical routers, device surface integration and full fallback."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_binary import HybridBinaryLightCurve  # noqa: E402
from scripts.validate_gpu_atmosphere import table_settings  # noqa: E402
from scripts.validate_gpu_fit import fit_assessment, normalized_vector, prepare_fit  # noqa: E402
from scripts.validate_gpu_native import (  # noqa: E402
    BANDS,
    CASES,
    case_settings,
    face_checks,
    fit_problem,
    fixture,
    phases_for,
)

from .test_gpu_binary import require_device  # noqa: E402
from .test_gpu_likelihood import fp64  # noqa: E402, F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_broader_complete_native_physics_curves(device: str, case: dict) -> None:
    from elisa import BinarySystem, Observer
    from elisa.binary_system.curves import lc_point

    require_device(device)
    with table_settings(), case_settings(case):
        binary = fixture(case)
        unchanged = BinarySystem.compute_lightcurve, lc_point.compute_lc_on_pos
        original = deepcopy(binary)
        phases = np.array([.5, .015, .0, .25, 1.015, -.5]) + binary.phase_shift
        if case.get("approximation", False):
            with pytest.raises(ValueError, match="zero-size"):
                Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)
            with pytest.raises(ValueError, match="zero-size"):
                HybridBinaryLightCurve(list(BANDS), device=device, general="device_integration")(
                    binary, phases, normalize=True,
                )
            phases = phases_for(binary)
        actual = HybridBinaryLightCurve(list(BANDS), device=device, general="device_integration", fallback="raise")(
            binary, phases, normalize=True, diagnostics=True,
        )
        expected = Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)[1]
        default = HybridBinaryLightCurve(list(BANDS), device=device)(binary, phases, normalize=True)
        assert actual.backend == "hybrid_native"
        assert default.backend == "cpu_fallback"
        for band in BANDS:
            assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)
            assert_allclose(default.fluxes[band], expected[band], rtol=0, atol=0)
        assert (BinarySystem.compute_lightcurve, lc_point.compute_lc_on_pos) == unchanged
        assert binary.primary.t_eff == original.primary.t_eff
        assert binary.eccentricity == original.eccentricity
        errors = face_checks(actual)
        assert errors["max_face_flux_over_component_abs"] <= 2e-12
        assert errors["max_component_flux_relative"] <= 3e-13
        if case.get("eccentricity", 0):
            assert np.ptp(errors["distances"]) > .05, "actual changing separation must reach the device callback"


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_changed_spot_geometry_and_immutable_surface_packet(device: str) -> None:
    from elisa import Observer

    require_device(device)
    case = CASES[1]
    engine = HybridBinaryLightCurve(list(BANDS), device=device, general="device_integration", fallback="raise")
    curves = []
    with table_settings(), case_settings(case):
        for longitude, factor in [(45, .9), (75, .9), (75, 1.03)]:
            binary = fixture({**case, "longitude": longitude, "temperature_factor": factor})
            actual = engine(binary, [0, .1, .25, .5, 1.1], normalize=True, diagnostics=True)
            expected = Observer(passband=list(BANDS), system=binary).lc(phases=[0, .1, .25, .5, 1.1], normalize=True)[1]
            for band in BANDS:
                assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)
            curves.append(actual.fluxes[BANDS[0]])
        assert not np.allclose(curves[0], curves[1], rtol=1e-8, atol=1e-8)
        assert not np.allclose(curves[1], curves[2], rtol=1e-8, atol=1e-8)
        record = actual.diagnostics["records"][0]
        before = jax.tree.map(np.asarray, record.packet.evaluate())
        record.system.primary.normal_radiance[BANDS[0]][:] = 0
        with jax.transfer_guard("disallow_explicit"):
            result = jax.block_until_ready(record.packet.evaluate())
        assert result[1].device.platform == device
        for a, b in zip(result, before, strict=True):
            assert_allclose(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("law", ["logarithmic", "square_root"])
def test_broader_native_limb_laws(device: str, law: str) -> None:
    from elisa import Observer

    require_device(device)
    case = {**CASES[0], "law": law}
    with table_settings(), case_settings(case):
        binary = fixture(case)
        actual = HybridBinaryLightCurve(list(BANDS), device=device, general="device_integration", fallback="raise")(
            binary, [0, .01, .25, .5], normalize=True,
        )
        expected = Observer(passband=list(BANDS), system=binary).lc(phases=[0, .01, .25, .5], normalize=True)[1]
        for band in BANDS:
            assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_original_source_error_restores_private_observer_proxy(device: str, tmp_path: Path) -> None:
    from elisa import Observer, settings
    from elisa.base.error import ElisaLimbDarkeningError
    from elisa.experimental.gpu_native import NativePhysicsIntegrator

    require_device(device)
    case = CASES[-1]
    with table_settings(), case_settings(case), settings.override(LD_TABLES=str(tmp_path)):
        binary = fixture(case)
        observer = Observer(passband=list(BANDS), system=binary)
        with pytest.raises(ElisaLimbDarkeningError):
            NativePhysicsIntegrator("linear", device=device)(observer, np.array([0, .25]), normalize=True)
        assert observer._system is binary


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_partial_attempt_resource_fallback_keeps_original_posterior(
    device: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elisa import settings

    require_device(device)
    with table_settings(), case_settings({"reflection": False}), settings.override(PENALIZE_ALL_EXCEPTIONS=True):
        problem = fit_problem("spot", shape=True)
        original = prepare_fit(problem)
        hybrid = prepare_fit(problem, device=device, general="device_integration")
        vector = normalized_vector(original)
        monkeypatch.setattr("elisa.experimental.gpu_native._MAX_CALLBACKS", 1)
        actual, expected = hybrid.ln_probability(vector), original.ln_probability(vector)
        assert np.isfinite(actual)
        assert_allclose(actual, expected, rtol=0, atol=0)
        assert hybrid.hybrid_synthesis.last_evaluation["backend"] == "cpu_fallback"
        with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain enabled"):
            hybrid.ln_probability(vector)
        with settings.override(NUMBER_OF_PROCESSES=2), pytest.raises(RuntimeError, match="NUMBER_OF_PROCESSES=1"):
            hybrid.ln_probability(vector)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_original_overcontact_public_fit_constraints_and_sampler(device: str) -> None:
    require_device(device)
    with table_settings(), case_settings({"reflection": False}):
        result = fit_assessment(device, steps=2, problem=fit_problem("overcontact"),
                                morphology="over-contact", general="device_integration")
    for row in result["least_squares"].values():
        config = row["serialized_result"]
        assert config["primary"]["surface_potential"]["value"] == config["secondary"]["surface_potential"]["value"]
    assert result["chain_parity"]["max_abs"] == 0
    assert result["chain_density_parity"]["max_abs"] <= 1e-6


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_selected_length_numpy_reduction_boundaries_and_cancellation(device: str) -> None:
    from elisa.experimental.gpu_ordered import OrderedSurfaceFluxIntegrator

    require_device(device)
    counts = np.array([0, 1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 511, 512, 8191, 8192])
    rng = np.random.default_rng(907)
    values = rng.normal(size=(len(counts), 8192))*10.0**rng.integers(-10, 11, size=(len(counts), 8192))
    values[9, :9] = [1e100, 1, -1e100, 3, 1e-100, -3, 7, 11, -11]
    kernel = OrderedSurfaceFluxIntegrator("linear", device=device)
    arrays = tuple(jax.device_put(x, kernel.device) for x in (values, counts))
    result = kernel._sum(*arrays)
    expected = np.array([np.sum(row[:n]) for row, n in zip(values, counts, strict=True)])
    assert_allclose(result, expected, rtol=0, atol=0)
    with jax.transfer_guard("disallow_explicit"):
        result = jax.block_until_ready(kernel._sum(*arrays))
    assert result.device.platform == device
