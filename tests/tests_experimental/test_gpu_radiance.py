"""Fresh device emission/optical-depth normalization in full eclipse curves."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import roots_legendre

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_binary import HybridBinaryLightCurve  # noqa: E402
from elisa.experimental.gpu_radiance import DeviceBlackbodyRadiance, passband_revision  # noqa: E402

from .test_gpu_binary import BANDS, binary_fixture, require_device  # noqa: E402
from .test_gpu_likelihood import fp64  # noqa: E402, F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("law", ["linear", "cosine", "logarithmic", "square_root"])
def test_fresh_normal_intensity_cpu_parity_energy_and_snapshot(device: str, law: str) -> None:
    from elisa import Observer, settings
    from elisa.atm.atm import NaiveInterpolatedAtm
    from elisa.binary_system.curves.utils import get_limbdarkening_cfs, prep_surface_params

    require_device(device)
    with settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW=law):
        binary = binary_fixture(law=law, discretization=12)
        engine = HybridBinaryLightCurve(BANDS, device=device, fallback="raise")
        system, _ = engine.build_intrinsic_system(binary)
        observer = Observer(passband=BANDS, system=binary)
        ld_cfs = get_limbdarkening_cfs(system, passband=observer.passband)
        spectrum = DeviceBlackbodyRadiance(observer.passband, law, device=device)
        batch = spectrum.prepare(system, ld_cfs).block_until_ready()
        result = jax.tree.map(np.asarray, batch.evaluate())
        prep_surface_params(system, return_values=False, write_to_containers=True, passband=observer.passband,
                            left_bandwidth=observer.left_bandwidth, right_bandwidth=observer.right_bandwidth)
        # Independent angular quadrature recovers emitted hemispheric flux
        # from intensity, demonstrating that pi/LD is applied exactly once.
        nodes, weights = roots_legendre(256)
        mu, weights = (nodes + 1) / 2, weights / 2
        a, b = 0.5 if law in ("linear", "cosine") else 0.3, 0.2
        angular = 1 - a * (1 - mu)
        if law == "logarithmic":
            angular -= b * mu * np.log(mu)
        elif law == "square_root":
            angular -= b * (1 - np.sqrt(mu))
        integrated = 2 * np.pi * np.sum(weights * mu * angular)
        for comp, name in enumerate(("primary", "secondary")):
            star = getattr(system, name)
            emitted = NaiveInterpolatedAtm.black_body_radiance(star.temperatures, passband=observer.passband)
            for index, band in enumerate(BANDS):
                actual = result["normal_radiance"][comp, index, :len(star.faces)]
                assert_allclose(actual, star.normal_radiance[band], rtol=3e-13, atol=0)
                assert_allclose(actual * integrated, emitted[band], rtol=3e-13, atol=0)
        assert result["proposal_valid"].all()
        # Mutating host candidate inputs cannot change a prepared packet.
        system.primary.temperatures[:] *= 1.001
        ld_cfs["primary"][BANDS[0]] = np.zeros_like(ld_cfs["primary"][BANDS[0]])
        with jax.transfer_guard("disallow_explicit"):
            same = jax.block_until_ready(batch.evaluate())
        assert same["normal_radiance"].device.platform == device
        assert_allclose(same["normal_radiance"], result["normal_radiance"], rtol=0, atol=0)
        changed = spectrum.prepare(system, ld_cfs).evaluate()
        assert not np.allclose(changed["normal_radiance"], result["normal_radiance"], rtol=1e-7, atol=0)
        # Nonpositive optical-depth integrals explicitly invalidate this device
        # row. They are not clamped into a plausible but different brightness.
        for band in BANDS:
            invalid = np.zeros_like(ld_cfs["primary"][band])
            invalid[:, 0] = 4
            ld_cfs["primary"][band] = invalid
        invalid = spectrum.prepare(system, ld_cfs).evaluate()
        assert not bool(invalid["proposal_valid"][0])
        assert bool(invalid["proposal_valid"][1])
        assert np.isnan(np.asarray(invalid["normal_radiance"])[0, :, :len(system.primary.faces)]).all()
        with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
            batch.evaluate()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_full_curve_response_revision_temperature_source_and_resident_radiance(
    device: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elisa import Observer, settings

    require_device(device)
    def unexpected_cpu_radiance(*_args, **_kwargs) -> None:
        pytest.fail("Device emission path called CPU spectral preparation")

    monkeypatch.setattr("elisa.experimental.gpu_binary.prep_surface_params", unexpected_cpu_radiance)
    binary = binary_fixture(discretization=12)
    observer = Observer(passband=BANDS, system=binary)
    phases = [0.5, 0.49, 0.25, 0.01, 0, 1, -0.5]
    engine = HybridBinaryLightCurve(BANDS, device=device, fallback="raise")
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear"):
        initial = engine(binary, phases, normalize=True, diagnostics=True, observer=observer)
        assert "device_radiance_ld_with_transfers" in initial.timings
        assert "cpu_radiance_ld" not in initial.timings
        revision = initial.diagnostics["radiance"]["response_revision"]
        assert "device temperatures" in initial.diagnostics["radiance"]["temperature_source"]
        # Alter the actual prepared response callable, retaining labels/table.
        # A label- or table-only cache would miss this physically changed input.
        old_response = observer.passband[BANDS[0]].akima
        observer.passband[BANDS[0]].akima = lambda w: old_response(w) * (w / 4500)**2
        assert revision != passband_revision(observer.passband)
        changed = engine(binary, phases, normalize=True, diagnostics=True, observer=observer)
        expected = deepcopy(observer).lc(phases=phases, normalize=True)[1]
        for band in BANDS:
            assert_allclose(changed.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)
        assert not np.allclose(changed.fluxes[BANDS[0]], initial.fluxes[BANDS[0]], rtol=1e-8, atol=1e-8)
        radiance = changed.diagnostics["radiance"]["result"]
        packets = changed.diagnostics["occultation"]["prepared_batches"]
        for packet in packets:
            assert packet._arrays[4] is radiance["normal_radiance"]
            assert packet._arrays[5] is radiance["coefficients"]
        with jax.transfer_guard("disallow_explicit"):
            for packet in packets:
                jax.block_until_ready(packet.evaluate())
    with settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW="linear"):
        reflected = engine(binary, phases, normalize=True, diagnostics=True)
        expected = Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=True)[1]
        assert reflected.diagnostics["radiance"]["temperature_source"] == "CPU post-reflection upload"
        for band in BANDS:
            assert_allclose(reflected.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_device_radiance_resource_failure_preserves_complete_cpu_fallback(device: str) -> None:
    from elisa import Observer, settings
    from elisa.atm.atm import get_standard_wavelengths

    require_device(device)
    binary = binary_fixture(discretization=12)
    observer = Observer(passband=BANDS, system=binary)
    # Narrow fixed response is legal on CPU but outside the device's minimum
    # three-wavelength quadrature; it is not a forbidden physical candidate.
    band = observer.passband[BANDS[0]]
    grid = get_standard_wavelengths()
    wave = grid[np.argmin(np.abs(grid - 4400))]
    band.left_bandwidth, band.right_bandwidth = wave - 1e-6, wave + 1e-6
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear"):
        expected = deepcopy(observer).lc(phases=[0, 0.25, 0.5], normalize=False)[1]
        result = HybridBinaryLightCurve(BANDS, device=device)(binary, [0, 0.25, 0.5], observer=observer)
    assert result.backend == "cpu_fallback"
    assert "wavelength" in result.fallback_reason
    for band in BANDS:
        assert_allclose(result.fluxes[band], expected[band], rtol=0, atol=0)
