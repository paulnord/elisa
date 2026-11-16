"""Real passband quadrature, Planck limits and resident changing Roche emission."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_blackbody import (  # noqa: E402
    BlackbodyPassbandFlux,
    RocheBlackbodyPipeline,
    _simpson_weights,
)

from .test_gpu_likelihood import fp64  # noqa: E402, F401
from .test_gpu_pipeline import fixture  # noqa: E402


def passbands() -> dict:
    from elisa import Observer

    return Observer(passband=["Generic.Bessell.B", "Generic.Bessell.V", "bolometric"]).passband


def kernel(device: str, chunk_size: int = 32) -> BlackbodyPassbandFlux:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    return BlackbodyPassbandFlux(passbands(), device=device, chunk_size=chunk_size)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("chunk_size", [1, 32, 128])
def test_real_cpu_blackbody_passbands(device: str, chunk_size: int) -> None:
    from elisa.atm.atm import NaiveInterpolatedAtm

    model = kernel(device, chunk_size)
    temperatures = np.array([[3000.0, 4500.123456789, 6500.0, 6500.0 + 1e-7], [9500.0, 12000.0, 20000.0, 40000.0]])
    result = jax.tree.map(np.asarray, model(temperatures))
    with np.errstate(over="ignore"):  # CPU Planck UV tails tend to zero through exp overflow
        expected = NaiveInterpolatedAtm.black_body_radiance(temperatures.ravel(), passband=passbands())
    assert result["proposal_valid"].all()
    for row, name in enumerate(model.passbands):
        assert_allclose(result["surface_flux"][:, row], expected[name].reshape(temperatures.shape), rtol=3e-13, atol=0)
    # Preserve tiny continuous temperature changes instead of whole-K caching.
    assert result["surface_flux"][0, 0, 3] > result["surface_flux"][0, 0, 2]


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_resident_roche_to_fresh_passband_flux(device: str, component: str) -> None:
    from elisa.atm.atm import NaiveInterpolatedAtm

    geometry, params, *_ = fixture(device, component)
    model = RocheBlackbodyPipeline(geometry, passbands())
    temperatures, beta = np.array([4500.0, 6500.0, 12000.0]), np.array([0.0, 0.32, 1.0])
    batch = model.prepare(*params, temperatures, beta).block_until_ready()
    jax.block_until_ready(batch.evaluate())
    with jax.transfer_guard("disallow_explicit"):
        result_device = jax.block_until_ready(batch.evaluate())
    assert result_device["blackbody"]["surface_flux"].device.platform == device
    result = jax.tree.map(np.asarray, result_device)
    assert result["proposal_valid"].all()
    t = result["thermal"]["temperatures"]
    expected = NaiveInterpolatedAtm.black_body_radiance(t.ravel(), passband=passbands())
    for index, name in enumerate(model.passbands):
        assert_allclose(
            result["blackbody"]["surface_flux"][:, index], expected[name].reshape(t.shape), rtol=3e-13, atol=0,
        )
    assert batch.nbytes == 3 * 9 * 8
    # Snapshot includes temperature parameters and sampled passbands/topology.
    before = result["blackbody"]["surface_flux"].copy()
    temperatures[:], params[0][:] = 8000.0, 1.1
    assert_allclose(batch.evaluate()["blackbody"]["surface_flux"], before, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_blackbody_unsupported_temperatures_and_roots(device: str) -> None:
    model = kernel(device)
    result = jax.tree.map(np.asarray, model([[5000.0, 6000.0], [0.0, 6000.0], [np.nan, 6000.0], [1e308, 1e308]]))
    assert np.array_equal(result["status"], [0, 1, 1, 2])
    assert np.array_equal(result["proposal_valid"], [True, False, False, False])
    assert np.isnan(result["surface_flux"][1:]).all()
    geometry, params, *_ = fixture(device)
    params[2][1] = 2  # root outside this solver's admitted quarter-separation interval
    result = RocheBlackbodyPipeline(geometry, passbands()).prepare(*params, [6500.0] * 3, [0.32] * 3).evaluate()
    assert not np.asarray(result["proposal_valid"])[1]
    assert np.isnan(np.asarray(result["blackbody"]["surface_flux"])[1]).all()


def test_cpu_quadrature_weights_even_odd_and_irregular() -> None:
    from elisa import umpy

    for count in [3, 4, 11, 12, 67]:
        x = np.random.default_rng(count).uniform(0.5, 2.0, count).cumsum()
        y = np.random.default_rng(count + 1).uniform(0, 2.0, (3, count))
        assert_allclose(y @ _simpson_weights(x), umpy.simps(y, x, axis=1), rtol=3e-15, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_finite_bolometric_grid_energy_limit(device: str) -> None:
    from elisa import const

    model = kernel(device)
    t = np.array([[3000., 6500., 20000., 40000.]])
    flux = np.asarray(model(t)["surface_flux"])[:, model.passbands.index("bolometric")]
    # Independent infinite-band Planck integral. The CPU's finite 50--2e6 A
    # grid truncates the IR tail and has its existing quadrature error. This
    # physical grid-accuracy bound is distinct from the 3e-13 CPU parity tests.
    sigma = 2 * np.pi**5 * const.BOLTZMAN_CONST**4 / (15 * const.PLANCK_CONST**3 * const.C**2)
    assert_allclose(flux, sigma * t**4, rtol=1e-6, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_fixed_passband_snapshot_and_negative_boundary_response(device: str) -> None:
    from elisa.atm.atm import NaiveInterpolatedAtm

    kernel(device)  # verify availability before constructing another snapshot
    bands = passbands()
    band = bands["Generic.Bessell.B"]
    response = band.akima
    # Exercise CPU-compatible negative Akima boundary overshoot clipping.
    band.akima = lambda wave: np.where(wave < band.left_bandwidth + 100, -1e-5, response(wave))
    model = BlackbodyPassbandFlux(bands, device=device)
    t = np.array([[5500., 6500.]])
    expected = NaiveInterpolatedAtm.black_body_radiance(t.ravel(), passband=bands)
    before = np.asarray(model(t)["surface_flux"])
    for index, name in enumerate(model.passbands):
        assert_allclose(before[:, index], expected[name].reshape(t.shape), rtol=3e-13, atol=0)
    band.akima = np.zeros_like
    assert_allclose(model(t)["surface_flux"], before, rtol=0, atol=0)


def test_passband_configuration_and_precision_boundaries() -> None:
    model = kernel("cpu")
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="FP64"):
        model([[6000.0]])
    with pytest.raises(ValueError, match="proposals,faces"):
        model([6000.0])
    for pb, chunk in [({}, 32), (passbands(), 0), (passbands(), True), (passbands(), 129)]:
        with pytest.raises(ValueError, match=r"passbands|chunk_size"):
            BlackbodyPassbandFlux(pb, device="cpu", chunk_size=chunk)
    bad = deepcopy(passbands())
    bad["bolometric"].wave_to_si_mult = 1.0
    with pytest.raises(ValueError, match="Angstrom"):
        BlackbodyPassbandFlux(bad, device="cpu")
