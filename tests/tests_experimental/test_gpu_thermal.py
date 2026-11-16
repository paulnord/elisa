"""Intrinsic temperature parity and independent surface-energy conservation."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_thermal import GravityDarkeningTemperature  # noqa: E402

from .test_gpu_likelihood import fp64  # noqa: E402, F401
from .test_gpu_pipeline import fixture  # noqa: E402


def cpu_temperature(gravity: np.ndarray, areas: np.ndarray, t_eff: float, beta: float) -> np.ndarray:
    """Use the actual ELISa function with a non-symmetric unspotted data view."""
    from elisa.base.surface.temperature import calculate_effective_temperatures

    view = SimpleNamespace(
        areas=areas,
        potential_gradient_magnitudes=gravity,
        polar_potential_gradient_magnitude=float(np.median(gravity)),
        t_eff=t_eff,
        gravity_darkening=beta,
        has_spots=lambda: False,
        symmetry_test=lambda: False,
    )
    return calculate_effective_temperatures(view, gravity)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_resident_changing_geometry_temperature_and_energy(component: str, device: str) -> None:
    from elisa import const

    pipeline, params, *_ = fixture(device, component)
    t_eff, beta = np.array([4500.0, 6500.0, 12000.0]), np.array([0.0, 0.32, 1.0])
    batch = pipeline.prepare_thermal(*params, t_eff, beta).block_until_ready()
    jax.block_until_ready(batch.evaluate())
    with jax.transfer_guard("disallow_explicit"):
        device_result = jax.block_until_ready(batch.evaluate())
    result = jax.tree.map(np.asarray, device_result)
    assert result["proposal_valid"].all()
    thermal = result["thermal"]
    assert batch.nbytes == 3 * 9 * 8
    for row in range(3):
        g, a = result["gravity"]["face_gravity"][row], result["geometry"]["areas"][row]
        expected = cpu_temperature(g, a, t_eff[row], beta[row])
        assert_allclose(thermal["temperatures"][row], expected, rtol=3e-13, atol=0)
        assert_allclose(thermal["surface_flux"][row], const.STEFAN_BOLTZMAN_CONST * expected**4, rtol=3e-13, atol=0)
        # Independent sum over returned temperatures, not the kernel's own ratio.
        energy = np.sum(a * thermal["temperatures"][row] ** 4) / (np.sum(a) * t_eff[row] ** 4)
        assert_allclose(energy, 1.0, rtol=3e-13, atol=0)
    assert_allclose(thermal["luminosity_ratio"], [1.0] * 3, rtol=3e-13, atol=0)
    assert_allclose(thermal["temperatures"][0], np.full(thermal["temperatures"].shape[1], t_eff[0]), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_actual_cpu_temperature_without_reflection(device: str) -> None:
    from elisa import BinarySystem, Star, settings
    from elisa.binary_system.container import OrbitalPositionContainer

    fixture(device)
    common = {
        "surface_potential": 12,
        "synchronicity": 1,
        "albedo": 0.5,
        "gravity_darkening": 0.32,
        "atmosphere": "bb",
        "limb_darkening_coefficients": {"bolometric": [0.5]},
        "discretization_factor": 10,
    }
    binary = BinarySystem(
        Star(mass=1, t_eff=6500, **common),
        Star(mass=0.7, t_eff=5500, **common),
        inclination=88,
        period=3,
        eccentricity=0,
        argument_of_periastron=90,
    )
    system = OrbitalPositionContainer.from_binary_system(binary, binary.calculate_orbital_motion([0.0])[0])
    with settings.override(REFLECTION_EFFECT=False):
        system.build()
    for name in ["primary", "secondary"]:
        star = getattr(system, name)
        result = GravityDarkeningTemperature(device=device)(
            star.potential_gradient_magnitudes[None],
            star.areas[None],
            [star.t_eff],
            [star.gravity_darkening],
        )
        assert np.asarray(result["proposal_valid"]).all()
        assert_allclose(result["temperatures"][0], star.temperatures, rtol=3e-13, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_temperature_scale_invariance_and_snapshot(device: str) -> None:
    fixture(device)
    kernel = GravityDarkeningTemperature(device=device)
    gravity = np.array([[30.0, 100.0, 500.0, 2000.0]])
    areas = np.array([[0.1, 0.3, 0.7, 0.2]])
    reference = np.asarray(kernel(gravity, areas, [6500.0], [0.32])["temperatures"])
    for scale in [1e-20, 1e20]:
        assert_allclose(
            kernel(gravity * scale, areas / scale, [6500.0], [0.32])["temperatures"],
            reference,
            rtol=3e-13,
            atol=0,
        )
    pipeline, params, *_ = fixture(device)
    t, b = np.full(3, 6500.0), np.full(3, 0.32)
    batch = pipeline.prepare_thermal(*params, t, b).block_until_ready()
    before = np.asarray(batch.evaluate()["thermal"]["temperatures"])
    t[:], b[:] = 12000.0, 1.0
    assert_allclose(batch.evaluate()["thermal"]["temperatures"], before, rtol=0, atol=0)
    assert not np.allclose(before[0], before[-1])
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        batch.evaluate()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_thermal_unsupported_state_propagation(device: str) -> None:
    pipeline, params, *_ = fixture(device)
    values = list(params)
    values[2] = [1.0] * 3
    result = pipeline.prepare_thermal(*values, [6500.0] * 3, [0.32] * 3).evaluate()
    assert not np.asarray(result["proposal_valid"]).any()
    assert np.isnan(result["thermal"]["temperatures"]).all()
    kernel = GravityDarkeningTemperature(device=device)
    for bad in [0.0, -1.0, np.nan, np.inf]:
        result = kernel([[30.0, bad]], [[1.0, 1.0]], [6500.0], [0.32])
        assert not np.asarray(result["proposal_valid"]).any()
        assert np.isnan(result["temperatures"]).all()
        assert (np.asarray(result["status"]) == 1).all()
    result = kernel([[30.0, 50.0]], [[1.0, 1.0]], [1e308], [0.32])
    assert (np.asarray(result["status"]) == 2).all()


def test_thermal_input_validation() -> None:
    kernel = GravityDarkeningTemperature(device="cpu")
    for g, a, t, beta, message in [
        ([], [], [6500.0], [0.32], "shapes"),
        ([[30.0, 50.0]], [[1.0]], [6500.0], [0.32], "matching"),
        ([[30.0]], [[1.0]], [-1.0], [0.32], "positive"),
        ([[30.0]], [[1.0]], [6500.0], [1.1], "beta"),
        ([[30.0]], [[1.0]], [np.nan], [0.32], "finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            kernel(g, a, t, beta)
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        kernel([[30.0]], [[1.0]], [6500.0], [0.32])
