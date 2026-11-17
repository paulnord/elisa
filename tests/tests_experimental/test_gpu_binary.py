"""Corrected state and complete HYBRID curves versus unmodified ELISa.

Parity thresholds retain the surface kernel's 3e-13 FP64 relative gate and
the existing fitter prediction tests' 2e-12 absolute normalized-flux gate.
These test implementation parity, not mesh resolution or astrophysical error.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_binary import HybridBinaryLightCurve, UnsupportedHybridModelError  # noqa: E402

from .test_gpu_likelihood import fp64  # noqa: E402, F401

BANDS = ["Generic.Bessell.B", "Generic.Bessell.V", "bolometric"]


def binary_fixture(
    q: float = 0.7, inclination: float = 88, *, spin: tuple = (1, 1),
    potentials: tuple = (8, 12), temperatures: tuple = (6500, 5000), discretization: float = 10,
    law: str = "linear", eccentricity: float = 0, phase_shift: float = 0, additional_light: float = 0,
) -> Any:
    from elisa import BinarySystem, Star

    coeff = [0.5] if law in ("linear", "cosine") else [0.3, 0.2]
    common = {"albedo": 0.5, "gravity_darkening": 0.32, "atmosphere": "bb",
              "limb_darkening_coefficients": dict.fromkeys(BANDS, coeff),
              "discretization_factor": discretization}
    return BinarySystem(
        Star(mass=1.2, t_eff=temperatures[0], surface_potential=potentials[0], synchronicity=spin[0], **common),
        Star(mass=1.2 * q, t_eff=temperatures[1], surface_potential=potentials[1], synchronicity=spin[1], **common),
        inclination=inclination, period=3, eccentricity=eccentricity, argument_of_periastron=90,
        phase_shift=phase_shift, additional_light=additional_light, distance=25,
    )


def require_device(device: str) -> None:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")


def contact_phases(binary: Any) -> np.ndarray:
    """Independent spherical-radius contact ESTIMATES, with nearby phases.

    These locate useful samples; the oracle and actual occultation still use
    the Roche mesh, CPU silhouette and CPU partial-face clipping algorithm.
    """
    radius_sum = binary.primary.side_radius + binary.secondary.side_radius
    radius_diff = abs(binary.primary.side_radius - binary.secondary.side_radius)
    sini, cosi = np.sin(binary.inclination), np.cos(binary.inclination)
    contacts = [np.arcsin(np.sqrt((radius**2 - cosi**2) / sini**2)) / (2 * np.pi)
                for radius in (radius_sum, radius_diff) if radius > abs(cosi)]
    samples = [0, 0.001, 0.005, 0.01, 0.02, 0.035, 0.05, 0.1, 0.25, 0.45, 0.48, 0.49, 0.5]
    for contact in contacts:
        for centre in (0, 0.5):
            for sign in (-1, 1):
                samples.extend(centre + sign * contact + np.array([-1e-5, 0, 1e-5]))
    # Deliberately unsorted, duplicated, cross-cycle input, not a sorted grid.
    return np.concatenate((np.array(samples)[::-1], [0.75, 1.0, -0.5, 0.01, 1.005]))


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(("q", "spin"), [(0.5, (0.5, 1.5)), (0.7, (1, 1)), (2, (1.5, 0.5))])
def test_complete_corrected_container_state(device: str, q: float, spin: tuple) -> None:
    from elisa import const, settings
    from elisa.base.surface.faces import calculate_surface_centres
    from elisa.binary_system.curves.c_router import prep_initial_system

    require_device(device)
    binary = binary_fixture(q=q, spin=spin, potentials=(12, 16))
    engine = HybridBinaryLightCurve(BANDS, device=device, fallback="raise")
    with settings.override(REFLECTION_EFFECT=False):
        expected = prep_initial_system(binary)
        actual, _ = engine.build_intrinsic_system(binary)
    for name in ("primary", "secondary"):
        cpu, gpu = getattr(expected, name), getattr(actual, name)
        assert np.array_equal(gpu.faces, cpu.faces)
        for key in ("points", "face_centres", "normals", "areas", "velocities",
                    "potential_gradient_magnitudes", "log_g", "temperatures"):
            assert_allclose(getattr(gpu, key), getattr(cpu, key), rtol=3e-13, atol=64 * np.finfo(float).eps)
        # Cached centres describe RAW vertices, not the corrected ones.
        assert not np.allclose(gpu.face_centres, calculate_surface_centres(gpu.points, gpu.faces), rtol=1e-6, atol=1e-8)
        energy = np.sum(gpu.areas * const.STEFAN_BOLTZMAN_CONST * gpu.temperatures**4)
        target = np.sum(gpu.areas) * const.STEFAN_BOLTZMAN_CONST * gpu.t_eff**4
        assert_allclose(energy / target, 1, rtol=3e-13, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_corrected_state_snapshot_device_guard_and_invalid(device: str) -> None:
    from elisa import const, settings
    from elisa.binary_system.container import OrbitalPositionContainer

    require_device(device)
    engine = HybridBinaryLightCurve(BANDS, device=device, fallback="raise")
    binary = binary_fixture()
    raw = OrbitalPositionContainer.from_binary_system(binary, const.Position(0, 1, 0, 0, 0))
    raw.build_mesh(1.0)
    raw.build_faces(1.0)
    for name in ("primary", "secondary"):
        star = getattr(raw, name)
        kernel = engine._states[name]
        batch = kernel.prepare(star, binary.mass_ratio, raw.primary.mass, binary.semi_major_axis).block_until_ready()
        reference = jax.tree.map(np.asarray, batch.evaluate())
        before = star.points.copy()
        star.points[:] = np.nan
        with jax.transfer_guard("disallow_explicit"):
            result = jax.block_until_ready(batch.evaluate())
        assert result["points"].device.platform == device
        assert_allclose(result["points"], reference["points"], rtol=0, atol=0)
        with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
            batch.evaluate()
        star.points[:] = before
        centre = np.array([0 if name == "primary" else 1, 0, 0])
        star.points = (star.points - centre) * 10 + centre
        invalid = kernel.prepare(star, binary.mass_ratio, raw.primary.mass, binary.semi_major_axis).evaluate()
        assert not bool(invalid["proposal_valid"])
        assert np.isnan(invalid["temperatures"]).all()
        assert (np.asarray(invalid["gravity_status"]) == 1).all()
    with settings.override(REFLECTION_EFFECT=False), jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        engine(binary, [0, 0.25])


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("inclination", [60, 79, 85, 90])
def test_fresh_changed_curves_both_eclipses_contacts_and_order(device: str, inclination: float) -> None:
    from elisa import Observer, settings

    require_device(device)
    binary = binary_fixture(inclination=inclination, phase_shift=0.017, additional_light=0.2)
    phases = contact_phases(binary) + binary.phase_shift
    engine = HybridBinaryLightCurve(BANDS, device=device, fallback="raise")
    # Reflection itself is the original CPU routine, using fresh corrected
    # areas, PRE-correction centres/normals/gravity and new intrinsic T.
    with settings.override(REFLECTION_EFFECT=True, NUMBER_OF_PROCESSES=1, LIMB_DARKENING_LAW="linear"):
        result = engine(binary, phases, normalize=False, diagnostics=True)
        out_phases, expected = Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=False)
        normalized = engine(binary, phases, normalize=True)
        _, expected_normalized = Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=True)
    assert result.backend == "hybrid"
    assert_allclose(result.phases, out_phases, rtol=0, atol=0)
    for band in BANDS:
        assert_allclose(result.fluxes[band], expected[band], rtol=3e-13, atol=0)
        assert_allclose(normalized.fluxes[band], expected_normalized[band], rtol=3e-13, atol=2e-12)
        components = result.component_fluxes[band]
        assert_allclose(
            (components.sum(axis=1) + components.sum(axis=1).mean() * 0.2 / 0.8) / binary.distance**2,
            result.fluxes[band], rtol=3e-13, atol=0,
        )
    # Changed inclination must change coverage, not just the normal projection.
    captures = result.diagnostics["positions"]
    fractions = [star.coverage[star.indices] / (star.areas[star.indices] * binary.semi_major_axis**2)
                 for pos in captures for star in (pos.primary, pos.secondary)]
    fractions = np.concatenate(fractions)
    if inclination == 60:
        # CPU's conservative conjunction gate may still request clipping;
        # actual uncovered areas, rather than the dispatch hint, prove no eclipse.
        assert_allclose(fractions, 1, rtol=3e-13, atol=0)
    else:
        assert np.any((fractions > 0.01) & (fractions < 0.99)), "actual partial-face clipping was not exercised"
    if inclination == 90:
        conjunction = next(pos for pos in captures if pos.position.phase == 0.5)
        assert np.all(conjunction.secondary.coverage[conjunction.secondary.indices] == 0), "total secondary eclipse"


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("law", ["logarithmic", "square_root"])
def test_fresh_parameters_and_limb_laws(device: str, law: str) -> None:
    from elisa import Observer, settings

    require_device(device)
    engine = HybridBinaryLightCurve(BANDS, device=device, fallback="raise")
    results, counts = [], []
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW=law, NUMBER_OF_PROCESSES=1):
        for q, potentials, temps, spin, disc in [(0.5, (8, 12), (6500, 5000), (0.5, 1.5), 10),
                                                 (2, (12, 16), (7500, 4800), (1.5, 0.5), 8)]:
            binary = binary_fixture(q=q, potentials=potentials, temperatures=temps, spin=spin,
                                    law=law, discretization=disc)
            phases = contact_phases(binary)
            result = engine(binary, phases, normalize=True, diagnostics=True)
            _, expected = Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=True)
            for band in BANDS:
                assert_allclose(result.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)
            results.append(result)
            counts.append(tuple(len(getattr(result.diagnostics["initial_system"], name).faces)
                                for name in ("primary", "secondary")))
    assert counts[0] != counts[1], "fresh candidate topology was not exercised"
    assert not np.allclose(results[0].fluxes[BANDS[0]][:10], results[1].fluxes[BANDS[0]][:10])


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("unsupported", ["eccentric", "large_radius", "spots", "atmosphere", "tabulated_ld"])
def test_explicit_full_cpu_fallback(monkeypatch: pytest.MonkeyPatch, device: str, unsupported: str) -> None:
    from elisa import Observer

    require_device(device)
    binary = binary_fixture(eccentricity=0.1 if unsupported == "eccentric" else 0)
    if unsupported == "large_radius":
        binary = binary_fixture(q=1, potentials=(4, 4))
    elif unsupported == "spots":
        binary.primary.spots = [{"longitude": 45, "latitude": 60, "angular_radius": 10, "temperature_factor": 0.9}]
    elif unsupported == "atmosphere":
        binary.primary.atmosphere = "ck04"
    elif unsupported == "tabulated_ld":
        binary.primary.limb_darkening_coefficients = None
    engine = HybridBinaryLightCurve(BANDS, device=device, fallback="raise")
    with pytest.raises(UnsupportedHybridModelError):
        engine(binary, [0, 0.25])
    # Verify dispatch without requiring unavailable scientific atmosphere/LD
    # tables. Spot/eccentricity classification is distinct from hybrid validation.
    calls = []

    def cpu(self: Any, **kwargs: Any) -> tuple:
        calls.append((self, kwargs))
        return np.array([0, 0.25]), {band: np.array([0.8, 1.0]) for band in BANDS}

    monkeypatch.setattr(Observer, "lc", cpu)
    result = HybridBinaryLightCurve(BANDS, device=device)(binary, [0, 0.25], normalize=True)
    assert result.backend == "cpu_fallback"
    assert result.fallback_reason
    assert len(calls) == 1
    assert calls[0][1]["normalize"]
    assert result.component_fluxes == {}


def test_actual_eccentric_cpu_fallback_and_input_immutability() -> None:
    from elisa import Observer, settings

    binary = binary_fixture(eccentricity=0.1)
    original = deepcopy(binary)
    engine = HybridBinaryLightCurve(BANDS, device="cpu")
    phases = np.concatenate((np.linspace(0, 1, 25), [1.01, 0.01]))
    with settings.override(REFLECTION_EFFECT=False, NUMBER_OF_PROCESSES=1):
        result = engine(binary, phases, normalize=True)
        _, expected = Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=True)
    assert result.backend == "cpu_fallback"
    for band in BANDS:
        assert_allclose(result.fluxes[band], expected[band], rtol=0, atol=0)
    assert binary.primary.t_eff == original.primary.t_eff
    assert binary.primary.surface_potential == original.primary.surface_potential
    assert binary.additional_light == original.additional_light
    with pytest.raises(ValueError, match="phases"):
        engine(binary, [np.nan])
    with pytest.raises(ValueError, match="fallback"):
        HybridBinaryLightCurve(BANDS, device="cpu", fallback="silent")
