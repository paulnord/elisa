"""Changing GPU projection/clipping with the original CPU hull/visibility bridge."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_occultation import DeviceEclipseIntegrator  # noqa: E402

from .test_gpu_binary import BANDS, binary_fixture, contact_phases, require_device  # noqa: E402
from .test_gpu_likelihood import fp64  # noqa: E402, F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(("inclination", "potentials"), [(60, (8, 12)), (79, (8, 12)), (85, (8, 12)),
                                                       (90, (8, 12)), (90, (8, 100))])
def test_projection_coverage_component_and_face_parity(  # noqa: PLR0915 - staged scientific assertions
    device: str, inclination: float, potentials: tuple,
) -> None:
    from elisa import Observer, settings
    from elisa.base.curves.utils import calculate_surface_element_fluxes
    from elisa.binary_system import dynamic
    from elisa.binary_system.curves.c_router import prep_initial_system
    from elisa.binary_system.curves.utils import prep_surface_params
    from elisa.binary_system.surface.coverage import compute_surface_coverage, test_size_similarity
    from elisa.binary_system.utils import move_sys_onpos

    require_device(device)
    binary = binary_fixture(inclination=inclination, potentials=potentials)
    phases = np.concatenate((np.mod(contact_phases(binary)[::-1], 1), [0.75, 0.5, 0, 0.5]))
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear"):
        system = prep_initial_system(binary)
        observer = Observer(passband=BANDS, system=binary)
        prep_surface_params(system, return_values=False, write_to_containers=True, passband=observer.passband,
                            left_bandwidth=observer.left_bandwidth, right_bandwidth=observer.right_bandwidth)
    motion = binary.calculate_orbital_motion(phases, return_nparray=False)
    eclipse = dynamic.in_eclipse_test(np.array([p.azimuth for p in motion]), dynamic.get_eclipse_boundaries(binary, 1))
    kernel = DeviceEclipseIntegrator("linear", device=device)
    result = kernel(system, motion, eclipse, tuple(BANDS), diagnostics=True)
    assert result["details"]["partial_faces"] > 0 or inclination == 60
    if potentials[1] == 100:
        assert test_size_similarity(system.secondary, system.primary), "small-occulter visibility branch"
    stable_mask_differences = 0
    for index, position in enumerate(motion):
        cpu = move_sys_onpos(system, position, on_copy=True)
        compute_surface_coverage(cpu, binary.semi_major_axis, in_eclipse=bool(eclipse[index]),
                                 return_values=False, write_to_containers=True)
        actual = result["positions"][index]
        for comp, name in enumerate(("primary", "secondary")):
            a, b = getattr(actual, name), getattr(cpu, name)
            eps = 64 * np.finfo(float).eps
            for key in ("points", "normals", "los_cosines"):
                assert_allclose(getattr(a, key), getattr(b, key), rtol=3e-13, atol=eps)
            stable = (np.abs(a.los_cosines) > eps) & (np.abs(b.los_cosines) > eps)
            stable_mask_differences += np.count_nonzero((a.los_cosines[stable] > 0) != (b.los_cosines[stable] > 0))
            assert_allclose(a.coverage / binary.semi_major_axis**2, b.coverage / binary.semi_major_axis**2,
                            rtol=3e-13, atol=eps)
            for band in BANDS:
                cpu_faces = np.zeros(len(b.faces))
                cpu_faces[b.indices] = calculate_surface_element_fluxes(band, b)
                gpu_faces = np.zeros(len(a.faces))
                gpu_faces[a.indices] = calculate_surface_element_fluxes(band, a)
                total = cpu_faces.sum()
                if total:
                    assert_allclose(gpu_faces / total, cpu_faces / total, rtol=3e-13, atol=2e-12)
                else:
                    assert_allclose(gpu_faces, cpu_faces, rtol=0, atol=0)
                assert_allclose(result["components"][band][index, comp], total, rtol=3e-13, atol=0)
    assert stable_mask_differences == 0
    packet = result["details"]["prepared_batches"][0].block_until_ready()
    snapshot_total = np.asarray(packet.evaluate()["component_flux"])
    system.primary.points[:] = np.nan
    system.primary.normal_radiance[BANDS[0]][:] = 0
    with jax.transfer_guard("disallow_explicit"):
        repeated = jax.block_until_ready(packet.evaluate())
    assert np.asarray(repeated["row_valid"]).all()
    assert_allclose(repeated["component_flux"], snapshot_total, rtol=0, atol=0)
    assert repeated["coverage"].device.platform == device
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        packet.evaluate()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_explicit_coverage_choice_and_backend_fallback(device: str) -> None:
    from elisa import Observer, settings
    from elisa.experimental.gpu_binary import HybridBinaryLightCurve, UnsupportedHybridModelError

    require_device(device)
    binary = binary_fixture()
    phases = contact_phases(binary)
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear"):
        cpu_bridge = HybridBinaryLightCurve(BANDS, device=device, coverage="cpu", fallback="raise")
        device_bridge = HybridBinaryLightCurve(BANDS, device=device, coverage="device", fallback="raise")
        expected = cpu_bridge(binary, phases, normalize=True)
        actual = device_bridge(binary, phases, normalize=True)
        for band in BANDS:
            assert_allclose(actual.fluxes[band], expected.fluxes[band], rtol=3e-13, atol=2e-12)
        # PYPEX is a different numeric/backend contract; do not replace it with
        # Sutherland-Hodgman silently. The whole original CPU curve is used.
        with settings.override(INTERSECTION_BACKEND="pypex"):
            with pytest.raises(UnsupportedHybridModelError, match="intersection backends"):
                device_bridge(binary, phases)
            result = HybridBinaryLightCurve(BANDS, device=device)(binary, phases, normalize=True)
            _, reference = Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=True)
        assert result.backend == "cpu_fallback"
        for band in BANDS:
            assert_allclose(result.fluxes[band], reference[band], rtol=0, atol=0)
    with pytest.raises(ValueError, match="coverage"):
        HybridBinaryLightCurve(BANDS, device=device, coverage="stale")


def test_phase_resource_bound_uses_complete_cpu_fallback() -> None:
    from elisa import Observer, settings
    from elisa.experimental.gpu_binary import HybridBinaryLightCurve, UnsupportedHybridModelError

    binary = binary_fixture()
    phases = np.linspace(0, 1, 601)
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear"):
        strict = HybridBinaryLightCurve(BANDS, device="cpu", fallback="raise")
        with pytest.raises(UnsupportedHybridModelError, match="128 synthesis phases"):
            strict(binary, phases)
        result = HybridBinaryLightCurve(BANDS, device="cpu")(binary, phases, normalize=True)
        _, expected = Observer(passband=BANDS, system=binary).lc(phases=phases, normalize=True)
    assert result.backend == "cpu_fallback"
    assert "128 synthesis phases" in result.fallback_reason
    for band in BANDS:
        assert_allclose(result.fluxes[band], expected[band], rtol=0, atol=0)
