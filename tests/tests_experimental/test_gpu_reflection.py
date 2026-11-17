"""Original mutual irradiation, immutable packets and complete synthesis/fits."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_binary import HybridBinaryLightCurve, UnsupportedHybridModelError  # noqa: E402
from elisa.experimental.gpu_reflection import DeviceReflection, UnsupportedReflectionError  # noqa: E402
from scripts.validate_gpu_atmosphere import table_settings  # noqa: E402
from scripts.validate_gpu_fit import normalized_vector, prepare_fit  # noqa: E402
from scripts.validate_gpu_reflection import CASES, fixture, reflection_case, table_problem  # noqa: E402

from .test_gpu_binary import require_device  # noqa: E402
from .test_gpu_likelihood import fp64  # noqa: E402, F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_original_reflection_matrices_and_cooler_first_iterations(device: str, case: dict) -> None:
    from elisa import settings

    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW=case.get("law", "linear")):
        result = reflection_case(case, device)
    assert all(row["maximum_heating_K"] > 0 for row in result["comparisons"]["temperatures"])


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_snapshot_zero_albedo_and_iteration_progress(device: str) -> None:
    from elisa import settings
    from elisa.binary_system.curves.c_router import prep_initial_system
    from elisa.binary_system.surface.temperature import apply_reflection_effect

    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=False):
        initial = prep_initial_system(fixture(CASES[0]))
    kernel = DeviceReflection("linear", device=device)
    old = initial.primary.temperatures.copy()
    initial.primary.albedo = 0
    batch = kernel.prepare(initial, 4).block_until_ready()
    before = jax.tree.map(np.asarray, batch.evaluate())
    assert_allclose(before["temperatures"][0], old, rtol=0, atol=0)
    assert np.max(before["temperatures"][1]-initial.secondary.temperatures) > 0
    initial.primary.temperatures[:] = 1
    initial.secondary.areas[:] = 0
    with jax.transfer_guard("disallow_explicit"):
        repeated = jax.block_until_ready(batch.evaluate())
    assert repeated["temperatures"][0].device.platform == device
    assert_allclose(repeated["temperatures"][1], before["temperatures"][1], rtol=0, atol=0)
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        batch.evaluate()
    with table_settings(), settings.override(REFLECTION_EFFECT=False):
        initial = prep_initial_system(fixture(CASES[1]))
    values = []
    with settings.override(REFLECTION_EFFECT=True):
        for iterations in [0, 1, 2, 4, 32]:
            a = kernel.prepare(initial, iterations).evaluate()
            b = apply_reflection_effect(initial.copy(), 1, iterations)
            for i, name in enumerate(("primary", "secondary")):
                assert_allclose(a["temperatures"][i], getattr(b, name).temperatures, rtol=3e-13, atol=0)
            values.append(np.asarray(a["temperatures"][0]))
    assert all(np.all(b >= a) for a, b in pairwise(values))
    assert np.max(values[-1]-values[1]) > 0


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_device_reflection_feeds_emission_and_rebuilds_albedo(device: str) -> None:
    from elisa import Observer, settings

    require_device(device)
    bands = ["Generic.Bessell.V", "bolometric"]
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=3,
                                           LIMB_DARKENING_LAW="logarithmic"):
        binary = fixture(CASES[3])
        engine = HybridBinaryLightCurve(bands, device=device, fallback="raise")
        curves = []
        for albedo in [0.1, 0.8]:
            binary.primary.albedo = albedo
            a = engine(binary, [.5, .49, 0, .01, .25, 1.01], normalize=True, diagnostics=True)
            b = Observer(passband=bands, system=binary).lc(phases=[.5, .49, 0, .01, .25, 1.01], normalize=True)[1]
            assert a.backend == "hybrid"
            for band in bands:
                assert_allclose(a.fluxes[band], b[band], rtol=3e-13, atol=2e-12)
            assert a.diagnostics["radiance"]["temperature_source"] == "fresh post-reflection device temperatures"
            reflected = a.diagnostics["reflection"]["result"]
            radiance_packet = a.diagnostics["radiance"]["prepared_batch"]
            # The emission packet derives its base-patch temperatures directly
            # from the reflected device outputs; evaluation needs no transfers.
            with jax.transfer_guard("disallow_explicit"):
                jax.block_until_ready(radiance_packet.evaluate())
            assert reflected["temperatures"][0].device.platform == device
            curves.append(a.fluxes[bands[0]])
        assert not np.allclose(curves[0], curves[1], rtol=1e-8, atol=1e-8)
        previous = HybridBinaryLightCurve(bands, device=device, reflection="cpu", fallback="raise")
        cpu = previous(binary, [.5, .49, 0, .01, .25, 1.01], normalize=True, diagnostics=True)
        assert cpu.diagnostics["radiance"]["temperature_source"] == "CPU post-reflection upload"
        assert_allclose(cpu.fluxes[bands[0]], curves[-1], rtol=3e-13, atol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_reflection_resources_full_cpu_posterior_and_original_source_errors(
    device: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from elisa import settings
    from elisa.base.error import ElisaLimbDarkeningError
    from elisa.binary_system.curves.c_router import prep_initial_system

    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, PENALIZE_ALL_EXCEPTIONS=True):
        problem = table_problem(shape_parameters=True)
        cpu, gpu = prepare_fit(problem), prepare_fit(problem, device=device)
        vector = normalized_vector(cpu)
        monkeypatch.setattr("elisa.experimental.gpu_reflection._MAX_PAIRS", 1)
        actual = gpu.ln_probability(vector)
        assert np.isfinite(actual)
        assert_allclose(actual, cpu.ln_probability(vector), rtol=0, atol=0)
        assert gpu.hybrid_synthesis.last_evaluation["backend"] == "cpu_fallback"
        with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
            gpu.ln_probability(vector)
        binary = fixture(CASES[-1])
        with pytest.raises(UnsupportedHybridModelError, match="pairs"):
            HybridBinaryLightCurve(list(problem.data), device=device, fallback="raise")(binary, [0, .25])
    with table_settings(), settings.override(REFLECTION_EFFECT=False):
        initial = prep_initial_system(binary)
    with table_settings(), settings.override(LD_TABLES=str(tmp_path)), pytest.raises(ElisaLimbDarkeningError):
        DeviceReflection("linear", device=device).prepare(initial, 2)
    with pytest.raises(UnsupportedReflectionError, match="32"):
        DeviceReflection("linear", device=device).prepare(initial, 33)
    with pytest.raises(ValueError, match="reflection"):
        HybridBinaryLightCurve(["bolometric"], device=device, reflection="silent")
