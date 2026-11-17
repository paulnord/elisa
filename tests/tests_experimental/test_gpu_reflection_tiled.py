"""Fine complete curves, independent CPU primitive tiles, parity and resources."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, settings
from elisa.binary_system.surface.temperature import apply_reflection_effect
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_protocol import MeshResolutionCheck, ProtocolDetachedMCMCFit, ResolutionNotQualifiedError
from elisa.experimental.gpu_reflection_tiled import TiledDeviceReflection
from scripts.fine_mesh_reference import cpu_reflect, intrinsic, observe
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, normalized_vector, physical, prepare_fit
from scripts.validate_gpu_protocol import protocol_problem
from scripts.validate_gpu_reflection import CASES, fixture

from .test_gpu_binary import require_device
from .test_gpu_likelihood import fp64  # noqa: F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_tiled_irradiation_original_cpu_primitives_and_iteration_order(device: str, case: dict) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW=case.get("law", "linear")):
        initial = intrinsic(fixture(case))
        unchanged = initial.primary.temperatures.copy()
        packet = TiledDeviceReflection(settings.LIMB_DARKENING_LAW, device=device, block_size=17).prepare(
            initial, case["iterations"],
        )
        actual = packet.evaluate()
        expected = apply_reflection_effect(initial.copy(), 1., case["iterations"])
        independent = cpu_reflect(initial.copy(), case["iterations"], block_size=11)
        assert bool(actual["proposal_valid"])
        for i, name in enumerate(("primary", "secondary")):
            assert_allclose(actual["temperatures"][i], getattr(expected, name).temperatures, rtol=3e-13, atol=0)
            assert_allclose(getattr(independent, name).temperatures, getattr(expected, name).temperatures,
                            rtol=3e-13, atol=0)
        assert_allclose(initial.primary.temperatures, unchanged, rtol=0, atol=0)
        assert "coupling" not in actual  # full quadratic diagnostics are deliberately not materialized


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_tiled_snapshot_zero_albedo_no_transfer_and_nonfinite_mask(device: str) -> None:
    import jax

    require_device(device)
    with table_settings():
        initial = intrinsic(fixture(CASES[0]))
        initial.primary.albedo = 0
        kernel = TiledDeviceReflection("linear", device=device, block_size=8)
        for iterations in [0, 2]:
            packet = kernel.prepare(initial, iterations).block_until_ready()
            actual = packet.evaluate()
            assert_allclose(actual["temperatures"][0], initial.primary.temperatures, rtol=0, atol=0)
            with jax.transfer_guard("disallow_explicit"):
                repeated = jax.block_until_ready(packet.evaluate())
            assert repeated["temperatures"][0].device.platform == device
        initial.primary.temperatures[:] = 1
        assert_allclose(packet.evaluate()["temperatures"][0], actual["temperatures"][0], rtol=0, atol=0)
        with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
            packet.evaluate()
    for block in [0, 65, True]:
        with pytest.raises(ValueError, match="tile size"):
            TiledDeviceReflection("linear", device=device, block_size=block)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_fine_complete_reflected_curves_match_original_and_independent_reference(device: str) -> None:
    from scripts.validate_gpu_binary import fixture as bb_fixture

    require_device(device)
    phases = np.array([.5, .49, .515, .25, 0, .01, .025, 1.01, -.5])+.017
    bands = ["Generic.Bessell.V", "bolometric"]
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        binary = bb_fixture({"inclination": 85, "discretization": 2})
        observer = Observer(passband=bands, system=binary)
        original = observer.lc(phases=phases, normalize=True)[1]
        independent = observe(observer, phases)[1]
        engine = HybridBinaryLightCurve(bands, device=device, reflection="tiled", coverage="cpu", fallback="raise")
        result = engine(binary, phases, normalize=True)
        assert result.backend == "hybrid"
        for band in bands:
            assert_allclose(result.fluxes[band], original[band], rtol=3e-13, atol=2e-12)
            assert_allclose(independent[band], original[band], rtol=3e-13, atol=2e-12)
        # Finer finite reference uses independent original CPU primitives without
        # allocating the dense original reflection matrices. It is not continuum truth.
        finer = bb_fixture({"inclination": 90, "discretization": 1.5})
        expected = observe(Observer(passband=bands, system=finer), phases)[1]
        actual = engine(finer, phases, normalize=True)
        for band in bands:
            assert_allclose(actual.fluxes[band], expected[band], rtol=3e-13, atol=2e-12)


def test_paired_component_meshes_and_finer_accuracy_failure_remain_explicit() -> None:
    from elisa.analytics.models import lc

    with table_settings(), settings.override(PENALIZE_ALL_EXCEPTIONS=True):
        problem = make_problem()
        problem.setup["discretization"] = 2.5
        native = prepare_fit(problem, phase_mode="observations")
        paired = prepare_fit(problem, phase_mode="observations", mesh_mode="paired")
        vector = normalized_vector(paired)
        props = physical(paired, vector)
        original = deepcopy(props)
        native.predict_light_curves(props)
        expected = lc.prepare_binary(discretization=2.5, **props)
        assert_allclose(native.observer._system.secondary.discretization_factor,
                        expected.secondary.discretization_factor, rtol=0, atol=0)
        assert native.observer._system.secondary.discretization_factor > np.radians(2.5)
        paired.predict_light_curves(props)
        assert_allclose(np.degrees(paired.observer._system.secondary.discretization_factor), 2.5, rtol=0, atol=1e-14)
        assert props == original
        with pytest.raises(ValueError, match="paired mesh"):
            ProtocolDetachedMCMCFit(phase_mode="observations", resolution_check=MeshResolutionCheck((2.5, 2)))
        problem.setup["discretization"] = 3
        strict = prepare_fit(problem, phase_mode="observations", mesh_mode="paired",
                             resolution_check=MeshResolutionCheck((2.5, 2)))
        with pytest.raises(ResolutionNotQualifiedError) as caught:
            strict.ln_probability(normalized_vector(strict))
        assert caught.value.report["mesh_mode"] == "paired"
        assert strict._candidate_failure_diagnostics.total == 0
        assert not strict._protocol_active


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_tiled_resource_failure_retains_complete_cpu_fallback(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        problem = protocol_problem()
        fit = prepare_fit(problem, device=device, phase_mode="observations", mesh_mode="paired", reflection="tiled",
                          coverage="cpu")
        original = prepare_fit(problem, phase_mode="observations", mesh_mode="paired")
        monkeypatch.setattr("elisa.experimental.gpu_reflection_tiled._MAX_FACES", 1)
        vector = normalized_vector(fit)
        assert_allclose(fit.ln_probability(vector), original.ln_probability(vector), rtol=0, atol=0)
        assert fit.hybrid_synthesis.last_evaluation["backend"] == "cpu_fallback"
