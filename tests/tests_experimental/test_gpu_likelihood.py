"""CPU-JAX tests are not GPU validation; hardware test explicitly skips without GPU."""

import copy
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_likelihood import FixedBinaryLikelihood  # noqa: E402 - optional JAX tests


@pytest.fixture(autouse=True)
def fp64() -> Iterator[None]:
    from elisa import settings

    original = {k: getattr(settings, k) for k in ("LIMB_DARKENING_LAW", "NUMBER_OF_PROCESSES")}
    try:
        with jax.enable_x64():
            yield
    finally:
        settings.configure(**original)


def pilot(device: str = "cpu") -> FixedBinaryLikelihood:
    return FixedBinaryLikelihood(
        [1, 0.6, 0.6, 0.9, 1],
        [1.1, 0.8, 0.85, 1, 1.15],
        [0.02] * 5,
        device=device,
        support_mean=0.875,
    )


def test_batched_cost_parity_and_bounds() -> None:
    model = pilot()
    theta = np.array([[0, -10], [0.2, -2], [0.9, 2], [0.95, 0], [-0.01, 0], [1, 0], [np.nan, 0]])
    assert_allclose(model(theta), model.numpy_reference(theta), rtol=2e-13, atol=2e-11)
    assert np.isneginf(model(theta)[-3:]).all()
    assert model.device.platform == "cpu"


def test_fp64_required() -> None:
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="FP64"):
        pilot()


@pytest.mark.parametrize(
    "change",
    [
        {"errors": [0] * 5},
        {"observed": [np.nan] * 5},
        {"baseline": [0] * 5},
        {"bounds": ((0, 1), (-10, 2))},
        {"device": "auto"},
        {"support_mean": 0},
    ],
)
def test_invalid_data(change: dict) -> None:
    args = {"baseline": [1] * 5, "observed": [1] * 5, "errors": [0.1] * 5, "device": "cpu"}
    args.update(change)
    with pytest.raises(ValueError, match=r"must|finite|bounds|device|mean|theta"):
        FixedBinaryLikelihood(**args)


def test_immutable_snapshot_and_shape() -> None:
    source = np.array([1.0, 0.8])
    model = FixedBinaryLikelihood(source, source, [0.1, 0.1], device="cpu")
    before = model([[0.2, 0]])
    source[:] = 99
    assert_allclose(before, model([[0.2, 0]]))
    with pytest.raises(ValueError, match=r"must|finite|bounds|device|mean|theta"):
        model([0.2, 0])
    with pytest.raises(ValueError, match=r"must|finite|bounds|device|mean|theta"):
        model(np.zeros((2, 3)))


def test_emcee_vectorized_matches_cpu_oracle() -> None:
    import emcee

    model = pilot()
    positions = np.random.default_rng(3).uniform([0.1, -3], [0.8, 1], size=(12, 2))
    cpu = emcee.EnsembleSampler(12, 2, model.numpy_reference, vectorize=True)
    gpu_api = model.make_sampler(12)
    state = np.random.RandomState(4).get_state()
    cpu.random_state = state
    gpu_api.random_state = state
    cpu.run_mcmc(positions, 50)
    gpu_api.run_mcmc(positions, 50)
    assert_allclose(cpu.get_chain(), gpu_api.get_chain(), rtol=0, atol=0)
    assert_allclose(cpu.get_log_prob(), gpu_api.get_log_prob(), rtol=2e-13, atol=2e-11)


def binary_fixture() -> Any:
    from elisa import BinarySystem, Star, settings

    settings.configure(LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1)
    common = {
        "mass": 1,
        "surface_potential": 5,
        "synchronicity": 1,
        "albedo": 0.5,
        "gravity_darkening": 0.32,
        "atmosphere": "bb",
        "limb_darkening_coefficients": {"bolometric": [0.5]},
        "discretization_factor": 10,
    }
    return BinarySystem(
        Star(t_eff=6500, **common),
        Star(t_eff=5500, **common),
        inclination=88,
        period=3,
        eccentricity=0,
        argument_of_periastron=90,
    )


@pytest.mark.parametrize("shift", [0.0, 0.1234567])
def test_real_elisa_direct_prediction_parity(shift: float) -> None:
    from elisa import Observer
    from elisa.analytics.binary_fit.lc_prediction import predict_light_curves
    from elisa.analytics.models.cost_fns import likelihood_fn

    binary = binary_fixture()
    binary.phase_shift = shift
    # Nonuniform, unordered, repeated observations spanning both eclipses and cycles.
    phases = np.array([0.5, 0, 0.015, 0.025, 0.05, 0.25, 0.49, 0.51, 1.0, 0, 0, -0.01, 1e16, 1e16 + 2])
    support, inverse = np.unique(phases, return_inverse=True)
    obs, err = np.ones(phases.size), np.full(phases.size, 0.001)
    model = FixedBinaryLikelihood.from_binary(
        binary,
        phases,
        "bolometric",
        obs,
        err,
        device="cpu",
        bounds=((0, 0.9999), (-1000, 1000)),
    )
    assert binary.additional_light == 0
    for l3, ln_f in [(0, -1000), (0.2, -2), (0.9, 2), (0.9999, 1000)]:
        # Run real Observer through the actual fitter mapping/normalization layer.
        def synth(xs: Any, _discretization: Any, _observer: Any, _l3: float = l3, **_params: Any) -> dict:
            b = copy.deepcopy(binary)
            b.additional_light = _l3
            return Observer(passband=["bolometric"], system=b).lc(phases=xs, normalize=True)[1]

        prediction = predict_light_curves(
            support,
            {"bolometric": obs},
            {"bolometric": inverse},
            10,
            None,
            1000,
            {},
            synthesis=synth,
        )
        assert_allclose(model.predict([[l3, ln_f]])[0], prediction["bolometric"], rtol=3e-13, atol=3e-13)
        expected = likelihood_fn({"bolometric": obs}, {"bolometric": err}, prediction, ln_f)
        assert_allclose(model([[l3, ln_f]])[0], expected, rtol=3e-12, atol=1e-8)
    binary.eccentricity = 0.1
    with pytest.raises(ValueError, match="circular"):
        FixedBinaryLikelihood.from_binary(binary, phases, "bolometric", obs, err, device="cpu")


def test_gpu_or_explicit_unavailable() -> None:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        with pytest.raises(RuntimeError, match="no CPU fallback"):
            pilot("gpu")
        pytest.skip("No GPU available: real GPU execution not validated")
    model = pilot("gpu")
    assert model.device.platform == "gpu"
    theta = np.array([[0, -10], [0.2, -2], [0.9, 2]])
    assert_allclose(model(theta), model.numpy_reference(theta), rtol=1e-11, atol=1e-9)


def test_reject_overflowed_normalization() -> None:
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="positive mean"):
        FixedBinaryLikelihood([1e308, 1e308], [1, 1], [0.1, 0.1], support_mean=1e308, device="cpu")
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="mean ratio"):
        FixedBinaryLikelihood([1e-300, 1e-300], [1, 1], [0.1, 0.1], support_mean=1e300, device="cpu")


def test_late_precision_change_rejected() -> None:
    model = pilot()
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain enabled"):
        model([[0.2, 0]])


def test_snapshot_rebinding_rejected() -> None:
    model = pilot()
    with pytest.raises(AttributeError, match="immutable snapshot"):
        model.mean_ratio = 2
    with pytest.raises(AttributeError, match="immutable snapshot"):
        model.bounds = np.array([[0,.5],[-3,2]])
