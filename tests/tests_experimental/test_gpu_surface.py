"""Actual Roche-container and facewise CPU parity for the surface kernel."""

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator  # noqa: E402

from .test_gpu_likelihood import binary_fixture, fp64  # noqa: E402, F401


@pytest.mark.parametrize("law", ["linear", "cosine", "logarithmic", "square_root"])
def test_face_edges(law: str) -> None:
    from elisa import ld

    mu = np.array([[-1, 0, 1e-300, 0.1, 0.6, 1.0]])
    r = np.array([[2, 3, 4, 5, 6, 7.0]])
    cov = np.array([[1, 1, 0.3, 0, 1, 0.5]])
    c = np.full((*mu.shape, 1 if law in ("linear", "cosine") else 2), 0.3)
    expected = r[0] * (
        (mu[0] * cov[0]) * ld.limb_darkening_factor(cos_theta=mu[0], coefficients=c[0], limb_darkening_law=law)
    )
    faces, total = SurfaceFluxIntegrator(law, device="cpu")(r, mu, cov, c)
    assert_allclose(faces[0], expected, rtol=2e-14, atol=1e-15)
    assert_allclose(total, [expected.sum()], rtol=2e-14)


@pytest.mark.parametrize("law", ["linear", "logarithmic", "square_root"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_actual_roche_surfaces(monkeypatch: pytest.MonkeyPatch, law: str, device: str) -> None:
    from elisa import Observer, settings
    from elisa.base.curves import utils

    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    binary = binary_fixture()
    settings.configure(LIMB_DARKENING_LAW=law)
    coeff = [0.5] if law == "linear" else [0.3, 0.2]
    for star in (binary.primary, binary.secondary):
        star.limb_darkening_coefficients = {"bolometric": coeff}
    captured, expected = [], []
    original = utils.flux_from_star_container

    def capture(band: str, star: Any) -> float:
        captured.append(copy.deepcopy(star))
        expected.append(utils.calculate_surface_element_fluxes(band, star))
        return original(band, star)

    monkeypatch.setattr(utils, "flux_from_star_container", capture)
    # Include ingress/egress, eclipse centers, quadrature and unequal phases.
    phases = np.array([0, 0.01, 0.025, 0.04, 0.06, 0.1, 0.25, 0.45, 0.48, 0.5, 0.51, 0.54, 0.75])
    _, curve = Observer(passband=["bolometric"], system=binary).lc(phases=phases, normalize=False)
    assert captured
    assert np.isfinite(curve["bolometric"]).all()
    coverage = np.concatenate([star.coverage[star.indices] for star in captured])
    assert np.any(coverage == 0)
    assert np.any(coverage > 0)
    fraction = np.concatenate([
        star.coverage[star.indices] / (star.areas[star.indices] * binary.semi_major_axis**2)
        for star in captured
    ])
    assert np.any((fraction > .01) & (fraction < .99))
    assert np.any(fraction > .999)
    kernel = SurfaceFluxIntegrator(law, device=device)
    faces, totals = map(np.asarray, kernel.from_containers("bolometric", captured))
    for i, reference in enumerate(expected):
        assert_allclose(faces[i, : len(reference)], reference, rtol=3e-13, atol=1e-10)
        assert np.all(faces[i, len(reference) :] == 0)
    assert_allclose(totals, [x.sum() for x in expected], rtol=3e-13)


def test_empty_invalid_and_gpu() -> None:
    empty = SimpleNamespace(
        indices=np.array([], dtype=int),
        normal_radiance={"b": np.array([])},
        los_cosines=np.array([]),
        coverage=np.array([]),
        ld_cfs={"b": np.empty((0, 1))},
    )
    faces, totals = SurfaceFluxIntegrator("linear", device="cpu").from_containers("b", [empty])
    assert faces.shape == (1, 0)
    assert_allclose(totals, [0])
    with pytest.raises(ValueError, match="Unsupported"):
        SurfaceFluxIntegrator("wrong", device="cpu")
    kernel = SurfaceFluxIntegrator("linear", device="cpu")
    with pytest.raises(ValueError, match="finite"):
        kernel([[1]], [[np.nan]], [[1]], [[[0.5]]])
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        kernel([[1]], [[1]], [[1]], [[[0.5]]])
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        with pytest.raises(RuntimeError, match="no CPU fallback"):
            SurfaceFluxIntegrator("linear", device="gpu")
        pytest.skip("No GPU hardware available")
    faces, total = SurfaceFluxIntegrator("linear", device="gpu")([[2]], [[0.5]], [[0.3]], [[[0.5]]])
    assert_allclose(faces, [[0.225]], rtol=1e-13)
    assert_allclose(total, [0.225], rtol=1e-13)


@pytest.mark.parametrize("law", ["linear", "cosine"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_linear_limb_cancellation(law: str, device: str) -> None:
    from elisa import ld

    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    mu = np.array([[1e-12, 1e-16, 1e-17, 1e-18]])
    radiance, coverage = np.full_like(mu, 1e7), np.full_like(mu, 1e18)
    coefficients = np.ones((*mu.shape, 1))
    expected = radiance[0] * ((mu[0] * coverage[0]) * ld.limb_darkening_factor(
        cos_theta=mu[0], coefficients=coefficients[0], limb_darkening_law=law,
    ))
    faces, _ = SurfaceFluxIntegrator(law, device=device)(radiance, mu, coverage, coefficients)
    assert_allclose(faces[0], expected, rtol=3e-14, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_prepared_device_snapshot(device: str) -> None:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    kernel = SurfaceFluxIntegrator("linear", device=device)
    radiance = np.array([[2.]])
    batch = kernel.prepare(radiance, [[.5]], [[.3]], [[[.5]]])
    radiance[0, 0] = 9
    assert batch.shape == (1, 1)
    assert batch.nbytes == 32
    for _ in range(2):
        faces, totals = batch.evaluate()
        assert_allclose(faces, [[.225]], rtol=1e-14)
        assert_allclose(totals, [.225], rtol=1e-14)
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        batch.evaluate()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_fixed_phase_padding_preserves_selected_face_flux(device: str) -> None:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    star = SimpleNamespace(indices=np.array([0]), normal_radiance={"b": np.array([2.0])},
                           los_cosines=np.array([0.5]), coverage=np.array([0.3]),
                           ld_cfs={"b": np.array([[0.5]])})
    kernel = SurfaceFluxIntegrator("linear", device=device)
    selected, total = kernel.from_containers("b", [star])
    padded, padded_total = kernel.from_containers("b", [star], pad_to=4)
    assert padded.shape == (1, 4)
    assert_allclose(padded[:, :1], selected, rtol=0, atol=0)
    assert_allclose(padded[:, 1:], 0, rtol=0, atol=0)
    assert_allclose(padded_total, total, rtol=3e-13, atol=0)
    with pytest.raises(ValueError, match="pad_to"):
        kernel.from_containers("b", [star], pad_to=0)


@pytest.mark.parametrize("option", ["--max-input-mib", "--discretization"])
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_benchmark_rejects_nonfinite_limits(option: str, value: str) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_gpu_surface.py"
    result = subprocess.run(  # noqa: S603 - fixed repository script and parametrized literals
        [sys.executable, str(script), "--device", "cpu", option, value],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "must be finite" in result.stderr
