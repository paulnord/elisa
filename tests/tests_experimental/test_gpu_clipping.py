"""Analytic/exact areas, real eclipse fixtures and strict device snapshots."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_clipping import ConvexTriangleClipper  # noqa: E402

from .test_gpu_likelihood import fp64  # noqa: E402, F401


def require_device(device: str) -> None:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_known_exact_areas_winding_and_contacts(device: str) -> None:
    from tests.tests_binary.test_convex_clipping import KNOWN_AREA_CASES

    require_device(device)
    cases = KNOWN_AREA_CASES
    capacity = max(len(case.polygon) for case in cases)
    polygons = np.zeros((len(cases), capacity, 2))
    triangles = np.stack([case.triangle for case in cases])[:, None]
    counts = np.array([len(case.polygon) for case in cases])
    for i, case in enumerate(cases):
        polygons[i, :counts[i]] = case.polygon
    kernel = ConvexTriangleClipper(device=device)
    for reverse in (False, True):
        current = polygons.copy()
        if reverse:
            for i, count in enumerate(counts):
                current[i, :count] = current[i, :count][::-1]
        result = kernel.prepare(current, counts, triangles[:, :, ::-1] if reverse else triangles).evaluate()
        assert np.asarray(result["row_valid"]).all()
        expected = np.array([float(case.expected) for case in cases])
        assert_allclose(result["areas"][:, 0], expected, rtol=3e-13, atol=0)
        assert np.all(np.asarray(result["areas"][:, 0])[expected > 0] > 0), "positive slivers must not be erased"


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_real_saved_eclipse_intersections(device: str) -> None:
    from elisa.binary_system.surface.clipping import sutherland_hodgman_intersection_areas

    require_device(device)
    path = Path(__file__).resolve().parents[1] / "data/clipping/eclipse_geometry.npz"
    with np.load(path, allow_pickle=False) as data:
        size = int(data["batches"])
        counts = np.array([len(data[f"h{i}"]) for i in range(size)])
        polygons = np.zeros((size, max(counts), 2))
        width = max(len(data[f"t{i}"]) for i in range(size))
        triangles = np.zeros((size, width, 3, 2))
        expected = np.zeros((size, width))
        for i in range(size):
            h, t = data[f"h{i}"], data[f"t{i}"]
            polygons[i, :len(h)] = h
            triangles[i, :len(t)] = t
            expected[i, :len(t)] = sutherland_hodgman_intersection_areas(h, t)
    result = ConvexTriangleClipper(device=device).prepare(polygons, counts, triangles).evaluate()
    assert np.asarray(result["row_valid"]).all()
    # Original geometry tolerance: relative3e-13 and absolute64eps in SMA².
    assert_allclose(result["areas"], expected, rtol=3e-13, atol=64 * np.finfo(float).eps)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_clipping_snapshot_empty_degenerate_and_transfer_guard(device: str) -> None:
    require_device(device)
    polygon = np.array([[[0., 0.], [1., 0.], [0., 1.]]])
    triangles = polygon[:, None].copy()
    kernel = ConvexTriangleClipper(device=device)
    batch = kernel.prepare(polygon, [3], triangles).block_until_ready()
    reference = np.asarray(batch.evaluate()["areas"])
    polygon[:], triangles[:] = np.nan, np.nan
    with jax.transfer_guard("disallow_explicit"):
        result = jax.block_until_ready(batch.evaluate())
    assert_allclose(result["areas"], reference, rtol=0, atol=0)
    assert result["areas"].device.platform == device
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        batch.evaluate()
    p = np.zeros((2, 4, 2))
    t = np.zeros((2, 1, 3, 2))
    assert_allclose(kernel.prepare(p, [0, 4], t).evaluate()["areas"], 0, rtol=0, atol=0)
    with pytest.raises(ValueError, match="finite"):
        kernel.prepare(polygon, [3], triangles)
    with pytest.raises(ValueError, match="counts"):
        kernel.prepare(p, [5, 4], t)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_clipping_numeric_overflow_is_unsupported(device: str) -> None:
    require_device(device)
    huge = np.array([[[1e308, 1e308], [-1e308, 1e308], [1e308, -1e308]]])
    result = ConvexTriangleClipper(device=device).prepare(huge, [3], huge[:, None]).evaluate()
    assert not np.asarray(result["row_valid"]).any()
    assert (np.asarray(result["status"]) == 2).all()
    assert np.isnan(result["areas"]).all()
