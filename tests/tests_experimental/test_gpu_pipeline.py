"""Resident pipeline parity, unsupported-state propagation and transfer guards."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial import ConvexHull

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_gravity import RocheSurfaceGravity  # noqa: E402
from elisa.experimental.gpu_mesh import TriangleGeometry  # noqa: E402
from elisa.experimental.gpu_pipeline import RocheGeometryPipeline  # noqa: E402
from elisa.experimental.gpu_roche import RocheRaySolver  # noqa: E402


def fixture(device: str, component: str = "primary") -> tuple:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    i = np.arange(64)
    theta = np.arccos(1 - 2 * (i + 0.5) / 64)
    phi = (i * np.pi * (3 - np.sqrt(5))) % (2 * np.pi)
    unit = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    faces = ConvexHull(unit).simplices
    parameters = tuple(
        np.array(x, dtype=float)
        for x in (
            [0.5, 1.0, 2.0],
            [0.5, 1.0, 1.5],
            [8.0, 10.0, 12.0],
            [1.1e30, 2e30, 2.5e30],
            [3e9, 5e9, 8e9],
            [-0.1, 0.0, 2 * np.pi],
            [0.0, np.pi / 2, np.pi],
        )
    )
    return RocheGeometryPipeline(component, phi, theta, faces, device=device), parameters, phi, theta, faces


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_pipeline_matches_stages_and_original_cpu(component: str, device: str) -> None:
    from elisa import const, utils
    from elisa.base.surface.faces import calculate_normals, calculate_surface_centres
    from elisa.binary_system import model
    from elisa.binary_system.surface.gravity import calculate_potential_gradient
    from elisa.binary_system.surface.mesh import get_surface_points
    from elisa.const import Position

    pipeline, params, phi, theta, faces = fixture(device, component)
    q, spin, potential, mass, axis, az, inc = params
    result = jax.tree.map(np.asarray, pipeline(*params))
    assert result["proposal_valid"].all()
    ray = RocheRaySolver(component, device=device)(q, spin, potential, phi, theta)
    points = np.array(ray["points_local"], copy=True)
    com = 0.0 if component == "primary" else 1.0
    if component == "secondary":
        points[..., 0] = 1 - points[..., 0]
    geometry = TriangleGeometry(device=device)(points, faces, [com] * 3, az, inc)
    gravity = RocheSurfaceGravity(component, device=device)(points, faces, q, spin, mass, axis)
    for expected, actual in [(ray, result["rays"]), (geometry, result["geometry"]), (gravity, result["gravity"])]:
        for key, value in expected.items():
            reference = np.asarray(value)
            if reference.dtype.kind in "biu":
                assert np.array_equal(actual[key], reference)
            else:
                assert_allclose(actual[key], reference, rtol=3e-13, atol=64 * np.finfo(float).eps)
    for row in range(3):
        cpu_points = get_surface_points(
            phi,
            theta,
            0.1,
            1.0,
            getattr(model, f"pre_calculate_for_potential_value_{component}"),
            getattr(model, f"potential_{component}_fn"),
            getattr(model, f"radial_{component}_potential_derivative"),
            potential[row],
            q[row],
            spin[row],
        )
        if component == "secondary":
            cpu_points[:, 0] = 1 - cpu_points[:, 0]
        normals = calculate_normals(cpu_points, faces, calculate_surface_centres(cpu_points, faces), com)
        observer = utils.rotate_item(normals, Position(row, 1.0, az[row], 0.0, 0.0), inc[row])
        gradient = calculate_potential_gradient(1.0, component, cpu_points, spin[row], q[row])
        face_gravity = np.mean(np.linalg.norm((const.G * mass[row] / axis[row] ** 2) * gradient, axis=1)[faces], axis=1)
        assert_allclose(result["points_binary"][row], cpu_points, rtol=3e-10, atol=1e-13)
        assert_allclose(
            result["geometry"]["areas"][row],
            utils.triangle_areas(faces, cpu_points),
            rtol=3e-10,
            atol=1e-15,
        )
        assert_allclose(result["geometry"]["normals"][row], normals, rtol=3e-10, atol=1e-12)
        assert_allclose(result["geometry"]["los_cosines"][row], -observer[:, 0], rtol=3e-10, atol=1e-12)
        assert_allclose(result["gravity"]["face_gravity"][row], face_gravity, rtol=3e-10, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_prepared_evaluation_has_no_host_transfers(device: str) -> None:
    pipeline, params, *_ = fixture(device)
    batch = pipeline.prepare(*params).block_until_ready()
    jax.block_until_ready(batch.evaluate())  # compile before the strict runtime guard
    with jax.transfer_guard("disallow_explicit"):
        for _ in range(2):
            result = jax.block_until_ready(batch.evaluate())
    assert np.asarray(result["proposal_valid"]).all()
    assert result["points_binary"].device.platform == device


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_prepared_snapshot_and_late_fp64_change(device: str) -> None:
    pipeline, params, phi, theta, faces = fixture(device)
    batch = pipeline.prepare(*params).block_until_ready()
    reference = np.asarray(batch.evaluate()["rays"]["radii"])
    params[0][:] = 0.9
    phi[:], theta[:], faces[:] = 0, 0, 0
    assert_allclose(batch.evaluate()["rays"]["radii"], reference, rtol=0, atol=0)
    assert batch.nbytes == 3 * 7 * 8
    assert batch.proposals == 3
    assert pipeline.topology_nbytes > 0
    with pytest.raises(FrozenInstanceError):
        batch._parameters = ()
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        batch.evaluate()
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        pipeline.prepare(*params)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_pipeline_propagates_unsupported_rays_and_gravity(device: str) -> None:
    pipeline, params, *_ = fixture(device)
    values = list(params)
    values[2] = np.array([1.0, 1e9, 12.0])
    result = jax.tree.map(np.asarray, pipeline(*values))
    assert np.array_equal(result["proposal_valid"], [False, False, True])
    assert np.isnan(result["rays"]["radii"][:2]).all()
    assert np.isnan(result["gravity"]["face_gravity"][:2]).all()
    values = list(params)
    values[4] = np.full(3, 1e-300)
    result = jax.tree.map(np.asarray, pipeline(*values))
    assert result["rays"]["proposal_valid"].all()
    assert result["geometry"]["proposal_valid"].all()
    assert not result["proposal_valid"].any()
    assert (result["gravity"]["status"] == 2).all()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_pipeline_preserves_geometry_conditioning_gates(device: str) -> None:
    _, params, *_ = fixture(device)
    # Three identical directions make a degenerate triangle, although all rays
    # and vertex gravity values are individually supported.
    pipeline = RocheGeometryPipeline("primary", [0.0] * 3, [np.pi / 2] * 3, [[0, 1, 2]], device=device)
    result = jax.tree.map(np.asarray, pipeline(*params))
    assert result["rays"]["proposal_valid"].all()
    assert result["gravity"]["proposal_valid"].all()
    assert not result["geometry"]["proposal_valid"].any()
    assert not result["proposal_valid"].any()


def test_pipeline_input_validation() -> None:
    pipeline, params, phi, theta, faces = fixture("cpu")
    for index, value, message in [
        (0, [0.1] * 3, "Unsupported"),
        (1, [2.0] * 3, "Unsupported"),
        (2, [-1.0] * 3, "positive"),
        (3, [0.0] * 3, "positive"),
        (4, [np.inf] * 3, "finite"),
        (5, [10.0] * 3, "Azimuth"),
        (6, [-1.0] * 3, "inclination"),
        (0, [1.0], "matching"),
    ]:
        values = list(params)
        values[index] = value
        with pytest.raises(ValueError, match=message):
            pipeline.prepare(*values)
    for ph, th, ff, message in [
        ([], [], faces, "Directions"),
        (phi, theta[:-1], faces, "Directions"),
        (phi - 10, theta, faces, "Angles"),
        (phi, theta, [[0.0, 1.0, 2.0]], "integer"),
        (phi, theta, [[0, 0, 1]], "index"),
        (phi, theta, [[0, 1, 999]], "index"),
    ]:
        with pytest.raises(ValueError, match=message):
            RocheGeometryPipeline("primary", ph, th, ff, device="cpu")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_large_batch_projection_preserves_stage_rounding(component: str, device: str) -> None:
    """Regression: unrestricted fusion exceeded the original near-limb tolerance."""
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    i = np.arange(256)
    theta = np.arccos(1 - 2 * (i + 0.5) / 256)
    phi = (i * np.pi * (3 - np.sqrt(5))) % (2 * np.pi)
    unit = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    faces = ConvexHull(unit).simplices
    t = np.linspace(0, 1, 128)
    q, spin, potential = 0.5 + 1.5 * t, 0.5 + t, 8 + 6 * t
    az, inc = -np.pi + 2 * np.pi * t, 0.2 + (np.pi - 0.4) * t
    pipeline = RocheGeometryPipeline(component, phi, theta, faces, device=device)
    actual = pipeline(q, spin, potential, (1 + t) * 1e30, (3 + 3 * t) * 1e9, az, inc)
    ray = RocheRaySolver(component, device=device)(q, spin, potential, phi, theta)
    points = np.array(ray["points_local"], copy=True)
    centre = 0.0 if component == "primary" else 1.0
    if component == "secondary":
        points[..., 0] = 1 - points[..., 0]
    expected = TriangleGeometry(device=device)(points, faces, np.full(128, centre), az, inc)
    assert np.asarray(actual["proposal_valid"]).all()
    for key in ["normals", "observer_normals", "los_cosines"]:
        assert_allclose(actual["geometry"][key], expected[key], rtol=3e-13, atol=64 * np.finfo(float).eps)
