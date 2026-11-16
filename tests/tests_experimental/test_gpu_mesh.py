"""Geometry/projection parity on actual ELISa Roche triangle meshes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_mesh import TriangleGeometry  # noqa: E402

from .test_gpu_likelihood import binary_fixture, fp64  # noqa: E402, F401


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_actual_binary_mesh_projection(device: str) -> None:
    from elisa import utils
    from elisa.base.surface.faces import calculate_normals, calculate_surface_centres
    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.binary_system.utils import move_sys_onpos

    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    binary = binary_fixture()
    positions = binary.calculate_orbital_motion(np.array([-0.1, 0, 0.01, 0.25, 0.49, 0.5, 0.75, 0.75]))
    initial = OrbitalPositionContainer.from_binary_system(binary, positions[0])
    initial.build()
    kernel = TriangleGeometry(device=device)
    for name, com in [("primary", 0.0), ("secondary", 1.0)]:
        star = getattr(initial, name)
        vertices = np.repeat(star.points[None], len(positions), axis=0)
        result = {
            k: np.asarray(v)
            for k, v in kernel(
                vertices,
                star.faces,
                np.full(len(positions), com),
                [p.azimuth for p in positions],
                np.full(len(positions), binary.inclination),
            ).items()
        }
        assert result["proposal_valid"].all()
        centres = calculate_surface_centres(star.points, star.faces)
        normals = calculate_normals(star.points, star.faces, centres, com)
        assert_allclose(result["centres"], np.broadcast_to(centres, result["centres"].shape), rtol=3e-13, atol=1e-15)
        # CPU caches centres BEFORE its later point correction. These outputs
        # intentionally describe supplied corrected vertices, not that cache.
        assert not np.allclose(centres, star.face_centres, rtol=1e-6, atol=1e-8)
        assert_allclose(
            result["areas"],
            np.broadcast_to(utils.triangle_areas(star.faces, star.points), result["areas"].shape),
            rtol=3e-13,
            atol=0,
        )
        assert_allclose(
            result["normals"],
            np.broadcast_to(normals, result["normals"].shape),
            rtol=3e-13,
            atol=64 * np.finfo(float).eps,
        )
        for i, pos in enumerate(positions):
            moved = getattr(move_sys_onpos(initial, pos), name)
            assert_allclose(result["observer_normals"][i], moved.normals, rtol=3e-13, atol=64 * np.finfo(float).eps)
            assert_allclose(result["los_cosines"][i], moved.los_cosines, rtol=3e-13, atol=64 * np.finfo(float).eps)
            stable = ~result["visibility_ambiguous"][i] & (np.abs(moved.los_cosines) > 64 * np.finfo(float).eps)
            assert np.array_equal(result["facing"][i, stable], moved.los_cosines[stable] > 0)


def test_degenerate_geometry_and_validation() -> None:
    kernel = TriangleGeometry(device="cpu")
    vertices = [[[1.0, 0, 0], [1.0, 1, 0], [1.0, 2, 0]]]
    result = kernel(vertices, [[0, 1, 2]], [0], [0], [np.pi / 2])
    assert not np.asarray(result["proposal_valid"]).any()
    assert np.isnan(result["normals"]).all()
    with pytest.raises(ValueError, match="index"):
        kernel(vertices, [[0, 1, 3]], [0], [0], [np.pi / 2])
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        kernel(vertices, [[0, 1, 2]], [0], [0], [np.pi / 2])


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_changed_roche_vertices_through_triangle_geometry(device: str, component: str) -> None:
    from scipy.spatial import ConvexHull

    from elisa import utils
    from elisa.base.surface.faces import calculate_normals, calculate_surface_centres
    from elisa.binary_system import model
    from elisa.binary_system.surface.mesh import get_surface_points
    from elisa.const import Position
    from elisa.experimental.gpu_roche import RocheRaySolver

    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    # Fixed spherical directions/topology: controlled geometry experiment,
    # not a substitute for ELISa candidate-specific angular mesh placement.
    i = np.arange(64)
    theta = np.arccos(1 - 2 * (i + 0.5) / 64)
    phi = (i * np.pi * (3 - np.sqrt(5))) % (2 * np.pi)
    unit = utils.spherical_to_cartesian(np.column_stack([np.ones(64), phi, theta]))
    faces = ConvexHull(unit).simplices
    q, spin, potential = [0.5, 1.0, 2.0], [0.5, 1.0, 1.5], [8.0, 10.0, 12.0]
    rays = RocheRaySolver(component, device=device)(q, spin, potential, phi, theta)
    assert np.asarray(rays["proposal_valid"]).all()
    vertices = np.array(rays["points_local"], copy=True)
    com = 0. if component == "primary" else 1.
    if component == "secondary":
        vertices[..., 0] = 1 - vertices[..., 0]
    inclination, azimuth = [0.0, np.pi / 2, np.pi], [0.1, 1.3, 3.2]
    result = {
        k: np.asarray(v)
        for k, v in TriangleGeometry(device=device)(vertices, faces, [com] * 3, azimuth, inclination).items()
    }
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
        centres = calculate_surface_centres(cpu_points, faces)
        normals = calculate_normals(cpu_points, faces, centres, com)
        observer_normals = utils.rotate_item(normals, Position(row, 1.0, azimuth[row], 0.0, 0.0), inclination[row])
        assert_allclose(result["areas"][row], utils.triangle_areas(faces, cpu_points), rtol=3e-10, atol=1e-15)
        assert_allclose(result["normals"][row], normals, rtol=3e-10, atol=1e-12)
        assert_allclose(result["los_cosines"][row], -observer_normals[:, 0], rtol=3e-10, atol=1e-12)
    assert not np.allclose(result["areas"][0], result["areas"][-1])


def test_nonfinite_intermediate_geometry_rejected() -> None:
    kernel = TriangleGeometry(device="cpu")
    result = kernel([[[1e308, 0, 0], [1e308, 1, 0], [1e308, 0, 1]]], [[0, 1, 2]], [0], [0], [np.pi / 2])
    assert not np.asarray(result["proposal_valid"]).any()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_limb_visibility_ambiguity(device: str) -> None:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    kernel = TriangleGeometry(device=device)
    vertices = np.repeat(np.array([[[1.0, 0, 0], [1.0, 1, 0], [1.0, 0, 1]]]), 3, axis=0)
    result = kernel(vertices, [[0, 1, 2]], [0.0] * 3, [-1e-12, 0.0, 1e-12], [np.pi / 2] * 3)
    assert np.array_equal(np.asarray(result["visibility_ambiguous"])[:, 0], [False, True, False])
    assert np.array_equal(np.asarray(result["facing"])[[0, 2], 0], [False, True])


def test_ambiguous_outward_orientation_and_winding() -> None:
    kernel = TriangleGeometry(device="cpu")
    points = [[[1e-20, 1.0, 0.0], [1e-20, 2.0, 0.0], [1e-20, 1.0, 1.0]]]
    for faces in ([[0, 1, 2]], [[0, 2, 1]]):
        result = kernel(points, faces, [0.0], [0.0], [np.pi / 2])
        assert np.asarray(result["orientation_ambiguous"]).all()
        assert not np.asarray(result["proposal_valid"]).any()
        assert np.isnan(result["normals"]).all()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_review_orientation_sign_flip_rejected(device: str) -> None:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    points = [
        [
            [0.08092567986876054, 0.7588066398912063, 0.08388129190687033],
            [0.15365026015703065, 0.6569075497919932, -0.48579604359095874],
            [0.5557769175439504, 0.42142712779741975, -0.8961860450648544],
        ],
    ]
    result = TriangleGeometry(device=device)(points, [[0, 1, 2]], [1.696679154131957], [0.0], [np.pi / 2])
    assert np.asarray(result["orientation_ambiguous"]).all()
    assert not np.asarray(result["proposal_valid"]).any()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_review_ill_conditioned_triangle_rejected(device: str) -> None:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")
    points = [
        [
            [-0.5453956084373934, 0.06473274754864833, 0.10963192489387019],
            [-0.5478983979144723, 0.24565918055451674, 0.2978366882406438],
            [-0.5478983979144633, 0.2456591805545179, 0.2978366882406416],
        ],
    ]
    result = TriangleGeometry(device=device)(points, [[0, 1, 2]], [0.0], [0.0], [np.pi / 2])
    assert np.asarray(result["triangle_ambiguous"]).all()
    assert not np.asarray(result["proposal_valid"]).any()


def test_large_coordinate_small_edges_unsupported() -> None:
    points = [[[1e8, 0.0, 0.0], [1e8, 1.0, 0.0], [1e8, 0.0, 1.0]]]
    result = TriangleGeometry(device="cpu")(points, [[0, 1, 2]], [0.0], [0.0], [np.pi / 2])
    assert np.asarray(result["coordinate_ambiguous"]).all()
    assert not np.asarray(result["proposal_valid"]).any()
