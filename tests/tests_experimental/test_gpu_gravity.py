"""Changing Roche gravity against ELISa and independent potential/force checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial import ConvexHull

jax = pytest.importorskip("jax")
from elisa.experimental.gpu_gravity import RocheSurfaceGravity  # noqa: E402
from elisa.experimental.gpu_roche import RocheRaySolver  # noqa: E402

from .test_gpu_likelihood import fp64  # noqa: E402, F401


def hardware(device: str) -> None:
    try:
        jax.devices(device)
    except RuntimeError:
        pytest.skip("Requested hardware unavailable")


def vertices(component: str) -> np.ndarray:
    points = np.array([[0.1, 0.02, 0.03], [0.04, 0.12, -0.03], [-0.06, 0.03, 0.14]])
    if component == "secondary":
        points[:, 0] = 1 - points[:, 0]
    return points


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_changing_roche_gravity(component: str, device: str) -> None:
    from elisa import const, utils
    from elisa.binary_system.surface.gravity import calculate_potential_gradient

    hardware(device)
    i = np.arange(64)
    theta = np.arccos(1 - 2 * (i + 0.5) / 64)
    phi = (i * np.pi * (3 - np.sqrt(5))) % (2 * np.pi)
    unit = utils.spherical_to_cartesian(np.column_stack([np.ones(64), phi, theta]))
    faces = ConvexHull(unit).simplices
    q, spin, potential = np.array([0.5, 1.0, 2.0]), np.array([0.5, 1.0, 1.5]), [8.0, 10.0, 12.0]
    rays = RocheRaySolver(component, device=device)(q, spin, potential, phi, theta)
    assert np.asarray(rays["proposal_valid"]).all()
    points = np.array(rays["points_local"], copy=True)
    if component == "secondary":
        points[..., 0] = 1 - points[..., 0]
    mass, axis = np.array([1.1e30, 2e30, 2.5e30]), np.array([3e9, 5e9, 8e9])
    kernel = RocheSurfaceGravity(component, device=device)
    result = {k: np.asarray(v) for k, v in kernel(points, faces, q, spin, mass, axis).items()}
    assert result["proposal_valid"].all()
    assert (result["status"] == 0).all()
    for row in range(3):
        gradient = calculate_potential_gradient(1.0, component, points[row], spin[row], q[row])
        vertex_g = np.linalg.norm((const.G * mass[row] / axis[row] ** 2) * gradient, axis=1)
        face_g = np.mean(vertex_g[faces], axis=1)
        assert_allclose(result["outward_gradients"][row], gradient, rtol=3e-13, atol=64 * np.finfo(float).eps)
        assert_allclose(result["vertex_gravity"][row], vertex_g, rtol=3e-13, atol=0)
        assert_allclose(result["face_gravity"][row], face_g, rtol=3e-13, atol=0)
        assert_allclose(result["log_g"][row], np.log10(face_g), rtol=3e-13, atol=0)
    assert not np.allclose(result["face_gravity"][0], result["face_gravity"][-1])


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_actual_cpu_container_pre_correction_gravity(device: str) -> None:
    from elisa import BinarySystem, Star
    from elisa.binary_system.container import OrbitalPositionContainer

    hardware(device)
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
    system.build_mesh(components_distance=1.0)
    system.build_faces(components_distance=1.0)
    system.build_surface_gravity(components_distance=1.0)
    for component in ["primary", "secondary"]:
        star = getattr(system, component)
        kernel = RocheSurfaceGravity(component, device=device)
        result = kernel(
            star.points[None],
            star.faces,
            [binary.mass_ratio],
            [star.synchronicity],
            [system.primary.mass],
            [system.semi_major_axis],
        )
        assert np.asarray(result["proposal_valid"]).all()
        assert_allclose(result["face_gravity"][0], star.potential_gradient_magnitudes, rtol=3e-13, atol=0)
        assert_allclose(result["log_g"][0], star.log_g, rtol=3e-13, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_independent_complex_step_potential_gradient(component: str, device: str) -> None:
    hardware(device)
    points = vertices(component)
    q, spin = 1.7, 0.8

    def potential(p: np.ndarray) -> np.ndarray:
        x, y, z = p.T
        radius = np.sqrt(x * x + y * y + z * z)
        companion = np.sqrt((1 - x) ** 2 + y * y + z * z)
        local_x = x if component == "primary" else 1 - x
        indirect_mass = q if component == "primary" else 1
        return 1 / radius + q / companion - indirect_mass * local_x + 0.5 * spin**2 * (1 + q) * (local_x**2 + y * y)

    gradient = []
    for direction in range(3):
        perturbed = points.astype(complex)
        perturbed[:, direction] += 1e-25j
        gradient.append(-np.imag(potential(perturbed)) / 1e-25)
    result = RocheSurfaceGravity(component, device=device)(points[None], [[0, 1, 2]], [q], [spin], [2e30], [5e9])
    assert_allclose(result["outward_gradients"][0], np.column_stack(gradient), rtol=3e-13, atol=2e-13)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_polar_physical_forces_and_units(component: str, device: str) -> None:
    from elisa import const

    hardware(device)
    q, mass, axis, radius = 0.7, 2e30, 5e9, 0.12
    points = vertices(component)
    points[0] = [0.0 if component == "primary" else 1.0, 0.0, radius]
    result = RocheSurfaceGravity(component, device=device)(
        np.repeat(points[None], 3, axis=0),
        [[0, 1, 2]],
        [q] * 3,
        [1.2] * 3,
        [mass, 2 * mass, mass],
        [axis, axis, 2 * axis],
    )
    own_mass = mass if component == "primary" else q * mass
    other_mass = q * mass if component == "primary" else mass
    distance = np.hypot(axis, axis * radius)
    ax = const.G * other_mass * (axis / distance**3 - 1 / axis**2)
    az = -const.G * own_mass / (axis * radius) ** 2 - const.G * other_mass * (axis * radius) / distance**3
    assert_allclose(result["vertex_gravity"][0, 0], np.hypot(ax, az), rtol=3e-13, atol=0)
    assert_allclose(result["face_gravity"][1], 2 * result["face_gravity"][0], rtol=3e-13, atol=0)
    assert_allclose(result["face_gravity"][2], result["face_gravity"][0] / 4, rtol=3e-13, atol=0)
    assert_allclose(result["log_g"][1] - result["log_g"][0], [np.log10(2)], rtol=3e-13, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_unsupported_vertices_and_scaling_are_not_physical_rejections(device: str) -> None:
    hardware(device)
    kernel = RocheSurfaceGravity("primary", device=device)
    points = np.array([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [np.nan, 0.0, 0.0]]])
    result = kernel(points, [[0, 1, 2]], [1.0], [1.0], [2e30], [5e9])
    assert not np.asarray(result["proposal_valid"]).any()
    assert (np.asarray(result["status"]) == 1).all()
    assert np.isnan(result["vertex_gravity"]).all()
    assert np.isnan(result["face_gravity"]).all()
    result = kernel(vertices("primary")[None], [[0, 1, 2]], [1.0], [1.0], [1e308], [1e-300])
    assert (np.asarray(result["status"]) == 2).all()
    assert not np.asarray(result["valid_faces"]).any()


def test_gravity_input_validation_and_fp64() -> None:
    kernel = RocheSurfaceGravity("primary", device="cpu")
    args = [vertices("primary")[None], [[0, 1, 2]], [1.0], [1.0], [2e30], [5e9]]
    for index, change, message in [
        (0, [], "Vertices"),
        (1, [[0.0, 1.0, 2.0]], "integer"),
        (1, [[0, 1, 8]], "index"),
        (2, [0.1], "Unsupported"),
        (3, [2.0], "Unsupported"),
        (4, [-1.0], "positive"),
        (5, [np.inf], "finite"),
        (2, [1.0, 2.0], "proposal"),
    ]:
        changed = list(args)
        changed[index] = change
        with pytest.raises(ValueError, match=message):
            kernel(*changed)
    with pytest.raises(ValueError, match="Component"):
        RocheSurfaceGravity("unknown", device="cpu")
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
        kernel(*args)
