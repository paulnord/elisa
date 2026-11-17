"""Independent root/Jacobian/shadow and analytic curved-eclipse controls."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq

from elisa.binary_system.radius import calculate_radius
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.curved_geometry import (
    CurvedRule,
    RocheGeometry,
    UnsupportedCurvedGeometryError,
    curved_packet,
    duffy_nodes,
)
from elisa.experimental.gpu_selective_mesh import seed_system
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator
from scripts.validate_gpu_binary import fixture

from .test_gpu_binary import require_device
from .test_gpu_likelihood import fp64  # noqa: F401


class Sphere:
    """Independent analytic test geometry; never a binary synthesis fallback."""

    def __init__(self, radius: float, centre: Any) -> None:
        self.radius, self.centre = radius, np.asarray(centre)

    def sample(self, vectors: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = vectors/np.linalg.norm(vectors, axis=1)[:, None]
        return self.centre+d*self.radius, d, np.full(len(d), self.radius**2)

    def shadow(self, points: Any, line: Any) -> tuple[np.ndarray, np.ndarray]:
        d = points-self.centre
        d -= np.outer(d@line, line)
        distance = np.linalg.norm(d, axis=1)-self.radius
        return distance < 0, distance


def triangles() -> np.ndarray:
    star = seed_system(fixture({"inclination": 85, "discretization": 12})).primary
    d = star.points/np.linalg.norm(star.points, axis=1)[:, None]
    return d[star.faces]


def disk_overlap(r: float, small: float, distance: float) -> float:
    if distance >= r+small:
        return 0.
    if distance <= abs(r-small):
        return np.pi*min(r, small)**2
    a = np.arccos((distance**2+r*r-small*small)/(2*distance*r))
    b = np.arccos((distance**2+small*small-r*r)/(2*distance*small))
    area = .5*np.sqrt((-distance+r+small)*(distance+r-small)*(distance-r+small)*(distance+r+small))
    return r*r*a+small*small*b-area


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("case", [{}, {"q": .5, "spin": [.5, 1.5]},
                                  {"q": 2., "spin": [1.5, .5], "potentials": [12, 16]}])
def test_curved_radial_points_match_original_independent_radius_and_jacobian(component: str, case: dict) -> None:
    binary = fixture({"inclination": 85, "discretization": 12, **case})
    geometry = RocheGeometry(binary, component)
    rng = np.random.default_rng(1301)
    d = rng.normal(size=(12, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    p, n, jac = geometry.sample(d)
    local = d.copy()
    if component == "secondary":
        local[:, 0] *= -1
    phi, theta = np.arctan2(local[:, 1], local[:, 0]), np.arccos(local[:, 2])
    roots = [calculate_radius(geometry.star.synchronicity, binary.mass_ratio, geometry.star.surface_potential,
                              component, 1., float(a), float(b)) for a, b in zip(phi, theta, strict=True)]
    assert_allclose(np.linalg.norm(p-geometry.centre, axis=1), roots, rtol=2e-11, atol=1e-13)
    assert_allclose(geometry.potential(p), geometry.star.surface_potential, rtol=2e-13, atol=0)
    assert_allclose(np.linalg.norm(n, axis=1), 1., rtol=3e-15, atol=0)
    assert np.all(jac > 0)
    # Independent angular finite differences, distinct from the implicit-gradient Jacobian.
    phi, theta, h = .7, 1.1, 2e-6
    def point(a: float, b: float) -> np.ndarray:
        direction = np.array([[np.sin(b)*np.cos(a), np.sin(b)*np.sin(a), np.cos(b)]])
        return geometry.sample(direction)[0][0]
    dp = (point(phi+h, theta)-point(phi-h, theta))/(2*h)
    dt = (point(phi, theta+h)-point(phi, theta-h))/(2*h)
    direction = np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]])
    assert_allclose(geometry.sample(direction)[2][0], np.linalg.norm(np.cross(dp, dt))/np.sin(theta),
                    rtol=5e-8, atol=0)


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_implicit_shadow_matches_independent_brent_line_tangency(component: str) -> None:
    binary = fixture({"inclination": 85, "discretization": 12})
    g = RocheGeometry(binary, component)
    line = np.array([.98, .1, .18])
    line /= np.linalg.norm(line)
    u = np.cross(line, [0, 0, 1])
    u /= np.linalg.norm(u)
    v = np.cross(line, u)
    rho = np.linspace(g.inner*.8, .245, 21)
    angle = np.linspace(0, 2*np.pi, len(rho))
    base = g.centre+rho[:, None]*(np.cos(angle)[:, None]*u+np.sin(angle)[:, None]*v)
    actual, _ = g.shadow(base-np.outer(np.ones(len(base)), line), line)
    expected = []
    for point, r in zip(base, rho, strict=True):
        def directional(t: float, point: np.ndarray = point) -> float:
            p = np.array([point+t*line])
            return float(calculate_potential_gradient(1., component, p, g.star.synchronicity,
                                                      binary.mass_ratio)[0]@line)
        root = brentq(directional, -r/2, r/2, xtol=5e-16)
        expected.append(g.potential(np.array([point+root*line]))[0] > g.star.surface_potential)
    assert np.array_equal(actual, expected)


def test_duffy_solid_angle_and_curved_sphere_area() -> None:
    tri = triangles()
    _, weights = duffy_nodes(tri, 4)
    assert_allclose(weights.sum(), 4*np.pi, rtol=2e-7, atol=0)
    packet = curved_packet(tri, Sphere(.137, [0, 0, 0]), np.array([1., 0, 0]), CurvedRule(4, 4))
    assert_allclose(np.sum(packet["area"]*packet["mu"]), np.pi*.137**2, rtol=1e-6, atol=0)


@pytest.mark.parametrize(("small", "distance"), [(.065, 0.), (.065, .12), (.065, .19), (.2, 0.)])
def test_sphere_eclipse_boundary_refinement_matches_analytic_overlap(small: float, distance: float) -> None:
    r = .137
    packet = curved_packet(triangles(), Sphere(r, [0, 0, 0]), np.array([1., 0, 0]), CurvedRule(3, 7),
                           Sphere(small, [1., distance, 0]))
    flux = np.sum(packet["area"]*packet["mu"])
    expected = np.pi*r*r-disk_overlap(r, small, distance)
    # Absolute disk-normalized error <=.05 sigma for a single1e-4 observation.
    assert_allclose(flux/(np.pi*r*r), expected/(np.pi*r*r), rtol=0, atol=5e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_device_curved_sphere_flux_uses_original_ld_and_weight_contract(device: str) -> None:
    require_device(device)
    r = .137
    packet = curved_packet(triangles(), Sphere(r, [0, 0, 0]), np.array([1., 0, 0]), CurvedRule(3, 4))
    n = len(packet["mu"])
    kernel = SurfaceFluxIntegrator("linear", device=device)
    _, total = kernel(np.ones((1, n)), packet["mu"][None], packet["area"][None], np.full((1, n, 1), .5))
    expected = np.sum(packet["mu"]*packet["area"]*(.5+.5*packet["mu"]))
    assert_allclose(np.asarray(total)[0], expected, rtol=3e-13, atol=0)
    assert_allclose(np.asarray(total)[0]/(np.pi*r*r), 1-.5/3, rtol=0, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"order": 1}, {"depth": 8}, {"order": True}, {"max_nodes": 0}])
def test_invalid_curved_rule_is_explicit(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="Curved rule"):
        CurvedRule(**kwargs)


def test_node_exhaustion_aborts_without_silent_coarsening() -> None:
    with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
        curved_packet(triangles(), Sphere(.137, [0, 0, 0]), np.array([1., 0, 0]), CurvedRule(3, 3, 10))
