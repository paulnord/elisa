"""Independent analytic and original-equation boundary integration controls."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq

from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import BoundaryRule, ConditionedRocheGeometry, boundary_packet
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from scripts.validate_gpu_binary import fixture

from .test_curved_geometry import Sphere, disk_overlap, triangles


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("case", [{}, {"q": .5, "spin": [.5, 1.5]},
                                  {"q": 2., "spin": [1.5, .5], "potentials": [12, 100000]}])
def test_local_equations_match_original_extended_precision_gradient_and_potential(component: str, case: dict) -> None:
    binary = fixture({"inclination": 85, "discretization": 12, **case})
    g = ConditionedRocheGeometry(binary, component)
    rng = np.random.default_rng(1401)
    local = rng.normal(size=(21, 3))
    local *= g.inner*1.1/np.linalg.norm(local, axis=1)[:, None]
    world = np.longdouble(local)+np.longdouble(g.centre)
    expected = calculate_potential_gradient(1., component, world, g.star.synchronicity, binary.mass_ratio)
    assert_allclose(g.gradient_local(local), expected, rtol=3e-13, atol=64*np.finfo(float).eps)
    # At ordinary radii the original potential world-coordinate mapping is also
    # independently compared; extended precision retains the tiny secondary.
    assert_allclose(g.potential_local(local), g.potential(world), rtol=3e-13, atol=0)


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("potentials", [[8, 100], [10000, 100000]])
def test_conditioned_small_star_shadow_matches_independent_original_brent(component: str, potentials: list) -> None:
    b = fixture({"inclination": 85, "potentials": potentials})
    g = ConditionedRocheGeometry(b, component)
    line = np.array([.98, .1, .18])
    line /= np.linalg.norm(line)
    u = np.cross(line, [0, 0, 1])
    u /= np.linalg.norm(u)
    v = np.cross(line, u)
    rho, angle = g.inner*np.linspace(.9, 1.3, 31), np.linspace(0, 2*np.pi, 31)
    points = g.centre+rho[:, None]*(np.cos(angle)[:, None]*u+np.sin(angle)[:, None]*v)-line
    actual, margin = g.shadow(points, line)
    expected = []
    # Oracle evaluates the unchanged original equation with an extended-precision
    # world coordinate, a distinct scalar Brent solve and no stationary shortcut.
    for point in points:
        delta = point-g.centre
        base = delta-(delta@line)*line
        radius = np.linalg.norm(base)
        def derivative(t: float, base: np.ndarray = base) -> float:
            world = np.longdouble(g.centre)+np.longdouble(base)+np.longdouble(t)*np.longdouble(line)
            grad = calculate_potential_gradient(1., component, world[None], g.star.synchronicity, b.mass_ratio)[0]
            return float(grad@np.longdouble(line))
        t = brentq(derivative, -radius/2, radius/2, xtol=1e-22)
        world = np.longdouble(g.centre)+np.longdouble(base)+np.longdouble(t)*np.longdouble(line)
        expected.append(g.potential(world[None])[0] > g.star.surface_potential)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, margin < 0)


@pytest.mark.parametrize(("small", "distance"), [(.065, 0.), (.065, .12), (.065, .19), (.2, 0.)])
def test_fitted_sphere_eclipse_has_independent_disk_overlap_accuracy(small: float, distance: float) -> None:
    r = .137
    packet = boundary_packet(triangles(), Sphere(r, [0, 0, 0]), np.array([1., 0, 0]), BoundaryRule(3, 5),
                             Sphere(small, [1., distance, 0]))
    actual = np.sum(packet["area"]*packet["mu"])/(np.pi*r*r)
    expected = 1-disk_overlap(r, small, distance)/(np.pi*r*r)
    # Fixed independent target: <=.05 sigma of a single1e-4 observation, matching
    # the preceding analytic disk control, without changing a CPU/GPU gate.
    assert_allclose(actual, expected, rtol=0, atol=5e-6)


@pytest.mark.parametrize("line", [[1., 0, 0], [.8, .36, .48]])
def test_fitted_limb_recovers_analytic_linear_ld_flux(line: list) -> None:
    r = .137
    p = boundary_packet(triangles(), Sphere(r, [0, 0, 0]), np.array(line), BoundaryRule(4, 3))
    flux = np.sum(p["area"]*p["mu"]*(.5+.5*p["mu"]))
    assert_allclose(flux/(np.pi*r*r), 1-.5/3, rtol=0, atol=1e-6)


def test_fitted_resource_and_unresolved_contour_abort_without_coarsening() -> None:
    tri = triangles()
    with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
        boundary_packet(tri, Sphere(.137, [0, 0, 0]), np.array([1., 0, 0]), BoundaryRule(3, 3, 10))
    # A small opaque disk centered on the interior ray is detected by the
    # seven-point probe, but cannot be represented by this unsplit leaf's edges.
    face = tri[np.argmax(tri.mean(axis=1)[:, 0])][None]
    centre = .137*face.mean(axis=1)[0]/np.linalg.norm(face.mean(axis=1)[0])
    with pytest.raises(UnsupportedCurvedGeometryError, match="topology"):
        boundary_packet(face, Sphere(.137, [0, 0, 0]), np.array([1., 0, 0]), BoundaryRule(3, 0, topology_depth=0),
                        Sphere(1e-6, centre+np.array([1., 0, 0])))


@pytest.mark.parametrize("value", [True, 31, 49, 36.5])
def test_boundary_root_protocol_is_fixed_and_explicit(value: int) -> None:
    with pytest.raises(ValueError, match="Boundary edge"):
        BoundaryRule(edge_bisections=value)


@pytest.mark.parametrize("value", [True, -1, 5, .3])
def test_topology_subdivision_limit_is_explicit(value: int) -> None:
    with pytest.raises(ValueError, match="Boundary topology"):
        BoundaryRule(topology_depth=value)
