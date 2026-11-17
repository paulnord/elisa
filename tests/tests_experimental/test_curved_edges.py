"""Independent eclipse areas, complete CPU parity and accumulated density stops."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, settings
from elisa.experimental.boundary_geometry import BoundaryRule
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, split_triangles
from elisa.experimental.gpu_curved import CurvedCurveResult
from elisa.experimental.gpu_curved_edges import (
    CurvedEdgeLightCurve,
    CurvedEdgeRule,
    InferenceBudget,
    curved_edge_packet,
)
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_binary import require_device
from .test_gpu_boundary import CASES
from .test_gpu_likelihood import fp64  # noqa: F401


def sphere_triangles() -> np.ndarray:
    e = np.eye(3)
    triangles = np.array([e*np.array([sx, sy, sz])[:, None]
                          for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]])
    for _ in range(3):
        triangles = split_triangles(triangles)
        triangles /= np.linalg.norm(triangles, axis=2)[:, :, None]
    return triangles


class Sphere:
    """Analytic sphere and projected disk, independent of Roche implementation."""

    def __init__(self, radius: float = 1., offset: float = 0.) -> None:
        self.radius, self.offset = radius, offset

    def sample(self, vectors: np.ndarray) -> tuple:
        normals = vectors/np.linalg.norm(vectors, axis=1)[:, None]
        return self.radius*normals, normals, np.full(len(vectors), self.radius**2)

    def shadow(self, points: np.ndarray, line: np.ndarray) -> tuple:
        assert_allclose(line, [0, 0, 1], rtol=0, atol=0)
        distance = np.linalg.norm(points[:, :2]-np.array([self.offset, 0.]), axis=1)-self.radius
        return distance < 0, distance


def overlap(radius: float, distance: float) -> float:
    if distance >= 1+radius:
        return 0.
    if distance <= abs(1-radius):
        return np.pi*min(1., radius)**2
    a = (distance**2+1-radius**2)/(2*distance)
    b = (distance**2+radius**2-1)/(2*distance*radius)
    return (np.arccos(a)+radius**2*np.arccos(b)
            -.5*np.sqrt((-distance+1+radius)*(distance+1-radius)
                         *(distance-1+radius)*(distance+1+radius)))


@pytest.mark.parametrize(("radius", "distance"), [(.6, 0.), (.6, .8), (.6, 1.59),
                                                 (.6, 1.61), (1.4, .1), (.01, .4)])
def test_projected_uniform_sphere_matches_independent_circle_intersection(radius: float, distance: float) -> None:
    packet = curved_edge_packet(sphere_triangles(), Sphere(), np.array([0., 0., 1.]),
                                CurvedEdgeRule(4, 5), Sphere(radius, distance))
    actual = np.sum(packet["mu"]*packet["area"])/np.pi
    expected = 1-overlap(radius, distance)/np.pi
    # 1e-7 of the unobscured disk is .001 sigma at the retained1e-4 scale.
    assert_allclose(actual, expected, rtol=0, atol=1e-7)
    assert np.all(packet["area"] > 0)
    assert packet["nodes"] <= 2097152


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("case", CASES)
def test_complete_curved_edge_reflected_curves_match_original_cpu_equations(device: str, case: dict) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
                                           LIMB_DARKENING_LAW=case.get("law", "linear")):
        b = table_fixture(case) if case.get("atlas") == "ck04" else fixture({"discretization": 12, **case})
        bands, phases = ["Generic.Bessell.V", "bolometric"], [0., .017, .027, .25, .517, 1.027]
        kwargs = {"levels": (0, 1), "integration_level": 1,
                  "radiance": "cpu" if case.get("atlas") == "ck04" else "device"}
        a, c = [CurvedEdgeLightCurve(bands, CurvedEdgeRule(3, 1), device=d, **kwargs)(
            b, phases, source_revision="assets", diagnostics=True,
        ) for d in ["reference", device]]
        assert a.quadrature_sha256 == c.quadrature_sha256
        assert a.diagnostics["phase_records"] == c.diagnostics["phase_records"]
        for key in a.curves:
            for name in ["primary", "secondary"]:
                x, y = (getattr(r.diagnostics["initial_systems"][key], name) for r in [a, c])
                for f in ["face_centres", "normals", "areas", "potential_gradient_magnitudes", "log_g", "temperatures"]:
                    assert_allclose(getattr(y, f), getattr(x, f), rtol=3e-13, atol=64*np.finfo(float).eps)
            for band in bands:
                for f in ["fluxes", "raw_fluxes", "component_fluxes"]:
                    assert_allclose(getattr(c.curves[key], f)[band], getattr(a.curves[key], f)[band],
                                    rtol=3e-13, atol=2e-12 if f == "fluxes" else 0)


def test_original_cpu_unchanged_and_physical_response_rebuilt() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        b = fixture({"inclination": 85, "discretization": 12})
        band, phases = "bolometric", [.017, .027, .25, .517]
        observer = Observer(passband=[band], system=b)
        before = observer.lc(phases=phases, normalize=True)[1][band]
        e = CurvedEdgeLightCurve([band], CurvedEdgeRule(3, 1), levels=(0,), integration_level=1, device="cpu")
        a = e(b, phases, observer=observer, source_revision="assets")
        c = e(fixture({"inclination": 87, "temperatures": [6600, 5100], "discretization": 12}),
              phases, observer=observer, source_revision="assets")
        assert a.quadrature_sha256 != c.quadrature_sha256
        assert a.curves["curvature_field0"].signature != c.curves["curvature_field0"].signature
        assert_allclose(observer.lc(phases=phases, normalize=True)[1][band], before, rtol=0, atol=0)


def test_accumulated_likelihood_stop_is_separate_from_distance_and_nuisance_noise() -> None:
    component = np.array([[.8, .2], [.7, .2], [.6, .2]])
    a = CurvedCurveResult(np.array([0., .1, .2]), {}, {}, {"b": component}, "fixed", {}, 0, 0., {}, {})
    change = np.array([[0., 0.], [-1e-8, 0.], [-1e-8, 0.]])
    b = replace(a, component_fluxes={"b": component+change})
    observed = {"b": np.array([.9, .8, .7])}
    budget = InferenceBudget()
    r = budget.compare(a, b, 1e-4, observed=observed, require_budget=False)
    assert r["nominal_report"]["within_finite_budget"]
    assert r["nominal_report"]["finite_log_likelihood_absolute_bound"] > .01
    assert not r["within_finite_budget"]
    assert r == budget.compare(a, b, 1e-4, observed=observed, ln_f=20., require_budget=False)
    actual = -.5*np.sum(((observed["b"]-(component+change).sum(axis=1))/1e-4)**2)
    actual += .5*np.sum(((observed["b"]-component.sum(axis=1))/1e-4)**2)
    assert_allclose(r["nominal_report"]["actual"]["delta_log_likelihood"], actual, rtol=1e-7, atol=1e-8)
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError):
        budget.compare(a, b, 1e-4, observed=observed, ln_f=20.)
    assert budget.compare(a, a, 1e-4, observed=observed)["within_finite_budget"]


@pytest.mark.parametrize("options", [{"junction_depth": -1}, {"junction_depth": 5},
                                     {"junction_depth": True}, {"curve_bisections": 35},
                                     {"slice_order": 3}, {"slice_order": 13}])
def test_invalid_fixed_edge_protocol_is_explicit(options: dict) -> None:
    with pytest.raises(ValueError, match="Curved edges"):
        CurvedEdgeRule(**options)


def test_unsupported_and_capacity_failures_do_not_coarsen_or_change_priors() -> None:
    with table_settings():
        e = CurvedEdgeLightCurve(["bolometric"], CurvedEdgeRule(3, 1, 10), levels=(0,),
                                integration_level=0, device="cpu")
        with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
            e(fixture({"inclination": 85}), [.017], source_revision="assets")
        with pytest.raises(UnsupportedCurvedGeometryError, match="q in"):
            e(fixture({"inclination": 85, "q": .4}), [.017], source_revision="assets")
        with pytest.raises(TypeError, match="CurvedEdgeRule"):
            CurvedEdgeLightCurve(["bolometric"], BoundaryRule(), device="cpu")
