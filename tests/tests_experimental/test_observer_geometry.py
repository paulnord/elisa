"""Original shadow, root and complete-curve controls for the observer device port."""

import os
from functools import lru_cache

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, settings
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_batched_geometry import BatchedRocheGeometry
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, InferenceBudget
from elisa.experimental.gpu_observer_edges import ObserverDeviceLightCurve
from elisa.experimental.gpu_observer_geometry import ObserverGeometryBackend
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_boundary import CASES

DEVICE = os.environ.get("ELISA_GEOMETRY_TEST_DEVICE", "cpu")
BANDS = ["Generic.Bessell.V", "bolometric"]


@lru_cache(None)
def backend() -> ObserverGeometryBackend:
    return ObserverGeometryBackend(device=DEVICE)


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("case", [CASES[0], CASES[3], CASES[4], CASES[5]])
def test_original_conditioned_shadows_and_stationarity(component: str, case: dict) -> None:
    b = fixture({"discretization": 12, **{k:v for k,v in case.items() if k != "law"}})
    g = backend().bind(b, component)
    for line in [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 2.0, 3.0]) / np.sqrt(14)]:
        axis = np.eye(3)[np.argmin(abs(line))]
        u = np.cross(line, axis)
        u /= np.linalg.norm(u)
        v = np.cross(line, u)
        rho = np.concatenate([np.linspace(0, 0.26, 101), [g.inner, np.nextafter(g.inner, 0), 0.25]])
        angles = np.arange(len(rho)) * 0.43
        points = g.centre + rho[:, None] * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
        points += 0.2 * line
        actual, original = g.shadow(points, line), g.reference.shadow(points, line)
        np.testing.assert_array_equal(actual[0], original[0])
        assert_allclose(actual[1], original[1], rtol=3e-13, atol=64 * np.finfo(float).eps)
        assert np.array_equal(actual[0], actual[1] < 0)


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_compiled_nested_limb_and_shadow_roots_match_original_final_bins(component: str) -> None:
    b = fixture({"inclination": 85, "discretization": 12})
    g = backend().bind(b, component)
    foreground = backend().bind(b, "secondary" if component == "primary" else "primary")
    line = np.array([1.0, 0.04, 0.02])
    line /= np.linalg.norm(line)
    rays = np.random.default_rng(19).normal(size=(64, 3))
    for f in [g.field(line), g.field(line, foreground)]:
        a, d = rays[:32], rays[32:] - rays[:32]
        low, high = np.zeros(32), np.ones(32)
        positive = f.reference(a) >= 0
        take = positive != (f.reference(a + d) >= 0)
        if not take.any():
            continue
        aa, dd, lp = a[take], d[take], positive[take]
        actual = f.bisect(aa, dd, low[take], high[take], lp, 36)
        lo, hi = low[take], high[take]
        for _ in range(36):
            mid = (lo + hi) / 2
            same = (f.reference(aa + dd * mid[:, None]) >= 0) == lp
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        np.testing.assert_array_equal(actual[0], lo)
        np.testing.assert_array_equal(actual[1], hi)
    assert backend().stats["edge_root_rows"] > 0


@pytest.mark.parametrize("case", CASES)
def test_complete_observer_curves_fields_and_adaptive_counts(case: dict) -> None:
    with (
        table_settings(),
        settings.override(
            REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2, LIMB_DARKENING_LAW=case.get("law", "linear"),
        ),
    ):
        b = table_fixture(case) if case.get("atlas") else fixture({"discretization": 12, **case})
        phases = [0.0, 0.017, 0.027, 0.25, 0.517, 1.027]
        results = [
            ObserverDeviceLightCurve(
                BANDS,
                CurvedEdgeRule(3, 1),
                levels=(0, 1),
                integration_level=1,
                device=d,
                radiance="cpu" if case.get("atlas") else "device",
            )(b, phases, source_revision="assets", diagnostics=True)
            for d in ["reference", DEVICE]
        ]
        a, r = results[1], results[0]
        assert a.diagnostics["phase_records"] == r.diagnostics["phase_records"]
        for key in r.curves:
            for attr in ["fluxes", "raw_fluxes", "component_fluxes"]:
                for band in BANDS:
                    assert_allclose(
                        getattr(a.curves[key], attr)[band],
                        getattr(r.curves[key], attr)[band],
                        rtol=3e-13,
                        atol=2e-12 if attr == "fluxes" else 0,
                    )
            for name in ["primary", "secondary"]:
                x, y = [getattr(v.diagnostics["initial_systems"][key], name) for v in [a, r]]
                for attr in ["face_centres", "normals", "areas", "temperatures"]:
                    assert_allclose(getattr(x, attr), getattr(y, attr), rtol=3e-13, atol=64 * np.finfo(float).eps)


def test_changes_rebuild_observer_physics_and_original_cpu_is_exact() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        b = fixture({"inclination": 85, "discretization": 12})
        phases = [0, 0.017, 0.25, 0.5]
        observer = Observer(passband=BANDS, system=b)
        before = observer.lc(phases=phases, normalize=True)[1]
        e = ObserverDeviceLightCurve(BANDS, CurvedEdgeRule(3, 1), levels=(0, 1), integration_level=1, device=DEVICE)
        changed = fixture(
            {
                "inclination": 86,
                "discretization": 12,
                "q": 0.8,
                "potentials": [8.1, 12.2],
                "temperatures": [6600, 5100],
                "spin": [1.1, 0.9],
            },
        )
        batch = e.batch([b, changed], [phases, phases], observer=observer, source_revision="assets")
        assert len({r.curves["curvature_field1"].signature for r in batch.results}) == 2
        after = observer.lc(phases=phases, normalize=True)[1]
        for band in BANDS:
            assert_allclose(before[band], after[band], rtol=0, atol=0)
        reference = ObserverDeviceLightCurve(
            BANDS, CurvedEdgeRule(3, 1), levels=(0, 1), integration_level=1, device="reference",
        )(changed, phases, observer=observer, source_revision="assets")
        for band in BANDS:
            assert_allclose(
                batch.results[1].curves["curvature_field1"].fluxes[band],
                reference.curves["curvature_field1"].fluxes[band],
                rtol=3e-13,
                atol=2e-12,
            )
        assert e.last_observer_stats["edge_solve_calls"] > 0


def test_resource_x64_and_accuracy_stops_are_not_priors() -> None:
    with table_settings(), settings.override(PENALIZE_ALL_EXCEPTIONS=True):
        b = fixture({"inclination": 85, "discretization": 12})
        e = ObserverGeometryBackend(device=DEVICE, max_nodes=16)
        g = e.bind(b, "primary")
        with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
            g.sample(np.ones((17, 3)))
        with e.jax.enable_x64(False), pytest.raises(RuntimeError, match="FP64"):
            g.sample(np.ones((3, 3)))
        with pytest.raises(UnsupportedCurvedGeometryError, match="unit LOS"):
            g.shadow(np.ones((3, 3)), np.ones(3))
        with pytest.raises(UnsupportedCurvedGeometryError, match="q in"):
            e.bind(fixture({"inclination": 85, "q": 0.4}), "primary")
        r = ObserverDeviceLightCurve(BANDS, CurvedEdgeRule(3, 1), levels=(0, 1), integration_level=1, device=DEVICE)(
            b, [0, 0.017, 0.25, 0.5], source_revision="assets",
        )
        with pytest.raises(ResolutionNotQualifiedError):
            InferenceBudget(max_distance_sigma=1e-12, max_log_likelihood=1e-12).compare(
                r.curves["curvature_field0"],
                r.curves["curvature_field1"],
                1e-4,
                observed=r.curves["curvature_field1"].fluxes,
            )


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("unroll", [2, 4, 8])
def test_unrolled_radial_graph_matches_original_primitive(component: str, unroll: int) -> None:
    b=fixture({"inclination":85,"discretization":12})
    g=backend().bind(b,component)
    rays=np.random.default_rng(19019).normal(size=(64,3))
    engine=BatchedRocheGeometry(component,device=DEVICE,root_unroll=unroll)
    p=np.array([[b.mass_ratio,g.star.synchronicity,g.star.surface_potential]])
    actual=engine.checked_host(engine.sample(p,rays,compact=True))
    expected=g.reference.sample(rays)
    for key,value in zip(["points","normals","area_per_solid_angle"],expected,strict=True):
        assert_allclose(actual[key][0],value,rtol=3e-13,
                        atol=64*np.finfo(float).eps if key!="area_per_solid_angle" else 0)
    for invalid in [True,0,3,8.]:
        with pytest.raises(ValueError,match="root_unroll"):
            BatchedRocheGeometry(component,device=DEVICE,root_unroll=invalid)
