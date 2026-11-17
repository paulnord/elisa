"""True Roche cell geometry, finite energy closure and full reflected-curve parity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import Observer, const, settings
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import BoundaryRule, ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_curvature_fields import CurvatureFieldLightCurve, CurvatureFieldSpec, curvature_geometry
from elisa.experimental.gpu_curved import curved_difference
from elisa.experimental.gpu_mesh_audit import roche_measures
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_binary import require_device
from .test_gpu_boundary import CASES
from .test_gpu_likelihood import fp64  # noqa: F401


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_curved_cell_areas_match_independent_angular_roots_and_centres_are_on_surface(
    component: str, level: int,
) -> None:
    b = fixture({"inclination": 85, "discretization": 12})
    system, cells, control = curvature_geometry(b, level)
    star, cell = getattr(system, component), cells[component]
    g = ConditionedRocheGeometry(b, component)
    assert_allclose(g.potential(cell["points"]), star.surface_potential, rtol=3e-13, atol=0)
    independent = roche_measures(b, component, 32)
    assert_allclose(4*cell["areas"].sum(), independent["area"], rtol=1e-10, atol=0)
    assert control[component]["scalar_correction"] is False
    assert np.all(cell["areas"] > 0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("case", CASES)
def test_curvature_fields_and_complete_reflected_curves_match_original_cpu_equations(device: str, case: dict) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
                                           LIMB_DARKENING_LAW=case.get("law", "linear")):
        b = table_fixture(case) if case.get("atlas") == "ck04" else fixture({"discretization": 12, **case})
        bands, phases = ["Generic.Bessell.V", "bolometric"], [.017, .027, .25, .517, 1.027]
        kwargs = {"levels": (0, 1), "integration_level": 1,
                  "radiance": "cpu" if case.get("atlas") == "ck04" else "device"}
        a, c = [CurvatureFieldLightCurve(bands, BoundaryRule(3, 1), device=d, **kwargs)(
            b, phases, source_revision="assets", diagnostics=True,
        ) for d in ["reference", device]]
        assert a.quadrature_sha256 == c.quadrature_sha256
        assert a.diagnostics["phase_records"] == c.diagnostics["phase_records"]
        for key in a.curves:
            for name in ["primary", "secondary"]:
                x, y = (getattr(r.diagnostics["initial_systems"][key], name) for r in [a, c])
                for f in ["face_centres", "normals", "areas", "potential_gradient_magnitudes", "log_g", "temperatures"]:
                    assert_allclose(getattr(y, f), getattr(x, f), rtol=3e-13, atol=64*np.finfo(float).eps)
                force = calculate_potential_gradient(1., name, x.face_centres, x.synchronicity, b.mass_ratio)
                independent_g = np.linalg.norm(force, axis=1)*const.G*b.primary.mass/b.semi_major_axis**2
                assert_allclose(y.potential_gradient_magnitudes, independent_g, rtol=3e-13, atol=0)
                assert_allclose(np.linalg.norm(y.normals, axis=1), 1., rtol=3e-15, atol=0)
                for r in [a, c]:
                    diag = r.diagnostics["field_geometry"][key][name]
                    assert_allclose(diag["intrinsic_luminosity_ratio"], 1., rtol=3e-13, atol=0)
                for band in bands:
                    assert_allclose(y.normal_radiance[band], x.normal_radiance[band], rtol=3e-13, atol=0)
            for band in bands:
                for f in ["fluxes", "raw_fluxes", "component_fluxes"]:
                    assert_allclose(getattr(c.curves[key], f)[band], getattr(a.curves[key], f)[band],
                                    rtol=3e-13, atol=2e-12 if f == "fluxes" else 0)
            assert c.curves[key].rule["correction_level"] == -1
            assert c.curves[key].rule["curvature_fields"] == 1


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_uniform_intrinsic_fields_are_independent_of_cell_level_on_shared_quadrature(device: str) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=False):
        b = fixture({"inclination": 85, "discretization": 12})
        b.primary.gravity_darkening = b.secondary.gravity_darkening = 0.
        r = CurvatureFieldLightCurve(["bolometric"], BoundaryRule(3, 1), levels=(0, 1, 2),
                                    integration_level=2, device=device)(
            b, [.017, .027, .25, .517], source_revision="assets", diagnostics=True,
        )
        base = r.curves["curvature_field0"]
        for key, c in r.curves.items():
            assert_allclose(c.raw_fluxes["bolometric"], base.raw_fluxes["bolometric"], rtol=3e-13, atol=0)
            for name in ["primary", "secondary"]:
                star = getattr(r.diagnostics["initial_systems"][key], name)
                assert_allclose(star.temperatures, star.t_eff, rtol=3e-13, atol=0)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_zero_albedo_reflection_preserves_intrinsic_curved_temperatures(device: str) -> None:
    require_device(device)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        b = fixture({"inclination": 85, "discretization": 12})
        b.primary.albedo = b.secondary.albedo = 0.
        r = CurvatureFieldLightCurve(["bolometric"], BoundaryRule(3, 1), levels=(0,), integration_level=0,
                                    device=device)(b, [.017, .25], source_revision="assets", diagnostics=True)
        for name in ["primary", "secondary"]:
            intrinsic = r.diagnostics["field_geometry"]["curvature_field0"][name]["intrinsic_temperatures"]
            star = getattr(r.diagnostics["initial_systems"]["curvature_field0"], name)
            assert_allclose(star.temperatures, intrinsic, rtol=0, atol=0)


def test_curvature_rebuilds_physics_response_and_keeps_original_cpu_route_identical() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=True):
        b = fixture({"inclination": 85, "discretization": 12})
        band, phases = "Generic.Bessell.V", [.017, .027, .25, .517]
        obs = Observer(passband=[band], system=b)
        before = obs.lc(phases=phases, normalize=True)[1]
        engine = CurvatureFieldLightCurve([band], BoundaryRule(3, 1), levels=(0, 1), integration_level=1, device="cpu")
        a = engine(b, phases, source_revision="assets", observer=obs)
        changed = engine(fixture({"inclination": 87, "temperatures": [6600, 5100], "discretization": 12}),
                         phases, source_revision="assets", observer=obs)
        assert a.quadrature_sha256 != changed.quadrature_sha256
        assert a.curves["curvature_field0"].signature != changed.curves["curvature_field0"].signature
        assert_allclose(obs.lc(phases=phases, normalize=True)[1][band], before[band], rtol=0, atol=0)
        tab = obs.passband[band].table.copy(deep=True)
        tab[settings.PASSBAND_DATAFRAME_THROUGHPUT] *= 1.3
        obs.passband[band].table = tab
        c = engine(b, phases, source_revision="assets", observer=obs)
        assert c.quadrature_sha256 == a.quadrature_sha256
        for key in a.curves:
            assert c.curves[key].signature != a.curves[key].signature
            assert_allclose(c.curves[key].raw_fluxes[band], 1.3*a.curves[key].raw_fluxes[band], rtol=3e-13, atol=0)
        with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(ResolutionNotQualifiedError):
            curved_difference(*a.curves.values(), 1e-7, ln_f=20., require_budget=True)


@pytest.mark.parametrize("level", [-1, 4, True, .5])
def test_curvature_level_invalid_protocol_is_explicit(level: int) -> None:
    with pytest.raises(ValueError, match="integer"):
        CurvatureFieldSpec(level)


def test_curvature_resource_and_unsupported_numerics_never_coarsen_or_cut_priors() -> None:
    with table_settings():
        engine = CurvatureFieldLightCurve(["bolometric"], BoundaryRule(3, 3, 10), levels=(0,), integration_level=0,
                                         device="cpu")
        with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
            engine(fixture({"inclination": 85, "discretization": 12}), [0], source_revision="assets")
        with pytest.raises(UnsupportedCurvedGeometryError, match="q in"):
            engine(fixture({"inclination": 85, "q": .4}), [0], source_revision="assets")
        with pytest.raises(ValueError, match="common level"):
            CurvatureFieldLightCurve(["bolometric"], BoundaryRule(), levels=(2,), integration_level=1)


def test_curvature_device_state_rejects_disabled_fp64_before_physics() -> None:
    import jax

    with table_settings():
        engine = CurvatureFieldLightCurve(["bolometric"], BoundaryRule(3, 1), levels=(0,),
                                         integration_level=0, device="cpu")
        b = fixture({"inclination": 85, "discretization": 12})
        with jax.enable_x64(False), pytest.raises(RuntimeError, match="remain"):
            engine(b, [.25], source_revision="assets")
