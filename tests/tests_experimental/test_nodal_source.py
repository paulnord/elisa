"""Physical corner samples and complete observer curves on common eclipse rays."""

import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_curvature_fields import _cpu_state, curvature_geometry
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_fused_observer import FusedObserverLightCurve
from elisa.experimental.gpu_nodal_source import NodalSourceLightCurve, _NodalBackend
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import contact_phases, fixture

from .test_gpu_boundary import CASES

DEVICE = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")
BANDS = ["Generic.Bessell.V", "bolometric"]


@pytest.mark.parametrize("case", CASES)
def test_nodal_curves(case: dict) -> None:
    with (
        table_settings(),
        settings.override(
            REFLECTION_EFFECT=True,
            REFLECTION_EFFECT_ITERATIONS=2,
            LIMB_DARKENING_LAW=case.get("law", "linear"),
            USE_SINGLE_LD_COEFFICIENTS=False,
        ),
    ):
        binary = table_fixture(case) if case.get("atlas") else fixture({"discretization": 12, **case})
        a, r = [
            NodalSourceLightCurve(BANDS, CurvedEdgeRule(3, 1), device=d, levels=(0, 1), integration_level=1)(
                binary,
                [0.017, 0.027, 0.25, 0.517, 0.533, 1.027],
                source_revision="fixture",
                diagnostics=True,
            )
            for d in [DEVICE, "reference"]
        ]
        assert a.diagnostics["phase_records"] == r.diagnostics["phase_records"]
        for key, curve in r.curves.items():
            assert curve.rule["face_local_spectral_nodes"] == curve.rule["thermal_area_order"] // 4 == 1
            for name in ("primary", "secondary"):
                x, y = [result.diagnostics["ancestry_controls"][key][name]["reconstructed_field"] for result in (a, r)]
                assert x["cell_T4_relative_error"] < 3e-13
                assert x["max_within_face_relative_intensity_span"] > 0
                for metric in ("max_within_face_relative_intensity_span", "max_within_face_LD_span"):
                    assert_allclose(x[metric], y[metric], rtol=3e-13, atol=3e-14)
                if case.get("atlas"):
                    assert x["max_within_face_LD_span"] > 0
            for attr in ("component_fluxes", "raw_fluxes", "fluxes"):
                for band in BANDS:
                    assert_allclose(
                        getattr(a.curves[key], attr)[band],
                        getattr(curve, attr)[band],
                        rtol=3e-13,
                        atol=2e-12 if attr == "fluxes" else 0,
                    )


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_source_energy_gravity_and_resource_guards(component: str) -> None:
    from elisa.binary_system.surface.gravity import calculate_polar_potential_gradient_magnitude
    from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
    from elisa.experimental.curved_geometry import duffy_nodes
    from elisa.experimental.gpu_reconstructed_field import shape_rule

    with table_settings():
        b = fixture({"discretization": 12, "inclination": 85, "spin": [0.85, 1.15]})
        system, cells, _ = curvature_geometry(b, 0, area_thermal=True)
        star = getattr(system, component)
        star.polar_potential_gradient_magnitude = calculate_polar_potential_gradient_magnitude(
            1.0,
            b.mass_ratio,
            star.polar_radius,
            component,
            star.synchronicity,
        )
        _cpu_state(b, star, component, cells[component])
        # Vary deposited T^4 across faces, retaining original mirror identities.
        star.temperatures *= 1 + 0.006 * np.sin(star.face_symmetry_vector)
        backend = _NodalBackend(component, "linear", device=DEVICE)
        tri, temp, logg, _ = backend.thermal_nodes(b, star)
        ref = _NodalBackend(component, "linear", device="reference").thermal_nodes(b, star)
        assert_allclose(temp, ref[1], rtol=3e-13)
        assert_allclose(logg, ref[2], atol=2e-13, rtol=0)
        count = len(temp) // 3
        geom = ConditionedRocheGeometry(b, component)
        nodes, weights = duffy_nodes(tri[:count], 4)
        areas = (weights * geom.sample(nodes)[2]).reshape(-1, 16)
        t4 = (np.asarray(temp).reshape(-1, 3) / star.t_eff) ** 4
        # Independently integrate the interpolated physical source, including the imposed heating.
        mean = np.sum(areas * (t4 @ shape_rule()[0].T), axis=1) / areas.sum(axis=1)
        assert_allclose(mean, (star.temperatures[:count] / star.t_eff) ** 4, rtol=3e-13)
        assert np.max(np.ptp(t4, axis=1)) > 1e-6
        star.gravity_darkening = 0
        _, flat, _, _ = backend.thermal_nodes(b, star)
        assert_allclose(flat, np.repeat(star.temperatures[:count], 3), rtol=3e-13)
        with pytest.raises(UnsupportedCurvedGeometryError, match="budget"):
            _NodalBackend(component, "linear", device=DEVICE, max_nodes=1).thermal_nodes(b, star)
        star.faces = star.faces.copy()
        star.faces[-1] = star.faces[-1, ::-1]
        with pytest.raises(UnsupportedCurvedGeometryError, match="symmetry"):
            backend.thermal_nodes(b, star)


def test_uniform_source_matches_constant_observer_at_contacts() -> None:
    with table_settings(), settings.override(REFLECTION_EFFECT=False):
        binary = fixture({"inclination": 85, "discretization": 12})
        binary.primary.gravity_darkening = binary.secondary.gravity_darkening = 0.0
        phases = contact_phases(binary)
        assert len(phases) == 57
        args = {"levels": (1, 2), "integration_level": 2}
        a = NodalSourceLightCurve(BANDS, CurvedEdgeRule(4, 1), device=DEVICE, **args)(
            binary,
            phases,
            source_revision="fixture",
        )
        r = FusedObserverLightCurve(BANDS, CurvedEdgeRule(4, 1), area_thermal=True, device="reference", **args)(
            binary,
            phases,
            source_revision="fixture",
        )
        assert a.diagnostics["phase_records"] == r.diagnostics["phase_records"]
        for key, curve in r.curves.items():
            for band in BANDS:
                assert_allclose(a.curves[key].component_fluxes[band], curve.component_fluxes[band], rtol=3e-13)
                assert_allclose(a.curves[key].fluxes[band], curve.fluxes[band], rtol=3e-13, atol=2e-12)
