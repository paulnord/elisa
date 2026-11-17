"""Independent thermal quadrature, matched packets and complete curve controls."""

import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_fused_observer import FusedObserverLightCurve
from elisa.experimental.gpu_resident_geometry import _angular_inputs, _ResidentField
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

from .test_gpu_boundary import CASES

DEVICE = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")
BANDS = ["Generic.Bessell.V", "bolometric"]


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("beta", [0.0, 0.32, 1.0])
def test_cell_emissivity_against_finer_original_gravity(component: str, beta: float) -> None:
    with table_settings():
        b = fixture({"inclination": 85, "discretization": 12, "q": 0.7, "spin": [0.8, 1.2]})
        _, inputs = _angular_inputs(b, 0)
        row = inputs[component]
        row["physics"][-1] = beta
        field = _ResidentField(component, device=DEVICE, area_thermal=True)
        result = field.evaluate([row])
        assert result["temperatures"].device.platform == DEVICE
        assert abs(float(result["intrinsic_luminosity_ratio"][0]) - 1) < 3e-13
        n = len(row["rays"])
        temp = np.asarray(result["temperatures"])[0, :n]
        if beta == 0:
            assert_allclose(temp, row["physics"][2], rtol=3e-14)
            return
        tri = row["triangles"][:n]
        ref = ConditionedRocheGeometry(b, component)

        def temperatures(order: int) -> np.ndarray:
            rays, w = duffy_nodes(tri, order)
            points, _, jac = ref.sample(rays)
            g = calculate_potential_gradient(1.0, component, points, getattr(b, component).synchronicity, b.mass_ratio)
            area = (w * jac).reshape(n, -1)
            weighted = (np.linalg.norm(g, axis=1) ** beta).reshape(area.shape) * area
            return (
                row["physics"][2] * ((weighted.sum(axis=1) / area.sum(axis=1)) * (area.sum() / weighted.sum())) ** 0.25
            )

        assert_allclose(temp, temperatures(4), rtol=3e-13, atol=0)
        high_order = temperatures(8)
        centre = _ResidentField(component, device=DEVICE).evaluate([row])
        assert np.linalg.norm(temp - high_order) < np.linalg.norm(
            np.asarray(centre["temperatures"])[0, :n] - high_order,
        )


@pytest.mark.parametrize("case", CASES)
def test_complete_area_source_curve_and_reflection(case: dict) -> None:
    with (
        table_settings(),
        settings.override(
            REFLECTION_EFFECT=True,
            REFLECTION_EFFECT_ITERATIONS=2,
            LIMB_DARKENING_LAW=case.get("law", "linear"),
        ),
    ):
        binary = table_fixture(case) if case.get("atlas") else fixture({"discretization": 12, **case})
        phases = np.array([0.517, 0.017, 0.027, 0.25, 0.508, 0.533, 1.027])
        kwargs = {
            "levels": (0, 1),
            "integration_level": 1,
            "area_thermal": True,
            "radiance": "cpu" if case.get("atlas") else "device",
        }
        a, r = [
            FusedObserverLightCurve(BANDS, CurvedEdgeRule(3, 1), device=d, **kwargs)(
                binary,
                phases,
                source_revision="test-tables",
                diagnostics=True,
            )
            for d in [DEVICE, "reference"]
        ]
        assert a.diagnostics["phase_records"] == r.diagnostics["phase_records"]
        for k, curve in r.curves.items():
            assert a.curves[k].rule == curve.rule
            assert curve.rule["thermal_area_order"] == 4
            for name in ("primary", "secondary"):
                x, y = [getattr(v.diagnostics["initial_systems"][k], name) for v in [a, r]]
                assert_allclose(x.temperatures, y.temperatures, rtol=3e-13)
                for band in BANDS:
                    assert_allclose(x.normal_radiance[band], y.normal_radiance[band], rtol=3e-13)
                    assert_allclose(x.ld_cfs[band], y.ld_cfs[band], rtol=3e-13, atol=2e-14)
            for attr in ("component_fluxes", "raw_fluxes", "fluxes"):
                for band in BANDS:
                    assert_allclose(
                        getattr(a.curves[k], attr)[band],
                        getattr(curve, attr)[band],
                        rtol=3e-13,
                        atol=2e-12 if attr == "fluxes" else 0,
                    )


def test_invalid_option_and_outside_prior_resource_stops() -> None:
    with table_settings(), settings.override(PENALIZE_ALL_EXCEPTIONS=True):
        with pytest.raises(TypeError, match="boolean"):
            FusedObserverLightCurve(BANDS, CurvedEdgeRule(3, 1), device="reference", area_thermal="yes")
        b = fixture({"inclination": 85, "discretization": 12})
        e = FusedObserverLightCurve(BANDS, CurvedEdgeRule(3, 1), device=DEVICE, area_thermal=True, max_resident_faces=1)
        with pytest.raises(UnsupportedCurvedGeometryError, match="face budget"):
            e(b, [0.25], source_revision="test-tables")


def test_refined_area_source_across_contacts() -> None:
    from scripts.validate_gpu_binary import contact_phases

    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        binary = fixture({"inclination": 85, "discretization": 12})
        phases = contact_phases(binary)
        assert len(phases) == 57
        actual, reference = [
            FusedObserverLightCurve(
                BANDS, CurvedEdgeRule(4, 1), levels=(1, 2), integration_level=2, device=d, area_thermal=True,
            )(binary, phases, source_revision="test-tables")
            for d in [DEVICE, "reference"]
        ]
        assert actual.diagnostics["phase_records"] == reference.diagnostics["phase_records"]
        for key, curve in reference.curves.items():
            for band in BANDS:
                assert_allclose(actual.curves[key].component_fluxes[band], curve.component_fluxes[band], rtol=3e-13)
                assert_allclose(actual.curves[key].fluxes[band], curve.fluxes[band], rtol=3e-13, atol=2e-12)
