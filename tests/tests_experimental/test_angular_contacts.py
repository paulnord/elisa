"""Reflected light through the contact phases of both eclipses."""

import os

import pytest

from numpy.testing import assert_allclose

from elisa import settings
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import AngularReflectedLightCurve, SpatialReflectedLightCurve
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import contact_phases, fixture


@pytest.mark.parametrize("observer", [AngularReflectedLightCurve, SpatialReflectedLightCurve])
def test_angular_reflection_across_both_contacts(observer):
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        binary = fixture({"inclination": 85, "discretization": 12})
        phases = contact_phases(binary)
        assert len(phases) == 57
        bands = ["Generic.Bessell.V", "bolometric"]
        kwargs = dict(levels=(1, 2), integration_level=2, donor_level=1)
        results = [
            observer(bands, CurvedEdgeRule(4, 1), device=device, **kwargs)(
                binary, phases, source_revision="contact-fixture"
            )
            for device in ("reference", os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu"))
        ]
        ref, chosen = results
        assert ref.diagnostics["phase_records"] == chosen.diagnostics["phase_records"]
        for key, curve in ref.curves.items():
            for attr in ("fluxes", "component_fluxes", "raw_fluxes"):
                for band, raw in getattr(curve, attr).items():
                    assert_allclose(
                        getattr(chosen.curves[key], attr)[band], raw, rtol=3e-13, atol=2e-12 if attr == "fluxes" else 0
                    )
