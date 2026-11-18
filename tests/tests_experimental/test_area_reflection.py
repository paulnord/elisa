"""Independent receiver means, iteration semantics, source grouping and complete curves."""

from copy import deepcopy
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa.experimental import area_reflection
from elisa.experimental.area_reflection import AreaReflection
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import AreaReflectedLightCurve
from elisa.experimental.gpu_reference import cpu_reflect
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture
from scripts.validate_reflection_area import intrinsic_state

from .test_gpu_boundary import CASES

DEVICE = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("temperatures", [(4800, 6400), (6400, 4800), (5600, 5600)])
def test_collocated_limit_matches_original(temperatures):
    with table_settings(), settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW="logarithmic"):
        binary = fixture({"temperatures": list(temperatures), "discretization": 12, "inclination": 85, "law": "logarithmic"})
        system, _ = intrinsic_state(binary, 0)
        op = AreaReflection("logarithmic", device=DEVICE, receiver_points=1)
        for loops in (0, 1, 2, 3):
            expected = cpu_reflect(deepcopy(system), loops)
            temps, stats = op.evaluate(binary, system, 0, loops)
            for name, x in temps.items():
                assert_allclose(x, getattr(expected, name).temperatures, rtol=3e-13)
                assert stats[name]["donor_faces"] == stats[name]["donor_groups"]


def test_area_protocol_guards(monkeypatch):
    with table_settings():
        binary = fixture({"discretization": 12, "inclination": 85})
        system, _ = intrinsic_state(binary, 0)
        op = AreaReflection(settings.LIMB_DARKENING_LAW, device="reference")
        monkeypatch.setattr(area_reflection, "_MAX_PAIRS", 1)
        with pytest.raises(UnsupportedReflectionError, match="pair cap"):
            op.evaluate(binary, system, 0, 2)
        with pytest.raises(ValueError, match="one or three"):
            AreaReflection("linear", device=DEVICE, receiver_points=2)
        with pytest.raises(UnsupportedReflectionError):
            op.evaluate(binary, system, 0, 33)


@pytest.mark.parametrize("case", CASES)
def test_complete_reflection_quadrature(case):
    with (
        table_settings(),
        settings.override(
            REFLECTION_EFFECT=True,
            REFLECTION_EFFECT_ITERATIONS=2,
            LIMB_DARKENING_LAW=case.get("law", "linear"),
            USE_SINGLE_LD_COEFFICIENTS=False,
        ),
    ):
        binary = (table_fixture if case.get("atlas") else fixture)({"discretization": 12, **case})
        bands = ["Generic.Bessell.V", "bolometric"]
        outputs = [
            AreaReflectedLightCurve(
                bands, CurvedEdgeRule(3, 1), device=d, levels=(0, 1), integration_level=1, donor_level=0
            )(binary, [0.017, 0.027, 0.25, 0.517, 0.533, 1.027], source_revision="fixture")
            for d in (DEVICE, "reference")
        ]
        a, r = outputs
        assert a.diagnostics["phase_records"] == r.diagnostics["phase_records"]
        for v in r.curves:
            assert a.curves[v].rule == r.curves[v].rule
            for name in ("primary", "secondary"):
                x, y = [m.diagnostics["ancestry_controls"][v][name]["reconstructed_field"] for m in outputs]
                for k in ("receiver_samples", "donor_faces", "donor_groups", "pairs_per_sweep_both_stars"):
                    assert x[k] == y[k]
                assert (x["donor_groups"] < x["donor_faces"]) == (r.curves[v].field_levels > 0)
                assert x["cell_T4_relative_error"] < 3e-13
                assert_allclose(
                    x["deposited_T4_area_mean_over_Teff4"], y["deposited_T4_area_mean_over_Teff4"], rtol=3e-13
                )
            for attr in ("component_fluxes", "raw_fluxes", "fluxes"):
                for band in bands:
                    assert_allclose(
                        getattr(a.curves[v], attr)[band],
                        getattr(r.curves[v], attr)[band],
                        rtol=3e-13,
                        atol=2e-12 if attr == "fluxes" else 0,
                    )
