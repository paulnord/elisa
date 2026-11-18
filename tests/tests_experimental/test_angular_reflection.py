"""Physical horizon, independently expanded directional emission, finite curves and limits."""

import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits
from elisa.experimental import area_reflection
from elisa.experimental.angular_reflection import AngularReflection, donor_layout
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import AngularReflectedLightCurve
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from scripts.validate_emitter_groups import incident
from scripts.validate_gpu_atmosphere import fixture as table_fixture, table_settings
from scripts.validate_gpu_binary import fixture
from scripts.validate_reflection_area import intrinsic_state
from .test_gpu_boundary import CASES

DEVICE = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")


def test_native_horizon_and_centroid_signs():
    pos = np.array([[0.0, 0.01, 0], [0.0, -0.01, 0]] * 4)
    n = np.array(
        [
            [1.0, 0, 0],
            [1.0, 0.1, 0],
            [-1.0, 0, 0],
            [-1.0, 0.1, 0],
            [0.0, 1.0, 0],
            [0.0, -1.0, 0],
            [1.0, 0, 0],
            [-1.0, 0, 0],
        ]
    )
    n /= np.linalg.norm(n, axis=1)[:, None]
    rows = np.array([[1.0, 0.2, 0], [1.0, -0.2, 0], [1.0, 0, 0]])
    group, centres, _, detail = donor_layout(pos, n, np.ones(8), np.repeat(np.arange(4), 2), rows)
    assert detail["positive_merged_groups"] == detail["negative_merged_groups"] == 1
    assert detail["split_horizon_groups"] == 2
    assert group[0] == group[1] and group[2] == group[3] and len(np.unique(group[4:])) == 4
    np.testing.assert_array_equal(centres[group[4:]], pos[4:])


def test_native_direction_sum_and_changing_luminosities(monkeypatch):
    with table_settings(), settings.override(LIMB_DARKENING_LAW="linear", USE_SINGLE_LD_COEFFICIENTS=False):
        binary = table_fixture({"inclination": 85})
        system, _ = intrinsic_state(binary, 1)
        op = AngularReflection("linear", device=DEVICE, donor_level=0)
        names = ("primary", "secondary")
        limits = faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
        native = [op._surface(binary, system, n, 1, limits[n]) for n in names]
        groups, details = op._prepare_pair(native, system, limits, 1)
        for i in range(2):
            receiver, d, g = native[i], native[1 - i], groups[1 - i]
            assert details[1 - i]["split_horizon_groups"] > 0 and len(g.emitter) < len(d.emitter)
            for factor in [0.0, 0.09]:
                heat = d.intrinsic * factor * (1 + 0.6 * np.sin(np.arange(len(d.intrinsic))))
                energy = (d.intrinsic + heat)[d.visible_base] * d.visible_area
                direct = incident(
                    receiver, np.arange(len(receiver.parent)), g.emitter[g.group], d.outward, energy, d.prefix_ld
                )
                expected = np.bincount(
                    receiver.parent, weights=direct * receiver.fraction, minlength=len(receiver.intrinsic)
                )
                got = op._update(*(op.jax.device_put(v, op.device) for v in (receiver, g, heat)))
                assert_allclose(
                    np.asarray(got), expected, rtol=3e-13, atol=1e-15 * getattr(system, names[i]).t_eff ** 4
                )
        monkeypatch.setattr(area_reflection, "_MAX_PAIRS", 1)
        with pytest.raises(UnsupportedReflectionError, match="pair cap"):
            op.evaluate(binary, system, 1, 2)


@pytest.mark.parametrize("case", CASES)
def test_complete_angular_curve(case):
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
        kw = dict(levels=(0, 1), integration_level=1, donor_level=0)
        data = []
        for device in ("reference", DEVICE):
            observer = AngularReflectedLightCurve(
                ["Generic.Bessell.V", "bolometric"], CurvedEdgeRule(3, 1), device=device, **kw
            )
            data.append(observer(binary, [0.017, 0.027, 0.25, 0.517, 0.533, 1.027], source_revision="fixture"))
        a, b = data
        assert a.diagnostics["phase_records"] == b.diagnostics["phase_records"]
        for key, c in a.curves.items():
            assert c.rule == b.curves[key].rule
            for name in ("primary", "secondary"):
                x, y = [v.diagnostics["ancestry_controls"][key][name]["reconstructed_field"] for v in data]
                for k in ("split_horizon_groups", "donor_groups", "pairs_per_sweep_both_stars"):
                    assert x[k] == y[k]
                if case.get("law") in ("logarithmic", "square_root"):
                    assert x["donor_groups"] == x["donor_faces"]
            for attr in ("component_fluxes", "raw_fluxes", "fluxes"):
                for band, val in getattr(c, attr).items():
                    assert_allclose(
                        val, getattr(b.curves[key], attr)[band], rtol=3e-13, atol=2e-12 if attr == "fluxes" else 0
                    )
