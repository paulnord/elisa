"""Independent receiver slots, quadrature means, limiter and reflected eclipse checks."""

import os

import numpy as np
import pytest

from elisa import settings
from elisa.binary_system.surface import faces
from elisa.experimental import area_reflection
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.quadratic_reflection import QuadraticReflection
from elisa.experimental.quadratic_source import receiver_slope
from elisa.experimental.receiver_heating import ReceiverHeatingLightCurve, SampledReceiverReflection
from scripts.validate_emitting_cell_area import projected_rule
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture
from scripts.validate_reflection_area import NAMES, intrinsic_state
from .test_quadratic_reflection import expanded


DEVICE = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")


def test_conservative_linear_reconstruction():
    mass = np.array([[0.3, 0.33, 0.37]] * 3)
    corners = np.array([[2.0, 3.0, 4.0], [-1.0, 1.0, 3.0], [0.0, 0.0, 0.0]])
    mean = np.sum(mass * corners, axis=1)
    samples = corners @ area_reflection._BARY.T
    slope, limiter = receiver_slope(np, samples, mean, mass)
    fitted = mean[:, None] + slope
    np.testing.assert_allclose(fitted[[0, 2]], corners[[0, 2]], atol=1e-15)
    np.testing.assert_allclose(np.sum(mass * fitted, axis=1), mean, atol=1e-15)
    assert 0 < limiter[1] < 1 and fitted.min() >= -2e-15


@pytest.mark.parametrize("temperatures", ([4900, 6400], [6500, 5000]))
def test_native_slots_at_last_cooler_first_update(temperatures, monkeypatch):
    with table_settings(), settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW="linear"):
        binary = fixture(dict(discretization=14, temperatures=temperatures, inclination=86))
        system, tri = intrinsic_state(binary, 1)
        limits = faces.faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
        oracle = QuadraticReflection("linear", device="reference", donor_level=0)
        src = [oracle._surface(binary, system, n, 1, limits[n]) for n in NAMES]
        pair, _ = oracle._prepare_pair(src, system, limits, 1)
        heat = [np.zeros_like(s.intrinsic) for s in pair]
        expected = {}
        for _ in range(2):
            for i in np.argsort(temperatures):
                r, donor = pair[i], pair[1 - i]
                name = NAMES[i]
                n = len(r.intrinsic)
                positions, _, frac = projected_rule(binary, name, tri[name][:n], 3)
                keep = faces.visibility_test(positions, limits[name], name)
                rows = np.arange(len(r.parent))
                native = expanded(
                    r._replace(
                        parent=rows,
                        fraction=np.ones(len(rows)),
                        intrinsic=r.intrinsic[r.parent],
                        absorption=r.absorption[r.parent],
                    ),
                    donor,
                    heat[1 - i],
                )
                full = np.zeros(3 * n)
                full[keep] = native
                expected[name] = full.reshape(n, 3)
                heat[i] = np.sum(frac * expected[name], axis=1)
        for device in ("reference", DEVICE):
            solver = SampledReceiverReflection("linear", device=device, donor_level=0)
            temperatures_out, stats = solver.evaluate(binary, system, 1, 2)
            for i, name in enumerate(NAMES):
                star = getattr(system, name)
                samples, mean = stats[name]["receiver_deposits"]
                np.testing.assert_allclose(samples, expected[name] / star.t_eff**4, rtol=3e-13, atol=4e-16)
                np.testing.assert_allclose(mean, heat[i] / star.t_eff**4, rtol=3e-13, atol=4e-16)
                np.testing.assert_allclose(
                    temperatures_out[name], (pair[i].intrinsic + heat[i])[star.face_symmetry_vector] ** 0.25, rtol=3e-13
                )
                counts = stats[name]["receiver_samples_per_face"]
                assert sum(counts) == len(mean) and counts[0] and counts[3]
        monkeypatch.setattr(area_reflection, "_MAX_PAIRS", stats["primary"]["pairs_per_sweep_both_stars"] - 1)
        with pytest.raises(UnsupportedReflectionError, match="pair cap"):
            solver.evaluate(binary, system, 1, 2)


def test_eclipse_with_sampled_heating_and_refined_shadow():
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        binary = fixture(dict(discretization=14, inclination=89, temperatures=[5000, 6400]))
        bands = ["Generic.Bessell.B", "bolometric"]
        phases = [0.018, 0.525, 0.24, 1.018, 0.039]
        reference, device = [
            ReceiverHeatingLightCurve(
                bands,
                CurvedEdgeRule(3, 1, slice_order=12, junction_depth=4),
                device=b,
                levels=(0, 1),
                integration_level=1,
                selective_shadow=True,
            )(binary, phases, source_revision="test")
            for b in ("reference", DEVICE)
        ]
        assert reference.diagnostics["phase_records"] == device.diagnostics["phase_records"]
        for k, c in reference.curves.items():
            for band in bands:
                for attr in ("raw_fluxes", "component_fluxes", "fluxes"):
                    np.testing.assert_allclose(
                        getattr(c, attr)[band], getattr(device.curves[k], attr)[band], rtol=3e-13
                    )
