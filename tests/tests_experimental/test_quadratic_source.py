"""Physical energy, independent pointwise gravity, passband interpolation and eclipse controls."""

import os

import numpy as np
import pytest

from elisa import settings
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_nodal_source import _NodalBackend
from elisa.experimental.gpu_reconstructed_field import shape_rule
from elisa.experimental.quadratic_source import QuadraticSourceBackend, QuadraticSourceLightCurve
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture
from scripts.validate_reflection_area import intrinsic_state

DEVICE = os.environ.get("ELISA_GEOMETRY_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_intrinsic_quadratic_energy_and_accuracy(component):
    with table_settings():
        binary = fixture(dict(discretization=12, inclination=84, spin=[0.85, 1.15]))
        system, _ = intrinsic_state(binary, 1)
        star = getattr(system, component)
        intrinsic = star.temperatures.copy()
        backend = QuadraticSourceBackend(component, "linear", device=DEVICE)
        tri, values, logg, record = backend.six_nodes(binary, star, intrinsic)
        _, cpu, cpu_g, _ = QuadraticSourceBackend(component, "linear", device="reference").six_nodes(
            binary, star, intrinsic
        )
        np.testing.assert_allclose(values, cpu, rtol=3e-13)
        np.testing.assert_allclose(logg, cpu_g, rtol=0, atol=2e-13)
        base = len(values)
        geom = ConditionedRocheGeometry(binary, component)
        rays, weights = duffy_nodes(tri[:base], 4)
        points, _, jac = geom.sample(rays)
        g = np.linalg.norm(
            calculate_potential_gradient(1.0, component, points, star.synchronicity, binary.mass_ratio), axis=1
        )
        power = g.reshape(base, -1) ** star.gravity_darkening
        w = (weights * jac).reshape(base, -1)
        expected = power * w.sum() / np.sum(w * power)
        # Reconstruct by linear interpolation plus the three independent edge bubbles.
        coords = shape_rule()[0]
        t4 = (values / star.t_eff) ** 4
        predicted = t4[:, :3] @ coords.T
        for i in range(3):
            j = (i + 1) % 3
            excess = t4[:, 3 + i] - (t4[:, i] + t4[:, j]) / 2
            predicted += excess[:, None] * (4 * coords[:, i] * coords[:, j])
        np.testing.assert_allclose(
            (predicted * w).sum(axis=1) / w.sum(axis=1), (intrinsic[:base] / star.t_eff) ** 4, rtol=3e-13
        )
        linear = _NodalBackend(component, "linear", device="reference").thermal_nodes(binary, star, intrinsic)[1]
        old = (linear.reshape(-1, 3) / star.t_eff) ** 4 @ coords.T
        assert np.sum(w * (predicted - expected) ** 2) < np.sum(w * (old - expected) ** 2)
        assert record["quadratic_T4_relative_error"] <= 3e-13
        with pytest.raises(UnsupportedCurvedGeometryError):
            QuadraticSourceBackend(component, "linear", device=DEVICE, max_nodes=4).six_nodes(binary, star, intrinsic)


def test_quadratic_optical_polynomial_and_bounds():
    tri = np.tile(np.eye(3)[None], (4, 1, 1))
    coords = np.vstack((np.eye(3), (np.eye(3) + np.roll(np.eye(3), -1, axis=0)) / 2))

    def polynomial(c):
        return 2 + c[:, 0] ** 2 + 0.4 * c[:, 1] * c[:, 2]

    rad = polynomial(coords)
    samples = np.stack((rad, 0.4 * rad), axis=-1)[None, :, None]
    directions = np.array([[0.13, 0.74, 0.13], [0.25, 0.15, 0.60], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0]])
    mu = np.array([0.5, 0.1, 0.8, 1.0])
    areas = np.array([2.0, 1.0, 4.0, 1.5])
    for device in ("reference", DEVICE):
        b = QuadraticSourceBackend("primary", "linear", device=device)
        f = b.optical_field(tri, samples, ("V",), {})
        np.testing.assert_allclose(f.sample(np.arange(4), directions)[:, 0, 0], polynomial(directions), rtol=3e-13)
        expected = np.sum(mu * areas * polynomial(directions) * (0.6 + 0.4 * mu))
        np.testing.assert_allclose(f.flux(np.arange(4), directions, mu, areas), [expected], rtol=3e-13)
        bad = samples.copy()
        bad[:, 3:] *= 0.01
        with pytest.raises(UnsupportedCurvedGeometryError, match="nonnegative"):
            b.optical_field(tri, bad, ("V",), {})
        with pytest.raises(UnsupportedCurvedGeometryError, match="linear limb"):
            QuadraticSourceBackend("primary", "logarithmic", device=device)


@pytest.mark.parametrize("inclination", [78.0, 90.0])
def test_complete_reflected_contacts(inclination):
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        binary = fixture(dict(discretization=14, inclination=inclination, temperatures=[4900, 6400]))
        phases = [0.017, 0.027, 0.065, 0.267, 0.517, 0.547, 1.027]
        bands = ["Generic.Bessell.V", "bolometric"]
        rule = CurvedEdgeRule(3, 1, slice_order=12, junction_depth=4)
        cpu, device = [
            QuadraticSourceLightCurve(bands, rule, device=d, levels=(0, 1), integration_level=1)(
                binary, phases, source_revision="fixture"
            )
            for d in ("reference", DEVICE)
        ]
        assert cpu.diagnostics["phase_records"] == device.diagnostics["phase_records"]
        for k, curve in cpu.curves.items():
            for band in bands:
                for attr in ("component_fluxes", "raw_fluxes", "fluxes"):
                    np.testing.assert_allclose(
                        getattr(curve, attr)[band],
                        getattr(device.curves[k], attr)[band],
                        atol=2e-12 if attr == "fluxes" else 0,
                        rtol=3e-13,
                    )
