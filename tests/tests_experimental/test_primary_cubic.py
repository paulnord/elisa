"""Independent polynomial, gravity-darkening and complete eclipse controls."""

import os

import numpy as np
import pytest

from elisa import settings
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
from elisa.experimental.gpu_reconstructed_field import shape_rule
from elisa.experimental.primary_cubic import (
    CUBIC_NODES,
    CubicPrimaryBackend,
    PrimaryCubicLightCurve,
    bernstein,
    cubic_density,
    lagrange,
)
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture


def test_cubic_fields_reproduce_all_simplex_monomials():
    rays = np.random.default_rng(1729).dirichlet(np.ones(3), size=80)
    np.testing.assert_allclose(lagrange(np, CUBIC_NODES), np.eye(10), atol=6e-16)
    for i in range(4):
        for j in range(4 - i):
            for k in range(4 - i - j):
                powers = [i, j, k]
                values = np.prod(CUBIC_NODES**powers, axis=1)
                expected = np.prod(rays**powers, axis=1)
                controls = CubicPrimaryBackend.control_values(values[None])[0]
                for result in (lagrange(np, rays) @ values, bernstein(np, rays) @ controls):
                    np.testing.assert_allclose(result, expected, atol=2e-15, rtol=0)


def test_cubic_thermal_reproduces_physical_power_and_conserves_deposits():
    q, w = shape_rule()

    def g(b):
        return 1.1 + 0.3 * b[:, 0] ** 3 + 0.2 * b[:, 1] * b[:, 2]

    weight = np.stack((w * (1 + q[:, 1]), w * (1 + q[:, 2])))
    gq = np.stack((g(q), 1.2 * g(q)))
    gn = np.stack((g(CUBIC_NODES), 1.2 * g(CUBIC_NODES)))
    norm = weight.sum() / np.sum(weight * gq)
    avg = np.sum(weight * gq, axis=1) / weight.sum(axis=1) * norm
    source_rays = np.eye(3) * 0.5 + 1 / 6
    heat = np.array([[0.001, 0.008, 0.006], [0.004, 0.002, 0.001]])
    mean = np.sum(weight * (heat @ q.T), axis=1) / weight.sum(axis=1)
    vals, fitted, limited = cubic_density(np, weight, gq, gn, 1.0, avg, (heat @ source_rays.T, mean))
    np.testing.assert_allclose(vals, norm * gn + heat @ CUBIC_NODES.T, rtol=2e-15)
    np.testing.assert_allclose(fitted, avg + mean, rtol=2e-15)
    assert limited == 0


def test_primary_scope_and_intensity_positivity():
    with pytest.raises(UnsupportedCurvedGeometryError, match="primary only"):
        CubicPrimaryBackend("secondary", "linear", device="reference")
    b = CubicPrimaryBackend("primary", "linear", device="reference")
    tri = np.tile(np.eye(3), (4, 1, 1))
    values = np.zeros((1, 10, 1, 2))
    values[..., 0] = 1.0
    values[:, 9, :, 0] = 0.01
    with pytest.raises(UnsupportedCurvedGeometryError, match="nonnegative intensity"):
        b.optical_field(tri, values, ["bolometric"], {})


def test_reflected_cubic_physical_primary_eclipse():
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        model = fixture(dict(discretization=14, inclination=86, temperatures=[6500, 5000]))
        bands = ["Generic.Bessell.V", "bolometric"]
        phases = [0.024, 0.027, 0.25, 0.527, 1.024]
        curves = [
            PrimaryCubicLightCurve(
                bands, SymmetricInteriorRule(3, 1), device=b, levels=(0, 1), integration_level=1, selective_shadow=True
            )(model, phases, source_revision="test")
            for b in ("reference", os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu"))
        ]
        ref, run = curves
        assert ref.diagnostics["phase_records"] == run.diagnostics["phase_records"]
        for key, x in ref.curves.items():
            y = run.curves[key]
            assert x.rule == y.rule and any("primary_cubic" in k for k in x.rule)
            controls = run.diagnostics["ancestry_controls"][key]
            for n, slots in (("primary", 10), ("secondary", 6)):
                f = controls[n]["reconstructed_field"]
                assert f["quarter_spectral_samples"] == slots * (f["source_faces"] // 4)
                assert f["cell_T4_relative_error"] <= 3e-13
            for band in bands:
                for attr in ("raw_fluxes", "component_fluxes", "fluxes"):
                    np.testing.assert_allclose(getattr(x, attr)[band], getattr(y, attr)[band], rtol=3e-13)


def test_cubic_temperature_inputs_and_cap():
    from scripts.validate_reflection_area import intrinsic_state

    with table_settings():
        model = fixture(dict(discretization=14, inclination=86))
        system, _ = intrinsic_state(model, 0)
        star = system.primary
        backend = CubicPrimaryBackend("primary", "linear", device="reference")
        original = star.temperatures.copy()
        for bad in (-original, original[:-1]):
            with pytest.raises(UnsupportedCurvedGeometryError, match="intrinsic temperatures"):
                backend.emission_nodes(model, star, bad)
        size = star.base_symmetry_faces_number
        with pytest.raises(UnsupportedCurvedGeometryError, match="receiver samples"):
            backend.emission_nodes(model, star, original, deposits=(np.zeros((size, 2)), np.zeros(size)))
        with pytest.raises(UnsupportedCurvedGeometryError, match="reflected face"):
            backend.emission_nodes(model, star, original, deposits=(np.zeros((size, 3)), np.full(size, 0.01)))
        backend.max_nodes = 26 * size - 1
        with pytest.raises(UnsupportedCurvedGeometryError, match="budget"):
            backend.emission_nodes(model, star, original)


def test_shared_packets_reproduce_standalone_quadratic_physics():
    from elisa.experimental.receiver_heating import ReceiverHeatingLightCurve
    from scripts.check_primary_curve_pair import PairedPrimary

    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        model = fixture(dict(discretization=14, inclination=86, temperatures=[6500, 5000]))
        opts = dict(levels=(0, 1), integration_level=1, device="reference", selective_shadow=True)
        phases = [.024, .027, .25]
        paired = PairedPrimary(["bolometric"], SymmetricInteriorRule(3, 1), **opts)
        result = paired(model, phases, source_revision="test")
        baseline = ReceiverHeatingLightCurve(["bolometric"], paired.rule, **opts)(model, phases, source_revision="test")
        assert result.quadrature_sha256 == baseline.quadrature_sha256
        for key, curve in baseline.curves.items():
            actual = np.array([x["bolometric"] for x in paired.quadratic_fluxes[key]])
            np.testing.assert_allclose(actual, curve.component_fluxes["bolometric"], rtol=3e-13)
            assert np.any(actual[:, 0] != result.curves[key].component_fluxes["bolometric"][:, 0])
