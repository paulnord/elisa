"""Separate contributions on a nonuniform observed grid including an obscured primary."""

from dataclasses import replace
import numpy as np
import pytest

from elisa.experimental.gpu_curved import CurvedCurveResult
from scripts.compare_primary_heating import field_contrast


@pytest.mark.parametrize("component", ["intrinsic", "face_mean", "within_face", "secondary"])
def test_isolated_resolution_change(component):
    phase = np.array([0.0, 0.04, 0.18, 0.43])
    intrinsic = np.array([0.0, 1.7, 2.0, 1.9])
    secondary = np.array([0.6, 0.65, 0.62, 0.58])
    shift = np.array([0.0, 0.03, -0.02, 0.01])
    lo = CurvedCurveResult(phase, {}, {}, {"V": np.column_stack((intrinsic, secondary))}, "fixed", {}, 2, 0.13, {}, {})
    high = {name: intrinsic.copy() for name in ("intrinsic", "mean", "full")}
    if component != "secondary":
        for name in {"intrinsic": high.keys(), "face_mean": ("mean", "full"), "within_face": ("full",)}[component]:
            high[name] += shift
    high_secondary = secondary + (shift if component == "secondary" else 0)
    hi = replace(lo, field_levels=3, component_fluxes={"V": np.column_stack((high["full"], high_secondary))})
    control = {
        2: {mode: {"V": intrinsic} for mode in ("mean", "intrinsic")},
        3: {mode: {"V": high[mode]} for mode in ("mean", "intrinsic")},
    }
    reducer = [2, 0, 1, 2, 3, 1]
    sigma = np.linspace(0.009, 0.012, len(reducer))
    data = np.array([1.12, 0.48, 0.90, 1.14, 1.0, 0.92])

    def observed(curve):
        light = curve.component_fluxes["V"].sum(axis=1)
        light += light.mean() * 0.13 / 0.87
        sampled = light[reducer]
        return sampled / sampled.mean()

    low, fine = observed(lo), observed(hi)
    flux = (fine - low) / sigma
    density = 0.5 * (((data - low) / sigma) ** 2 - ((data - fine) / sigma) ** 2)
    result = field_contrast(lo, hi, control, {"V": reducer}, {"V": sigma}, {"V": data})
    np.testing.assert_array_equal([v["phase"] for v in result["observations"]], phase[reducer])
    for name in ("intrinsic", "face_mean", "within_face", "secondary", "total"):
        changes = result["changes"][name]
        active = name in (component, "total")
        np.testing.assert_allclose(changes["flux_change_sigma"], flux if active else 0, rtol=0, atol=5e-14)
        np.testing.assert_allclose(
            changes["log_likelihood_by_observation"], density if active else 0, rtol=0, atol=3e-12
        )


def test_phase_block_preserves_shift_and_order():
    from types import SimpleNamespace
    from scripts.receiver_phase_parts import pack_block

    phases = np.array([0.012, 0.63, 1.012])
    indices = np.array([2, 0])
    shift = 0.15
    returned = (phases[indices] - shift) + shift
    physical = SimpleNamespace(
        phases=returned, component_fluxes={}, rule={}, signature="fixture", field_levels=3, additional_light=0.1
    )
    batch = SimpleNamespace(timings={})
    result = SimpleNamespace(
        curves={"fine": physical},
        diagnostics={"ancestry_controls": {}, "phase_records": []},
        quadrature_sha256="packet",
        timings={},
    )
    packed = pack_block("fixture", "reference", 4, 0, indices, phases, {}, shift, result, batch)
    np.testing.assert_array_equal(packed["input_phases"], phases[indices])
    physical.phases = returned[::-1]
    with pytest.raises(AssertionError):
        pack_block("fixture", "reference", 4, 0, indices, phases, {}, shift, result, batch)


def test_unheated_primary_disk_with_reflected_secondary():
    from elisa import settings
    from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
    from scripts.measure_primary_heating import ProjectedHeating
    from scripts.validate_gpu_atmosphere import table_settings
    from scripts.validate_gpu_binary import fixture

    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        model = fixture(dict(discretization=14, inclination=86))
        model.primary.albedo = 0.0
        solver = ProjectedHeating(
            ["bolometric"],
            SymmetricInteriorRule(3, 1),
            levels=(0, 1),
            integration_level=1,
            selective_shadow=True,
            device="reference",
        )
        result = solver(model, [0.024, 0.25], source_revision="zero-primary-albedo")
        for key, records in solver.disk_flux.items():
            source = result.diagnostics["ancestry_controls"][key]
            assert source["primary"]["reconstructed_field"]["deposited_T4_area_mean_over_Teff4"] == 0
            assert source["secondary"]["reconstructed_field"]["deposited_T4_area_mean_over_Teff4"] > 0
            assert len(records) == 2
            for record in records:
                for mode in ("intrinsic", "mean"):
                    np.testing.assert_allclose(record[mode], record["full"], rtol=3e-13, atol=0)


def test_common_parent_uses_solid_angle_weights_for_spherical_faces():
    from types import SimpleNamespace
    from scripts.compare_common_parent_heat import HeatField, means_on_common_faces

    a, b, c = np.eye(3)
    u, v, w = [(x + y) / np.linalg.norm(x + y) for x, y in ((a, b), (b, c), (c, a))]
    tri = np.array([[a, u, w], [b, v, u], [c, w, v], [u, v, w]])
    coarse = HeatField(np.eye(3)[None], np.ones(1), np.ones((1, 6)))
    delta = np.array([1.0, 2.0, 3.0, 4.0])
    fine = HeatField(tri, np.linalg.cond(tri), np.repeat(delta[:, None] + 1, 6, axis=1))
    sphere = SimpleNamespace(sample=lambda r: (None, None, np.ones(len(r))))
    mean, s = means_on_common_faces(coarse, fine, np.zeros(4, dtype=int), sphere, 10)
    area = np.array([2 * np.arctan2(abs(x @ np.cross(y, z)), 1 + x @ y + y @ z + z @ x) for x, y, z in tri])
    average = np.average(delta, weights=area)
    np.testing.assert_allclose(s["area"], np.pi / 2, rtol=5e-12)
    np.testing.assert_allclose(mean, [average], rtol=5e-12)
    np.testing.assert_allclose(s["within_rms"], np.sqrt(np.average((delta - average) ** 2, weights=area)), rtol=5e-12)
