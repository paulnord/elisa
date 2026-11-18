"""Independent face-by-face expansion through order two and two complete irradiation sweeps."""

import os

import numpy as np
import pytest

from elisa import settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits
from elisa.experimental.quadratic_reflection import QuadraticReflection
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture
from scripts.validate_reflection_area import NAMES, intrinsic_state


def expanded(receiver, donor, heat):
    plan = donor.spatial.base
    group = donor.group
    a = donor.prefix_ld[0]
    normal, offset = donor.native_normals, plan.offset
    light = (donor.intrinsic + heat)[donor.visible_base] * donor.visible_area
    values = []
    for i, (point, h) in enumerate(zip(receiver.receiver, receiver.normal, strict=True)):
        ray = point - plan.centres[group]
        r = np.linalg.norm(ray, axis=1)
        u = ray / r[:, None]
        mu, b = -(u @ h), np.sum(normal * u, axis=1)
        t, nd = np.sum(u * offset, axis=1) / r, np.sum(normal * offset, axis=1) / r
        s2, hd = np.sum(offset**2, axis=1) / r**2, offset @ h / r
        linear0, linear1, linear2 = mu * b, hd * b - mu * nd, -hd * nd
        square0, square1, square2 = mu * b * b, hd * b * b - 2 * mu * b * nd, mu * nd**2 - 2 * hd * b * nd
        approximation = (1 - a) * (
            linear0 + linear1 + 4 * t * linear0 + linear2 + 4 * t * linear1 + (12 * t * t - 2 * s2) * linear0
        )
        approximation += a * (
            square0 + square1 + 5 * t * square0 + square2 + 5 * t * square1 + (17.5 * t * t - 2.5 * s2) * square0
        )
        lit = np.unpackbits(plan.lit[i], bitorder="little")[group].astype(bool)
        first, last = np.searchsorted(plan.horizon_row, [i, i + 1])
        native = plan.horizon_face[first:last]
        v = point - plan.native_positions[native]
        dist = np.linalg.norm(v, axis=1)
        v /= dist[:, None]
        angle = np.maximum(np.sum(normal[native] * v, axis=1), 0)
        exact = np.maximum(-(v @ h), 0) * angle * (1 - a[native] + a[native] * angle) / dist**2
        values.append(light[lit] @ (approximation[lit] / r[lit] ** 2) + light[native] @ exact)
    return receiver.absorption * np.bincount(
        receiver.parent, weights=receiver.fraction * values, minlength=len(receiver.intrinsic)
    )


@pytest.mark.parametrize("temperatures", ([4900, 6400], [6500, 5000]))
def test_quadratic_iteration(temperatures):
    device = os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu")
    with table_settings(), settings.override(REFLECTION_EFFECT=True, LIMB_DARKENING_LAW="linear"):
        binary = fixture(dict(discretization=14, temperatures=temperatures, inclination=86))
        system, _ = intrinsic_state(binary, 1)
        limits = faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
        oracle = QuadraticReflection("linear", device="reference", donor_level=0)
        source = [oracle._surface(binary, system, name, 1, limits[name]) for name in NAMES]
        surfaces, _ = oracle._prepare_pair(source, system, limits, 1)
        heat = [np.zeros_like(s.intrinsic) for s in surfaces]
        for _ in range(2):
            for i in (0, 1) if temperatures[0] < temperatures[1] else (1, 0):
                heat[i] = expanded(surfaces[i], surfaces[1 - i], heat[1 - i])
        for mode in ("reference", device):
            op = QuadraticReflection("linear", device=mode, donor_level=0)
            temps, stats = op.evaluate(binary, system, 1, 2)
            for i, name in enumerate(NAMES):
                star = getattr(system, name)
                expected = (surfaces[i].intrinsic + heat[i]) ** 0.25
                np.testing.assert_allclose(temps[name], expected[star.face_symmetry_vector], rtol=3e-13)
                measured = np.average(heat[i] / star.t_eff**4, weights=star.areas[: len(heat[i])])
                np.testing.assert_allclose(stats[name]["deposited_T4_area_mean_over_Teff4"], measured, rtol=3e-13)
