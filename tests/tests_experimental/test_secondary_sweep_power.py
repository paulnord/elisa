"""Independently reconstruct the emitted heat at mirrored nodes during both reflection sweeps."""

import os

import numpy as np
import pytest

from elisa import settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits, visibility_test
from elisa.experimental.area_reflection import _BARY
from elisa.experimental.node_reemission import NodeReemissionReflection
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture
from scripts.validate_reflection_area import intrinsic_state


@pytest.mark.parametrize("device", ["reference", "accelerated"])
def test_secondary_preserves_face_power_in_each_sweep(device):
    device = os.environ.get("ELISA_GEOMETRY_TEST_DEVICE", "cpu") if device == "accelerated" else device
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        model = fixture(dict(discretization=14, inclination=86))
        system, _ = intrinsic_state(model, 0)
        solver = NodeReemissionReflection("linear", device=device, donor_level=1)
        order = sorted(("primary", "secondary"), key=lambda k: getattr(system, k).t_eff)
        calls = []
        original = solver._sampled_update

        def record(receiver, donor, deposited):
            value, mean = original(receiver, donor, deposited)
            calls.append((receiver, donor, np.asarray(deposited), np.asarray(value), np.asarray(mean)))
            return value, mean

        solver._sampled_update = record
        solver.evaluate(model, system, 0, 2)
        star = system.secondary
        origin = np.array([1., 0., 0.])
        vertices = star.points - origin
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
        xlim = faces_visibility_x_limits(system.primary.polar_radius, star.polar_radius, 1.)["secondary"]
        face_ids = np.flatnonzero(visibility_test(star.face_centres, xlim, "secondary"))
        parents = star.face_symmetry_vector[face_ids]
        base = vertices[star.faces[:star.base_symmetry_faces_number]]
        power = np.zeros((len(base), 3))
        means = np.zeros(len(base))
        for i, (receiver, donor, emitted, value, mean) in enumerate(calls):
            if order[i % 2] == "secondary":
                p = np.asarray(receiver.parent)
                coord = np.linalg.solve(base[p].transpose(0, 2, 1), (np.asarray(receiver.receiver)-origin)[..., None])[..., 0]
                coord /= coord.sum(axis=1, keepdims=True)
                slot = coord.argmax(axis=1)
                np.testing.assert_allclose(coord, _BARY[slot], rtol=0, atol=3e-12)
                power[p, slot] = value
                means = mean
            else:
                # The largest barycentric coefficient selects the corresponding triangle vertex.
                # Identify its base-quarter counterpart by reflected vertices independently of the ray map.
                full = abs(vertices[star.faces[face_ids]])
                dist = np.linalg.norm(full[:, :, None] - abs(base[parents, None]), axis=-1)
                slot = dist.argmin(axis=2)
                np.testing.assert_allclose(dist.min(axis=2), 0, rtol=0, atol=3e-12)
                np.testing.assert_array_equal(np.sort(slot, axis=1), np.tile(np.arange(3), (len(slot), 1)))
                np.testing.assert_array_equal(emitted.reshape(-1, 3), power[parents[:, None], slot])
                area = np.asarray(donor.visible_area).reshape(-1, 3)
                actual = (area * emitted.reshape(-1, 3)).sum(axis=1) / area.sum(axis=1)
                np.testing.assert_allclose(actual, means[parents], rtol=3e-13, atol=0)
        assert len(calls) == 4 and np.max(np.ptp(power, axis=1)) > 0
