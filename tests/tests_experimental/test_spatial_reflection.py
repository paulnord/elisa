"""Check the grouped position derivative against explicitly displaced native faces."""

import os

import numpy as np
import pytest

from elisa import settings
from elisa.binary_system.surface.faces import faces_visibility_x_limits
from elisa.experimental import spatial_reflection
from elisa.experimental.spatial_reflection import SpatialReflection, _layout
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from scripts.validate_donor_positions import evaluate_samples
from scripts.validate_emitter_groups import incident
from scripts.validate_gpu_atmosphere import fixture, table_settings
from scripts.validate_reflection_area import intrinsic_state


def test_spatial_update_matches_native_expansion(monkeypatch):
    # Exercise both complete tiles and a trailing partial tile on the real crossing geometry.
    monkeypatch.setattr(spatial_reflection, "_HORIZON_TILE", 37)
    with table_settings(), settings.override(LIMB_DARKENING_LAW="linear", USE_SINGLE_LD_COEFFICIENTS=False):
        binary = fixture({"inclination": 85})
        system, _ = intrinsic_state(binary, 1)
        solver = SpatialReflection("linear", device=os.getenv("ELISA_GEOMETRY_TEST_DEVICE", "cpu"), donor_level=0)
        names = ("primary", "secondary")
        limits = faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
        native = [solver._surface(binary, system, n, 1, limits[n]) for n in names]
        grouped, details = solver._prepare_pair(native, system, limits, 1)
        for i, g in enumerate(grouped):
            r, d = native[1 - i], native[i]
            rows = np.arange(len(r.parent))
            assert details[i]["receiver_horizon_groups"] > 0
            with pytest.raises(UnsupportedReflectionError, match="pair cap"):
                _layout(r, d, g, details[i]["extra_native_pairs"] - 1)
            for factor in (0, 0.09):
                heat = d.intrinsic * factor * (1 + 0.6 * np.sin(np.arange(len(d.intrinsic))))
                errors, count = evaluate_samples(r, d, g, rows, solver.law, heat)
                assert count <= details[i]["receiver_horizon_groups"]
                light = (d.intrinsic + heat)[d.visible_base] * d.visible_area
                direct = incident(r, rows, d.emitter, d.outward, light, d.prefix_ld)
                oracle = direct + errors["first_order_with_resolved_horizon"]
                expected = np.bincount(r.parent, weights=oracle * r.fraction, minlength=len(r.intrinsic))
                actual = solver._update(*(solver.jax.device_put(a, solver.device) for a in (r, g, heat)))
                for result in (actual, solver._reference_update(r, g, heat)):
                    np.testing.assert_allclose(result, expected, rtol=3e-13, atol=1e-15 * np.max(r.intrinsic))
                assert np.linalg.norm(oracle - direct) < np.linalg.norm(errors["shared_position"])
