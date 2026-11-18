"""Compare every adaptive parent with its independently synthesized single-parent packet."""

from dataclasses import replace

import numpy as np
import pytest

from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_curvature_fields import curvature_geometry
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_reconstructed_observer import recorded_packet
from elisa.experimental.shadow_refinement import deeper_shadow_packet, shadow_parents
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture


def test_parent_selection_conservation_and_aggregate_cap():
    with table_settings():
        binary = fixture(dict(inclination=85, discretization=14))
        _, cells, _ = curvature_geometry(binary, 0)
        tri = cells["primary"]["triangles"]
        geom, front = [ConditionedRocheGeometry(binary, n) for n in ("primary", "secondary")]
        line = np.array([1.0, 0.1, 0.03])
        line /= np.linalg.norm(line)
        crossing = shadow_parents(tri, geom, line, front)
        # Select visible faces on both sides of the seven-probe decision.
        visible = geom.sample(tri.mean(axis=1))[1] @ line > 0.2
        indices = np.ravel(
            np.stack([np.flatnonzero(visible & crossing)[:5], np.flatnonzero(visible & ~crossing)[:5]], axis=1)
        )
        tri, crossing = tri[indices], crossing[indices]
        base = CurvedEdgeRule(3, 1, junction_depth=4, slice_order=12)
        packet = deeper_shadow_packet(tri, geom, line, base, front)
        original = [
            recorded_packet(t[None], geom, line, replace(base, depth=base.depth + int(c)), front)
            for t, c in zip(tri, crossing, strict=True)
        ]
        expected = [sum(p["mu"] * p["area"]) for p in original]
        actual = np.bincount(packet["parents"], weights=packet["mu"] * packet["area"], minlength=len(tri))
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-16)
        assert packet["shadow_crossing_parents"] == int(crossing.sum()) == 5
        assert packet["nodes"] == sum(p["nodes"] for p in original)
        with pytest.raises(UnsupportedCurvedGeometryError):
            deeper_shadow_packet(tri, geom, line, replace(base, max_nodes=packet["nodes"] - 1), front)
