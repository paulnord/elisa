"""Regression tests for the opt-in fused shadow and independent injection evaluation."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from scripts.fused_stationary_shadow import FusedShadow, StationarySolve
from scripts.walker_roche_geometry import WalkerRocheGeometry


def binary():
    return SimpleNamespace(
        mass_ratio=0.8,
        eccentricity=0.0,
        morphology="detached",
        has_spots=lambda: False,
        has_pulsations=lambda: False,
        primary=SimpleNamespace(surface_potential=8.0, synchronicity=1.0),
        secondary=SimpleNamespace(surface_potential=14.0, synchronicity=0.95),
    )


def rays(geom):
    line = np.asarray([0.95, 0.2, 0.12])
    line /= np.linalg.norm(line)
    across = np.cross(line, [0, 0, 1])
    across /= np.linalg.norm(across)
    rho = np.r_[0.0, geom.inner, np.linspace(geom.inner, 0.21, 200), 0.25, 1.0]
    return geom.centre + np.outer(rho, across) + np.outer(np.full_like(rho, 0.82), line), line


@pytest.mark.parametrize("name", ["primary", "secondary"])
def test_uncertain_stationary_candidate_uses_reference(name):
    model = binary()
    fused = FusedShadow(model, name)
    points, line = rays(fused)
    fused.solver = SimpleNamespace(
        stats=dict(calls=0, reference_rays=0), audit=False, offsets=lambda base, line, g: np.full(len(base), np.nan)
    )
    expected = WalkerRocheGeometry(model, name).shadow(points, line)
    result = fused.shadow(points, line)
    assert 0 < fused.solver.stats["reference_rays"] < len(points)
    assert np.any(result[0]) and np.any(~result[0])
    for actual, ref in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual, ref)


def test_fused_geometry_preserves_unsupported_domain():
    model = binary()
    model.eccentricity = 0.1
    with pytest.raises(UnsupportedCurvedGeometryError):
        FusedShadow(model, "secondary")


@pytest.mark.skipif("ELISA_STATIONARY_LIBRARY" not in os.environ, reason="requires opt-in compiled solver")
@pytest.mark.parametrize("name", ["primary", "secondary"])
def test_native_stationary_margins(name):
    model = binary()
    backend = StationarySolve(Path(os.environ["ELISA_STATIONARY_LIBRARY"]), gpu=False, audit=True)
    try:
        fused = backend.geometry(model, name)
        points, line = rays(fused)
        with pytest.raises(ValueError, match="three Cartesian coordinates"):
            backend.offsets(points[:, :2], line, fused)
        a, b = fused.shadow(points, line)
        expected, distance = WalkerRocheGeometry(model, name).shadow(points, line)
        np.testing.assert_array_equal(a, expected)
        np.testing.assert_allclose(b, distance, atol=2e-12, rtol=3e-13)
        assert backend.stats["audited_rays"] > 0
        assert backend.stats["h2d_bytes"] == backend.stats["d2h_bytes"] == 0
    finally:
        backend.close()
    with pytest.raises(RuntimeError, match="Closed"):
        fused.shadow(points, line)


def test_resource_stops_must_match():
    from scripts.compare_stationary_eclipse import compare

    assert compare({"stops": [0, 1]}, {"stops": [0, 1]}) == dict(curves=0, stopped_parts=[0, 1])
    with pytest.raises(ArithmeticError, match="resource stops differ"):
        compare({"stops": [0]}, {"stops": []})


def test_shadow_demand_counts_all_nodes_before_applying_combined_cap(monkeypatch):
    from scripts import count_eclipse_nodes as counts
    from scripts.reemission_walker_fields import rule_for

    rule = rule_for("duffy", 4)
    # Each partition fits separately. The physical packet would exceed the bound when assembled.
    monkeypatch.setattr(counts, "shadow_parents", lambda *args: np.array([False, True, True]))

    def packet(faces, geometry, line, local_rule, foreground):
        assert len(faces) == (1 if local_rule.depth == rule.depth else 2)
        return {"nodes": rule.max_nodes // 2 + 1}

    monkeypatch.setattr(counts, "recorded_packet", packet)
    result = counts.shadow_demand(np.zeros((3, 3, 3)), None, None, rule, None)
    assert result["total"] == rule.max_nodes + 2
    assert result["crossing_parents"] == 2
    assert not result["fits"]
