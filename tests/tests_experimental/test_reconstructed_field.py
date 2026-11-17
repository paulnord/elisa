"""Moment conservation, shared-edge continuity, CPU physics and device parity."""

from __future__ import annotations

import os
from copy import deepcopy
from functools import lru_cache

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import settings
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule
from elisa.experimental.gpu_fused_observer import FusedObserverLightCurve
from elisa.experimental.gpu_reconstructed_field import MomentFieldBackend, incidence, project_moments
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_binary import fixture

DEVICE = os.environ.get("ELISA_GEOMETRY_TEST_DEVICE", "cpu")
BANDS = ("Generic.Bessell.V", "bolometric")


@lru_cache(None)
def source(law: str) -> tuple:
    with (
        table_settings(),
        settings.override(LIMB_DARKENING_LAW=law, REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2),
    ):
        binary = fixture({"inclination": 85, "discretization": 12, "law": law})
        result = FusedObserverLightCurve(
            list(BANDS),
            CurvedEdgeRule(3, 1),
            levels=(0,),
            integration_level=0,
            device="reference",
        )(binary, [0.25], source_revision="assets", diagnostics=True)
        return binary, result.diagnostics["initial_systems"]["curvature_field0"]


def test_positive_lumped_projection_preserves_all_moments_and_constants() -> None:
    faces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [0, 2, 3]])
    f, slots, active = incidence(faces)
    weights = np.random.default_rng(21).uniform(0.01, 0.2, (4, 3))
    moments = np.random.default_rng(22).normal(size=(4, 3, 3))
    vertex, mass = project_moments(weights, moments, slots, active)
    assert_allclose(
        (mass[:, None, None] * vertex).sum(axis=0),
        (weights.sum(axis=1)[:, None, None] * moments).sum(axis=0),
        rtol=3e-14,
        atol=3e-15,
    )
    assert np.all(vertex >= moments.min(axis=0) - 1e-14)
    assert np.all(vertex <= moments.max(axis=0) + 1e-14)
    constant = np.broadcast_to(np.array([[4.0, 1.0, -0.2]]), (4, 1, 3))
    result, _ = project_moments(weights, constant, slots, active)
    assert_allclose(result, constant, rtol=3e-15, atol=1e-15)
    assert np.array_equal(f, faces)


@pytest.mark.parametrize("law", ["linear", "logarithmic", "square_root"])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_physical_cell_projection_conservation_continuity_and_original_ld_flux(law: str, component: str) -> None:
    with table_settings(), settings.override(LIMB_DARKENING_LAW=law):
        binary, system = source(law)
        star = getattr(system, component)
        saved = deepcopy(star.normal_radiance)
        reference = MomentFieldBackend(component, law, device="reference").prepare(binary, star, BANDS)
        selected = MomentFieldBackend(component, law, device=DEVICE).prepare(binary, star, BANDS)
        assert reference.control["moment_integral_relative_error"] <= 3e-13
        assert selected.control["moment_integral_relative_error"] <= 3e-13
        assert selected.control["area_relative_error"] <= 3e-13
        assert_allclose(np.asarray(selected.vertex), reference.vertex, rtol=3e-13, atol=0)
        rays, _ = duffy_nodes(reference.triangles[:20], 3)
        parents = np.repeat(np.arange(20), 9)
        a, b = selected.sample(parents, rays), reference.sample(parents, rays)
        assert_allclose(a, b, rtol=3e-13, atol=0)
        mu = np.linspace(0.01, 1.0, len(rays))
        area = np.linspace(0.1, 0.2, len(rays))
        assert_allclose(
            selected.flux(parents, rays, mu, area),
            reference.flux(parents, rays, mu, area),
            rtol=3e-13,
            atol=0,
        )
        # A common edge has the SAME vertex field from either parent, independent
        # of incidence order or projected ray length. No nearest-face selection.
        edges = {}
        pair = None
        for i, face in enumerate(reference.faces):
            for j in range(3):
                edge = tuple(sorted([int(face[j]), int(face[(j + 1) % 3])]))
                if edge in edges:
                    pair = (edges[edge], i, edge)
                    break
                edges[edge] = i
            if pair is not None:
                break
        left, right, edge = pair
        lv = [list(reference.faces[left]).index(v) for v in edge]
        ray = 0.37 * reference.triangles[left, lv[0]] + 0.63 * reference.triangles[left, lv[1]]
        both = selected.sample(np.array([left, right]), np.array([ray, 3 * ray]))
        assert_allclose(both[0], both[1], rtol=3e-13, atol=0)
        for band in BANDS:
            assert_allclose(star.normal_radiance[band], saved[band], rtol=0, atol=0)


def test_cone_parent_capacity_empty_and_fp64_stops_are_explicit() -> None:
    with table_settings():
        binary, system = source("linear")
        star = system.primary
        backend = MomentFieldBackend("primary", "linear", device=DEVICE)
        prepared = backend.prepare(binary, star, BANDS)
        empty = prepared.flux(np.array([], dtype=int), np.empty((0, 3)), np.array([]), np.array([]))
        np.testing.assert_array_equal(empty, [0.0, 0.0])
        for parents, rays in [
            (np.array([len(star.faces)]), np.ones((1, 3))),
            (np.array([0]), -prepared.triangles[:1].sum(axis=1)),
            (np.array([0]), np.full((1, 3), np.nan)),
        ]:
            with settings.override(PENALIZE_ALL_EXCEPTIONS=True), pytest.raises(UnsupportedCurvedGeometryError):
                prepared.sample(parents, rays)
        with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
            MomentFieldBackend("primary", "linear", device=DEVICE, max_nodes=16).prepare(binary, star, BANDS)
        with backend.jax.enable_x64(False), pytest.raises(RuntimeError, match="FP64"):
            backend.prepare(binary, star, BANDS)
        with pytest.raises(ValueError, match="backend"):
            MomentFieldBackend("primary", "linear", device="automatic")
