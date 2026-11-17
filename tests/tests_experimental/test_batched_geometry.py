"""Independent original-ELISa geometry controls for a batched device port."""

import os
from functools import lru_cache
from typing import Any

import numpy as np
import pytest

from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_batched_geometry import BatchedRocheGeometry
from scripts.batched_geometry_cases import triangles
from scripts.validate_gpu_binary import fixture

DEVICE = os.environ.get("ELISA_GEOMETRY_TEST_DEVICE", "cpu")

CASES = [
    (0.5, 0.5, 1.5, 8.0, 12.0),
    (0.7, 1.0, 1.0, 8.0, 12.0),
    (1.0, 1.0, 1.0, 12.0, 12.0),
    (2.0, 1.5, 0.5, 12.0, 16.0),
    (0.8, 0.7, 1.3, 16.0, 18.0),
    (1.9, 1.4, 0.6, 20.0, 24.0),
    (0.7, 1.0, 1.0, 8.0, 100.0),
    (1.3, 0.8, 1.2, 100.0, 12.0),
]


@lru_cache(None)
def binaries() -> Any:
    return [
        fixture({"inclination": 88.0, "q": q, "spin": [s1, s2], "potentials": [p1, p2]}) for q, s1, s2, p1, p2 in CASES
    ]


def parameters(component: Any) -> Any:
    return np.array(
        [
            [b.mass_ratio, getattr(b, component).synchronicity, getattr(b, component).surface_potential]
            for b in binaries()
        ],
    )


def rays(count: Any=64) -> Any:
    vectors = np.random.default_rng(20261117).normal(size=(count, 3))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    return np.concatenate([np.eye(3), -np.eye(3), vectors])


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("changing_rays", [False, True])
def test_sample_against_original_cpu(component: Any, changing_rays: Any) -> None:
    engine = BatchedRocheGeometry(component, device=DEVICE)
    vectors = rays()
    batch = np.broadcast_to(vectors, (len(CASES), *vectors.shape)).copy()
    if changing_rays:
        for i in range(len(batch)):
            angle = 0.071 * i
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
            )
            batch[i] = batch[i] @ rotation.T * (1 + 0.01 * i)
    result = engine.checked_host(engine.sample(parameters(component), batch if changing_rays else vectors))
    for i, binary in enumerate(binaries()):
        reference = ConditionedRocheGeometry(binary, component)
        points, normals, jacobian = reference.sample(batch[i])
        np.testing.assert_allclose(result["points"][i], points, rtol=3e-13, atol=64 * np.finfo(float).eps)
        np.testing.assert_allclose(result["normals"][i], normals, rtol=3e-13, atol=64 * np.finfo(float).eps)
        np.testing.assert_allclose(result["area_per_solid_angle"][i], jacobian, rtol=3e-13, atol=0.0)
        potential = reference.potential(result["points"][i])
        assert np.max(np.abs(potential / getattr(binary, component).surface_potential - 1)) < 3e-13
    assert np.all(result["status"] == 0)


@pytest.mark.parametrize("component", ["primary", "secondary"])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("level", [0, 1])
def test_curved_area_matches_same_original_quadrature(component: Any, order: Any, level: Any) -> None:
    engine = BatchedRocheGeometry(component, device=DEVICE)
    faces = triangles(level)
    result = engine.checked_host(engine.areas(parameters(component), faces, order=order))
    nodes, weights = duffy_nodes(faces, order)
    for i, binary in enumerate(binaries()):
        reference = ConditionedRocheGeometry(binary, component)
        jacobian = reference.sample(nodes)[2]
        expected = (weights * jacobian).reshape((len(faces), order**2)).sum(axis=1)
        np.testing.assert_allclose(result["areas"][i], expected, rtol=3e-13, atol=0.0)
    assert np.all(result["max_relative_potential_residual"] < 3e-13)


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_changing_triangles_and_singleton_equivalence(component: Any) -> None:
    engine = BatchedRocheGeometry(component, device=DEVICE)
    shared = triangles(1)
    faces = np.broadcast_to(shared, (len(CASES), *shared.shape)).copy()
    for i in range(len(faces)):
        faces[i] = np.roll(faces[i], i, axis=0)
    batch = engine.checked_host(engine.areas(parameters(component), faces, order=3))
    for i in (0, 3, 7):
        singleton = engine.checked_host(engine.areas(parameters(component)[i : i + 1], faces[i], order=3))
        np.testing.assert_allclose(batch["areas"][i], singleton["areas"][0], rtol=3e-13, atol=0.0)


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_padding_inactive_rays_and_faces(component: Any) -> None:
    engine = BatchedRocheGeometry(component, device=DEVICE)
    p = parameters(component)[:2]
    vectors = rays(12)
    padded = np.concatenate([vectors, np.zeros((3, 3))])
    mask = np.arange(len(padded)) < len(vectors)
    a = engine.checked_host(engine.sample(p, vectors))
    b = engine.checked_host(engine.sample(p, padded, active=mask))
    np.testing.assert_array_equal(a["points"], b["points"][:, : len(vectors)])
    assert np.all(b["points"][:, len(vectors) :] == 0)
    assert np.all(b["status"][:, len(vectors) :] == 7)
    faces = triangles()
    padded = np.concatenate([faces, np.zeros((2, 3, 3))])
    mask = np.arange(len(padded)) < len(faces)
    a = engine.checked_host(engine.areas(p, faces))
    b = engine.checked_host(engine.areas(p, padded, active=mask))
    np.testing.assert_allclose(a["areas"], b["areas"][:, : len(faces)], rtol=3e-13, atol=0.0)
    assert np.all(b["areas"][:, len(faces) :] == 0)


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_unsupported_rows_are_not_prior_values(component: Any) -> None:
    engine = BatchedRocheGeometry(component, device=DEVICE)
    p = parameters(component)[:5].copy()
    p[0, 0] = 2.01
    p[1, 1] = 1.51
    p[2, 2] = np.nan
    p[3, 2] = 1.0
    result = engine.sample(p, rays())
    host = engine.jax.tree.map(np.asarray, result)
    assert not host["proposal_valid"][:4].any()
    assert host["proposal_valid"][4]
    assert np.all(host["status"][:3] == 1)
    assert np.all(host["status"][3] == 3)
    assert not np.isneginf(host["radii"]).any()
    with pytest.raises(UnsupportedCurvedGeometryError, match="no prior penalty"):
        engine.checked_host(result)


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_endpoint_ambiguity_and_bad_active_geometry(component: Any) -> None:
    engine = BatchedRocheGeometry(component, device=DEVICE)
    p = np.array([[1.0, 1.0, 5.0]])
    # x=y=0, z=.25: both equal-mass components share this target.
    p[:, 2] = 1 / 0.25 + 1 / np.sqrt(1 + 0.25**2)
    host = engine.jax.tree.map(np.asarray, engine.sample(p, np.array([[0.0, 0.0, 1.0]])))
    assert not host["proposal_valid"].any()
    assert host["status"][0, 0] == 4
    host = engine.jax.tree.map(np.asarray, engine.sample(np.array([[1.0, 1.0, 12.0]]), np.zeros((2, 3))))
    assert np.all(host["status"] == 2)
    with pytest.raises(UnsupportedCurvedGeometryError):
        engine.checked_host(engine.areas(np.array([[1.0, 1.0, 12.0]]), np.ones((1, 3, 3))))


def test_shape_capacity_fp64_and_no_host_callbacks() -> None:
    engine = BatchedRocheGeometry("primary", device=DEVICE, max_nodes=32)
    p = np.array([[1.0, 1.0, 12.0]])
    for vector in [np.ones(3), np.ones((2, 2)), np.ones((2, 2, 3))]:
        with pytest.raises(ValueError, match="shape"):
            engine.sample(p, vector)
    with pytest.raises(ValueError, match="cap"):
        engine.sample(p, np.ones((33, 3)))
    with pytest.raises(ValueError, match="cap"):
        engine.areas(p, triangles(), order=4)
    with pytest.raises(ValueError, match="order"):
        engine.areas(p, triangles(), order=True)
    result = engine.sample(p, rays(2))
    assert all(hasattr(value, "device") for value in result.values())
    graph = str(
        engine.jax.make_jaxpr(engine.sample_kernel)(
            engine.jnp.asarray(p), engine.jnp.ones((1, 2, 3)), engine.jnp.ones((1, 2), dtype=bool),
        ),
    )
    assert "pure_callback" not in graph
    assert "io_callback" not in graph
    with engine.jax.enable_x64(False), pytest.raises(RuntimeError, match="FP64"):
        engine.sample(p, rays(2))


@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_compact_and_resident_consumer_use_fresh_physics(component: Any) -> None:
    engine = BatchedRocheGeometry(component, device=DEVICE)
    p = parameters(component)
    vectors = np.broadcast_to(rays(12), (len(p), 18, 3)).copy()
    compact = engine.checked_host(engine.sample(p, vectors, compact=True))
    complete = engine.checked_host(engine.sample(p, vectors))
    for key in compact:
        np.testing.assert_allclose(compact[key], complete[key], rtol=3e-13, atol=64 * np.finfo(float).eps)

    # A downstream calculation consumes device geometry directly in one JIT;
    # this sum is a composition control, not a light curve or likelihood.
    def consumer(params: Any, rays: Any, mask: Any) -> Any:
        values = engine.sample_kernel(params, rays, mask)
        return engine.jnp.sum(values["area_per_solid_angle"], axis=1), values["proposal_valid"]

    compiled = engine.jax.jit(consumer)
    first = np.asarray(
        compiled(engine.jnp.asarray(p), engine.jnp.asarray(vectors), engine.jnp.ones(vectors.shape[:2], dtype=bool))[0],
    )
    changed = p.copy()
    changed[:, 2] += 0.2
    second = np.asarray(
        compiled(
            engine.jnp.asarray(changed), engine.jnp.asarray(vectors), engine.jnp.ones(vectors.shape[:2], dtype=bool),
        )[0],
    )
    assert np.all(first != second)
