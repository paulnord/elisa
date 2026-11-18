"""Conservation and interpolation controls independent of the stellar eclipse calculation."""

import numpy as np
import pytest

from elisa.experimental.area_reflection import _BARY
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.receiver_curvature import curvature_stencil, heat_basis, heat_controls


def patch():
    # A regular refined octahedron provides a closed surface including symmetry seams.
    from scipy.spatial import ConvexHull

    vertices = np.vstack((np.eye(3), -np.eye(3)))
    indices = ConvexHull(vertices).simplices
    directions = [*vertices]
    midpoints = {}
    children = []
    for a, b, c in indices:
        sides = []
        for left, right in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((left, right)))
            if edge not in midpoints:
                midpoints[edge] = len(directions)
                v = vertices[left] + vertices[right]
                directions.append(v / np.linalg.norm(v))
            sides.append(midpoints[edge])
        u, v, w = sides
        children.extend(((a, u, w), (b, v, u), (c, w, v), (u, v, w)))
    faces = np.array(children)
    return faces, np.array(directions)[faces]


def test_quadratic_heating_in_projective_coordinates():
    faces, tri = patch()
    stencil = curvature_stencil(faces, tri, len(faces))
    mass = np.array([0.12, 0.14, 0.18, 0.21, 0.16, 0.19])
    target = np.array([0.8, 1.3, 0.7, 0.43, 0.95, 1.17])
    rays = np.einsum("ij,fjk->fik", _BARY, tri)
    # Set native deposits by independent pointwise evaluation through the entire stencil.
    for f, neighbours in enumerate(stencil[0]):
        samples = np.zeros((len(faces), 3))
        for g in [f, *neighbours]:
            bary = np.linalg.solve(tri[f].T, rays[g].T).T
            bary /= bary.sum(axis=1, keepdims=True)
            samples[g] = heat_basis(np, bary) @ target
        controls, limited = heat_controls(np, samples, np.full(len(faces), mass @ target), stencil, mass)
        assert not limited[f]
        np.testing.assert_allclose(controls[f], target, rtol=3e-14)


def test_dawn_limiter_conserves_mean_and_nonnegative_heat():
    faces, tri = patch()
    stencil = curvature_stencil(faces, tri, len(faces))
    samples = np.maximum(0, np.einsum("ij,fj->fi", _BARY, tri[..., 0])) ** 5
    means = samples.mean(axis=1)
    mass = np.full(6, 1 / 6)
    controls, limited = heat_controls(np, samples, means, stencil, mass)
    assert np.any(limited) and np.all(controls >= 0)
    np.testing.assert_allclose(controls @ mass, means, atol=1e-16)
    np.testing.assert_array_equal(controls[means == 0], 0)
    with pytest.raises(UnsupportedCurvedGeometryError):
        curvature_stencil(faces[:-2], tri[:-2], len(faces) - 2)
