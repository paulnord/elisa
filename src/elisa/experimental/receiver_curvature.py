"""Recover within-face heating curvature from the three adjacent faces' irradiation samples."""

import numpy as np

from elisa.experimental.area_reflection import _BARY
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.receiver_irradiation import adjacent_faces


def heat_basis(xp, bary):
    """Degree-two Bernstein basis, vertices followed by cyclic edges."""
    return xp.concatenate((bary**2, 2 * bary * xp.roll(bary, -1, axis=-1)), axis=-1)


_NATIVE_INVERSE = np.linalg.inv(_BARY)
_NATIVE_EDGES = heat_basis(np, _BARY)[:, 3:]
# An affine field has midpoint averages as its three Bernstein edge controls.
_AFFINE = np.vstack((np.eye(3), (np.eye(3) + np.roll(np.eye(3), -1, axis=0)) / 2)) @ _NATIVE_INVERSE
_EDGE = np.vstack((np.zeros((3, 3)), np.eye(3))) - _AFFINE @ _NATIVE_EDGES


def curvature_stencil(faces, triangles, base):
    """Express all nine exterior samples in the centre face's angular coordinates."""
    neighbours = adjacent_faces(faces, base)
    if not np.all(np.linalg.cond(triangles[:base]) <= 1e5):
        raise UnsupportedCurvedGeometryError("Ill-conditioned primary angular face")
    rays = np.einsum("ij,fkjl->fkil", _BARY, triangles[neighbours]).reshape(base, 9, 3)
    coordinates = np.linalg.solve(triangles[:base].transpose(0, 2, 1), rays.transpose(0, 2, 1)).transpose(0, 2, 1)
    total = coordinates.sum(axis=-1, keepdims=True)
    if not np.all(np.isfinite(total) & (total > 0)):
        raise UnsupportedCurvedGeometryError("Receiver neighbour lies outside the angular chart")
    bary = coordinates / total
    linear = bary @ _NATIVE_INVERSE
    design = heat_basis(np, bary)[..., 3:] - linear @ _NATIVE_EDGES
    condition = np.linalg.cond(design)
    if not np.all(np.isfinite(condition) & (condition <= 1e5)):
        raise UnsupportedCurvedGeometryError("Ill-conditioned quadratic receiver stencil")
    return neighbours % base, linear, np.linalg.pinv(design)


def heat_controls(xp, samples, means, stencil, mass):
    """Fit curvature, restore the original curved-area mean and certify nonnegative heat."""
    neighbours, linear, inverse = stencil
    # Differencing against a local constant keeps a uniformly heated face exact.
    constant = samples[:, :1]
    local = samples - constant
    surrounding = samples[neighbours].reshape(-1, 9) - constant
    residual = surrounding - xp.einsum("fij,fj->fi", linear, local)
    curvature = xp.einsum("fij,fj->fi", inverse, residual)
    values = local @ xp.asarray(_AFFINE.T) + curvature @ xp.asarray(_EDGE.T)
    offset = values - (mass * values).sum(axis=1, keepdims=True)
    drop = xp.maximum(0.0, -xp.min(offset, axis=1))
    limited = drop > means
    scale = xp.where(limited, means / xp.where(drop > 0, drop, 1.0), 1.0)
    # Blending about the mean leaves the deposited energy unchanged. Clipping
    # a final rounding ulp at zero does not change the prescribed limiter.
    return xp.maximum(0.0, means[:, None] + scale[:, None] * offset), limited
