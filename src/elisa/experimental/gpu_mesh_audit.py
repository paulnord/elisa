"""Independent CPU angular Roche area/volume diagnostics, not curve certificates.

Integrate a star-shaped implicit equipotential over dOmega = dmu dphi. The
original CPU analytic gradient gives its exact radial surface Jacobian. Angular
order comparisons are finite quadrature checks, separate from CPU/GPU parity,
triangulation, mesh correction and observational normalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from elisa import utils
from elisa.binary_system import model
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.binary_system.surface.mesh import _load_correction_factors
from elisa.experimental.gpu_nested_mesh import UnsupportedNestedMeshError

_MIN_ORDER, _MAX_ORDER = 4, 64
_Q_MIN, _Q_MAX, _SPIN_MIN, _SPIN_MAX = .5, 2., .5, 1.5


def angular_grid(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre mu and periodic midpoint phi, covering the full sphere."""
    if isinstance(order, bool) or not isinstance(order, int) or not _MIN_ORDER <= order <= _MAX_ORDER:
        msg = "Angular audit requires integer order4-64"
        raise ValueError(msg)
    mu, w = leggauss(order)
    phi = (np.arange(2*order)+.5)*np.pi/order
    z, p = np.broadcast_arrays(mu[:, None], phi[None])
    s = np.sqrt(1-z*z)
    directions = np.stack([s*np.cos(p), s*np.sin(p), z], axis=-1).reshape((-1, 3))
    return directions, np.repeat(w*np.pi/order, len(phi))


def radial_measures(directions: Any, radii: Any, outward_gradient: Any, weights: Any) -> dict[str, float]:
    """Independent radial volume/Jacobian formula, with star-shaped branch guard."""
    d, r, g, w = map(np.asarray, (directions, radii, outward_gradient, weights))
    norm = np.linalg.norm(g, axis=1)
    radial = np.einsum("ij,ij->i", g, d)
    if (not all(np.isfinite(v).all() for v in (d, r, g, w)) or np.any(r <= 0)
            or np.any(radial <= 0) or np.any(w <= 0)):
        msg = "Angular audit requires finite positive star-shaped radial geometry"
        raise UnsupportedNestedMeshError(msg)
    return {"area": float(np.sum(w*r*r*norm/radial)), "volume": float(np.sum(w*r**3/3)),
            "min_normal_dot_radial": float(np.min(radial/norm))}


def roche_measures(binary: Any, component: str, order: int) -> dict[str, Any]:
    """Original inner equipotential, Brent roots and independent CPU gradient."""
    if (component not in ("primary", "secondary") or binary.eccentricity != 0 or binary.morphology != "detached"
            or not _Q_MIN <= binary.mass_ratio <= _Q_MAX):
        msg = "Angular audit requires supported circular detached component/q"
        raise UnsupportedNestedMeshError(msg)
    star = getattr(binary, component)
    if binary.has_spots() or binary.has_pulsations() or not _SPIN_MIN <= star.synchronicity <= _SPIN_MAX:
        msg = "Angular audit requires unspotted/nonpulsating spin in[.5,1.5]"
        raise UnsupportedNestedMeshError(msg)
    d, weights = angular_grid(order)
    phi, theta = np.arctan2(d[:, 1], d[:, 0]), np.arccos(d[:, 2])
    q = binary.mass_ratio
    c = getattr(model, f"pre_calculate_for_potential_value_{component}")(
        star.synchronicity, q, 1., phi, theta, return_as_tuple=True,
    )
    potential = getattr(model, f"potential_value_{component}")
    try:
        r = np.array([brentq(lambda value, i=i: potential(value, q, *(v[i] for v in c))-star.surface_potential,
                            1e-6, .25, xtol=5e-16) for i in range(len(d))])
    except ValueError as exc:
        msg = "Angular audit roots left the validated inner[1e-6,.25] branch"
        raise UnsupportedNestedMeshError(msg) from exc
    points = r[:, None]*d
    normal_direction = d.copy()
    if component == "secondary":
        points[:, 0] = 1-points[:, 0]
        normal_direction[:, 0] *= -1
    gradient = calculate_potential_gradient(1., component, points, star.synchronicity, q)
    return {"order": order, "rays": len(d), **radial_measures(normal_direction, r, gradient, weights)}


def mesh_measures(star: Any, component: str) -> dict[str, Any]:
    """Raw and originally corrected faceted area/volume; cached fields untouched."""
    if component not in ("primary", "secondary") or any(
        getattr(star, key) is not None and np.asarray(getattr(star, key)).size for key in ("areas", "temperatures")
    ):
        msg = "Measure a named fresh raw component before correction/thermal state"
        raise UnsupportedNestedMeshError(msg)
    p = np.asarray(star.points)-np.array([0 if component == "primary" else 1, 0, 0])
    triangles = p[star.faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    # ELISa supplies quarter-mirrored winding; orient each contribution outward.
    sign = np.sign(np.einsum("ij,ij->i", cross, triangles.mean(axis=1)))
    volume = np.sum(sign*np.einsum("ij,ij->i", triangles[:, 0],
                                 np.cross(triangles[:, 1], triangles[:, 2])))/6
    area = np.sum(np.linalg.norm(cross, axis=1))/2
    factor = utils.discretization_correction_factor(star.discretization_factor,
                                                    _load_correction_factors()["detached"])
    return {"faces": len(star.faces), "effective_degrees": float(np.degrees(star.discretization_factor)),
            "scalar_correction": float(factor), "raw_area": float(area), "raw_volume": float(volume),
            "corrected_area": float(area*factor**2), "corrected_volume": float(volume*factor**3),
            "contract": "Geometric finite diagnostics only; no curve or continuum error certificate"}
