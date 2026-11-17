"""CPU bridge for true Roche patch quadrature and implicit eclipse predicates.

No global mesh correction is applied to quadrature geometry. Thermal/radiance
fields remain a separately declared finite original-ELISa model. Boundary
refinement is a finite numerical rule, never a certified continuum bound.
"""

# ruff: noqa: PLR2004 - explicit quadrature, triangle and validated geometry bounds
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from elisa.binary_system import model
from elisa.binary_system.surface.gravity import calculate_potential_gradient

_R_MIN, _R_MAX = 1e-6, .25
_MAX_NODES = 2097152


class UnsupportedCurvedGeometryError(RuntimeError):
    """Abort the requested numerical protocol; never convert into a prior cut."""


@dataclass(frozen=True)
class CurvedRule:
    """Duffy order and finite adaptive boundary depth, fixed before inference."""

    order: int = 3
    depth: int = 3
    max_nodes: int = _MAX_NODES

    def __post_init__(self) -> None:
        if (isinstance(self.order, bool) or not isinstance(self.order, int) or not 2 <= self.order <= 4
                or isinstance(self.depth, bool) or not isinstance(self.depth, int) or not 0 <= self.depth <= 7
                or not isinstance(self.max_nodes, int) or isinstance(self.max_nodes, bool)
                or not 1 <= self.max_nodes <= _MAX_NODES):
            msg = "Curved rule requires order2-4, depth0-7, positive node cap<=2097152"
            raise ValueError(msg)


class RocheGeometry:
    """Original CPU potential/gradient, inner radial branch and projected limb."""

    def __init__(self, binary: Any, component: str) -> None:
        if (component not in ("primary", "secondary") or binary.morphology != "detached" or binary.eccentricity != 0
                or binary.has_spots() or binary.has_pulsations() or not .5 <= binary.mass_ratio <= 2):
            msg = "Curved geometry requires circular detached unspotted/nonpulsating q in[.5,2]"
            raise UnsupportedCurvedGeometryError(msg)
        self.binary, self.component = binary, component
        self.star = getattr(binary, component)
        if not .5 <= self.star.synchronicity <= 1.5:
            msg = "Curved geometry requires spin in[.5,1.5]"
            raise UnsupportedCurvedGeometryError(msg)
        self.centre = np.array([0. if component == "primary" else 1., 0., 0.])
        self.mass = 1. if component == "primary" else binary.mass_ratio
        self.inner = self.mass/self.star.surface_potential
        if not _R_MIN < self.inner < _R_MAX:
            msg = "Curved geometry left the validated radial interval"
            raise UnsupportedCurvedGeometryError(msg)

    def potential(self, points: np.ndarray) -> np.ndarray:
        p = points-self.centre
        if self.component == "secondary":
            p[:, 0] *= -1
        r = np.linalg.norm(p, axis=1)
        phi, theta = np.arctan2(p[:, 1], p[:, 0]), np.arccos(np.clip(p[:, 2]/r, -1, 1))
        coeff = getattr(model, f"pre_calculate_for_potential_value_{self.component}")(
            self.star.synchronicity, self.binary.mass_ratio, 1., phi, theta, return_as_tuple=True,
        )
        return getattr(model, f"potential_value_{self.component}")(r, self.binary.mass_ratio, *coeff)

    def sample(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return true points, outward normals and dA/dOmega for unit rays."""
        d = np.asarray(vectors)/np.linalg.norm(vectors, axis=1)[:, None]
        local = d.copy()
        if self.component == "secondary":
            local[:, 0] *= -1
        coeff = getattr(model, f"pre_calculate_for_potential_value_{self.component}")(
            self.star.synchronicity, self.binary.mass_ratio, 1., np.arctan2(local[:, 1], local[:, 0]),
            np.arccos(np.clip(local[:, 2], -1, 1)), return_as_tuple=True,
        )
        potential = getattr(model, f"potential_value_{self.component}")
        lo, hi = np.full(len(d), _R_MIN), np.full(len(d), _R_MAX)
        target, q = self.star.surface_potential, self.binary.mass_ratio
        if np.any(potential(lo, q, *coeff) <= target) or np.any(potential(hi, q, *coeff) >= target):
            msg = "Curved rays are outside the inner[1e-6,.25] bracket"
            raise UnsupportedCurvedGeometryError(msg)
        for _ in range(56):
            mid = (lo+hi)/2
            inner = potential(mid, q, *coeff) > target
            lo, hi = np.where(inner, mid, lo), np.where(inner, hi, mid)
        r = (lo+hi)/2
        points = self.centre+r[:, None]*d
        g = calculate_potential_gradient(1., self.component, points, self.star.synchronicity, q)
        norm, radial = np.linalg.norm(g, axis=1), np.einsum("ij,ij->i", g, d)
        if not np.isfinite(g).all() or np.any(radial <= 0):
            msg = "Curved surface is not on the supported star-shaped branch"
            raise UnsupportedCurvedGeometryError(msg)
        return points, g/norm[:, None], r*r*norm/radial

    def shadow(self, points: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact implicit ray predicate with a finite CPU stationary-point solve.

        The inscribed radius m/Omega follows from positive companion/rotation
        terms for this q/radius domain. Outside that sphere, solve gradPhi dot
        LOS=0 and check the original potential. Returned distance is ONLY an
        adaptive indicator, not a signed-distance or truncation-error bound.
        """
        delta = points-self.centre
        closest = delta-np.outer(delta@line, line)
        rho = np.linalg.norm(closest, axis=1)
        inside = rho < self.inner
        distance = rho-self.inner
        selected = (rho >= self.inner) & (rho < _R_MAX)
        if not selected.any():
            return inside, distance
        base = closest[selected]+self.centre
        t = np.zeros(len(base))
        q, spin = self.binary.mass_ratio, self.star.synchronicity
        for _ in range(12):
            p = base+np.outer(t, line)
            g = calculate_potential_gradient(1., self.component, p, spin, q)
            derivative = np.zeros(len(p))
            for centre, mass in [(np.zeros(3), 1.), (np.array([1., 0., 0.]), q)]:
                v = p-centre
                r2 = np.sum(v*v, axis=1)
                derivative += mass*(r2-3*(v@line)**2)/r2**2.5
            derivative -= spin**2*(1+q)*np.sum(line[:2]**2)
            if np.any(derivative <= 0):
                msg = "Implicit projected-limb solve lost the supported local maximum"
                raise UnsupportedCurvedGeometryError(msg)
            step = (g@line)/derivative
            t -= np.clip(step, -rho[selected]/4, rho[selected]/4)
        p = base+np.outer(t, line)
        g = calculate_potential_gradient(1., self.component, p, spin, q)
        if (np.any(np.abs(t) > rho[selected]/2)
                or np.any(np.abs(g@line) > 64*np.finfo(float).eps*np.linalg.norm(g, axis=1))):
            msg = "Implicit eclipse predicate did not meet its FP64 stationary residual gate"
            raise UnsupportedCurvedGeometryError(msg)
        value = self.potential(p)
        inside[selected] = value > self.star.surface_potential
        r = np.linalg.norm(p-self.centre, axis=1)
        extras = value-self.mass/r
        effective_radius = self.mass/(self.star.surface_potential-extras)
        distance[selected] = rho[selected]-effective_radius
        return inside, distance


def split_triangles(triangles: np.ndarray) -> np.ndarray:
    """Split the barycentric plane, retaining exact curved-patch ancestry."""
    a, b, c = triangles.transpose(1, 0, 2)
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return np.stack([np.stack(v, axis=1) for v in [(a, ab, ca), (ab, b, bc),
                                                  (ca, bc, c), (ab, bc, ca)]], axis=1).reshape((-1, 3, 3))


def duffy_nodes(triangles: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return mapped triangle nodes and exact solid-angle Jacobian weights."""
    x, w = leggauss(order)
    s, t = np.meshgrid((x+1)/2, (x+1)/2, indexing="ij")
    weight = np.outer(w/2, w/2)*(1-s)
    bary = np.stack([1-s-(1-s)*t, s, (1-s)*t], axis=-1).reshape((-1, 3))
    vectors = np.einsum("qj,njk->nqk", bary, triangles)
    det = np.abs(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])))
    weights = det[:, None]*weight.reshape(1, -1)/np.linalg.norm(vectors, axis=2)**3
    return vectors.reshape((-1, 3)), weights.reshape(-1)


def curved_packet(triangles: np.ndarray, geometry: Any, line: np.ndarray, rule: CurvedRule,
                  foreground: Any = None) -> dict[str, Any]:
    """Finite adaptive rule with exact node geometry and implicit node masking.

    The empirical 7-point boundary indicator adds a two-diameter safety band;
    it is not an interval enclosure. Independent depth/order controls remain
    mandatory before inference. Node-cap failure aborts, never coarsens.
    """
    current, parents = np.array(triangles, copy=True), np.arange(len(triangles))
    accepted, ancestry, counts = [], [], []
    probes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5],
                       [1/3, 1/3, 1/3]])
    for depth in range(rule.depth+1):
        v = np.einsum("qj,njk->nqk", probes, current)
        p, normal, _ = geometry.sample(v.reshape((-1, 3)))
        p, normal = p.reshape((-1, 7, 3)), normal.reshape((-1, 7, 3))
        mu = normal@line
        spread = np.linalg.norm(normal-normal[:, :1], axis=2).max(axis=1)
        possible = mu.max(axis=1) > -2*spread
        boundary = possible & (np.min(np.abs(mu), axis=1) <= 2*spread)
        if foreground is not None:
            _, distance = foreground.shadow(p.reshape((-1, 3)), line)
            diameter = np.linalg.norm(p-p[:, :1], axis=2).max(axis=1)
            boundary |= possible & (np.abs(distance).reshape((-1, 7)).min(axis=1) <= 2*diameter)
        refine = boundary & (depth < rule.depth)
        take = possible & ~refine
        accepted.append(current[take])
        ancestry.append(parents[take])
        counts.append({"depth": depth, "leaves": len(current), "refined": int(refine.sum()),
                       "accepted": int(take.sum()), "boundary_leaves": int(boundary.sum())})
        leaves = sum(len(a) for a in accepted)+int(refine.sum())*4
        if leaves*rule.order**2 > rule.max_nodes:
            msg = "Curved adaptive quadrature exceeds its explicit node cap; abort protocol"
            raise UnsupportedCurvedGeometryError(msg)
        if not refine.any():
            break
        current, parents = split_triangles(current[refine]), np.repeat(parents[refine], 4)
    leaves, parent = np.concatenate(accepted), np.concatenate(ancestry)
    if not len(leaves):
        return {"mu": np.zeros(0), "area": np.zeros(0), "parents": np.zeros(0, dtype=int), "counts": counts}
    vectors, weights = duffy_nodes(leaves, rule.order)
    points, normal, jacobian = geometry.sample(vectors)
    mu = normal@line
    valid = mu > 0
    if foreground is not None:
        blocked, _ = foreground.shadow(points, line)
        valid &= ~blocked
    return {"mu": mu[valid], "area": (weights*jacobian)[valid],
            "parents": np.repeat(parent, rule.order**2)[valid], "counts": counts,
            "nodes": len(vectors), "selected_nodes": int(valid.sum())}
