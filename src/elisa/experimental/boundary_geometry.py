"""Finite boundary-fitted Roche patch integration, with conditioned shadows.

Edge roots lie on the implicit limb/shadow. Their straight chords in each
barycentric leaf approximate the curved boundary; this is not an exact boundary
integral or an interval certificate. Depth/order/field controls remain required.
"""

# ruff: noqa: PLR2004 - explicit geometry, subdivision and resource bounds
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa.binary_system import model
from elisa.experimental.curved_geometry import (
    CurvedRule,
    RocheGeometry,
    UnsupportedCurvedGeometryError,
    duffy_nodes,
    split_triangles,
)


@dataclass(frozen=True)
class BoundaryRule(CurvedRule):
    """Fixed finite subdivision and bracketed edge-root protocol."""

    edge_bisections: int = 36
    topology_depth: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if (isinstance(self.edge_bisections, bool) or not isinstance(self.edge_bisections, int)
                or not 32 <= self.edge_bisections <= 48):
            msg = "Boundary edge bisections must be integer32-48"
            raise ValueError(msg)
        if (isinstance(self.topology_depth, bool) or not isinstance(self.topology_depth, int)
                or not 0 <= self.topology_depth <= 4):
            msg = "Boundary topology depth must be integer0-4"
            raise ValueError(msg)


class ConditionedRocheGeometry(RocheGeometry):
    """Same original potential in star-local coordinates throughout tangency.

    The outer gradient is minus the original potential derivative. Avoiding
    repeated addition/subtraction of the secondary's unit centre preserves
    small-star stationary residuals without loosening the original 64eps gate.
    """

    def potential_local(self, local: np.ndarray) -> np.ndarray:
        p = np.array(local, copy=True)
        if self.component == "secondary":
            p[:, 0] *= -1
        r = np.linalg.norm(p, axis=1)
        coeff = getattr(model, f"pre_calculate_for_potential_value_{self.component}")(
            self.star.synchronicity, self.binary.mass_ratio, 1., np.arctan2(p[:, 1], p[:, 0]),
            np.arccos(np.clip(p[:, 2]/r, -1, 1)), return_as_tuple=True,
        )
        return getattr(model, f"potential_value_{self.component}")(r, self.binary.mass_ratio, *coeff)

    def gradient_local(self, local: np.ndarray) -> np.ndarray:
        q, spin = self.binary.mass_ratio, self.star.synchronicity
        companion = local+np.array([-1. if self.component == "primary" else 1., 0., 0.])
        other_mass = q if self.component == "primary" else 1.
        g = (self.mass*local/np.linalg.norm(local, axis=1)[:, None]**3
             +other_mass*companion/np.linalg.norm(companion, axis=1)[:, None]**3)
        g[:, :2] -= spin**2*(1+q)*local[:, :2]
        g[:, 0] += q if self.component == "primary" else -1.
        return g

    def shadow(self, points: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = points-self.centre
        base = delta-np.outer(delta@line, line)
        rho = np.linalg.norm(base, axis=1)
        inside = rho < self.inner
        distance = rho-self.inner
        selected = (rho >= self.inner) & (rho < .25)
        if not selected.any():
            return inside, distance
        base, rhos = base[selected], rho[selected]
        t = np.zeros(len(base))
        q, spin = self.binary.mass_ratio, self.star.synchronicity
        other_mass = q if self.component == "primary" else 1.
        shift = np.array([-1. if self.component == "primary" else 1., 0., 0.])
        for _ in range(12):
            local = base+np.outer(t, line)
            g = self.gradient_local(local)
            derivative = np.zeros(len(local))
            for v, mass in [(local, self.mass), (local+shift, other_mass)]:
                r2 = np.sum(v*v, axis=1)
                derivative += mass*(r2-3*(v@line)**2)/r2**2.5
            derivative -= spin**2*(1+q)*np.sum(line[:2]**2)
            if np.any(derivative <= 0) or not np.isfinite(derivative).all():
                msg = "Conditioned shadow lost the supported stationary maximum"
                raise UnsupportedCurvedGeometryError(msg)
            t -= np.clip((g@line)/derivative, -rhos/4, rhos/4)
        local = base+np.outer(t, line)
        g = self.gradient_local(local)
        residual = np.abs(g@line)
        if (not np.isfinite(g).all() or np.any(np.abs(t) > rhos/2)
                or np.any(residual > 64*np.finfo(float).eps*np.linalg.norm(g, axis=1))):
            msg = "Conditioned shadow did not meet the original FP64 stationary residual gate"
            raise UnsupportedCurvedGeometryError(msg)
        value = self.potential_local(local)
        inside[selected] = value > self.star.surface_potential
        # Unlike rho-effective_radius, this margin has its zero at the actual
        # stationary-potential silhouette, even when tangency t is nonzero.
        distance[selected] = (self.star.surface_potential-value)*rhos*rhos/self.mass
        return inside, distance


def _clip(triangles: np.ndarray, parents: np.ndarray, field: Any,
          steps: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Sutherland-style triangle clipping, with independently bracketed edges.

    Zero sets are replaced by finite leaf chords. Endpoint sign topology is a
    finite approximation, checked by the enclosing subdivision probes below.
    """
    if not len(triangles):
        return triangles, parents, 0
    values = field(triangles.reshape((-1, 3))).reshape((-1, 3))
    positive = values >= 0
    following = np.roll(positive, -1, axis=1)
    crossing = positive != following
    roots = np.zeros_like(triangles)
    a, b = triangles[crossing], np.roll(triangles, -1, axis=1)[crossing]
    low, high = np.zeros(len(a)), np.ones(len(a))
    low_positive = positive[crossing]
    for _ in range(steps if len(a) else 0):
        mid = (low+high)/2
        same = (field(a+(b-a)*mid[:, None]) >= 0) == low_positive
        low, high = np.where(same, mid, low), np.where(same, high, mid)
    roots[crossing] = a+(b-a)*((low+high)/2)[:, None]
    slots = np.zeros((len(triangles), 6, 3))
    size = np.zeros(len(triangles), dtype=int)
    for edge in range(3):
        take = positive[:, edge]
        index = np.flatnonzero(take)
        slots[index, size[index]] = triangles[index, edge]
        size += take
        take = crossing[:, edge]
        index = np.flatnonzero(take)
        slots[index, size[index]] = roots[index, edge]
        size += take
    pieces, ancestry = [], []
    for j in range(1, 3):
        take = size > j+1
        pieces.append(slots[take][:, [0, j, j+1]])
        ancestry.append(parents[take])
    return np.concatenate(pieces), np.concatenate(ancestry), len(a)


def boundary_packet(triangles: np.ndarray, geometry: Any, line: np.ndarray, rule: BoundaryRule,  # noqa: C901, PLR0915
                    foreground: Any = None) -> dict[str, Any]:
    """Boundary-fitted finite leaf chords, fresh true geometry and parent fields.

    Seven probes and a two-spread/diameter band guide subdivision. Sampled
    interior contours receive bounded extra subdivisions under the fixed rule;
    exhaustion aborts. This is not proof that all sub-probe loops are absent.
    """
    current, parents = np.array(triangles, copy=True), np.arange(len(triangles))
    accepted, ancestry, counts = [], [], []
    probes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5],
                       [1/3, 1/3, 1/3]])
    for depth in range(rule.depth+rule.topology_depth+1):
        v = np.einsum("qj,njk->nqk", probes, current)
        p, normal, _ = geometry.sample(v.reshape((-1, 3)))
        p, normal = p.reshape((-1, 7, 3)), normal.reshape((-1, 7, 3))
        mu = normal@line
        spread = np.linalg.norm(normal-normal[:, :1], axis=2).max(axis=1)
        possible = mu.max(axis=1) > -2*spread
        boundary = possible & (np.min(np.abs(mu), axis=1) <= 2*spread)
        shadow = None
        if foreground is not None:
            blocked, distance = foreground.shadow(p.reshape((-1, 3)), line)
            shadow = distance.reshape((-1, 7))
            diameter = np.linalg.norm(p-p[:, :1], axis=2).max(axis=1)
            boundary |= possible & (np.abs(shadow).min(axis=1) <= 2*diameter)
            if not np.array_equal(blocked.reshape((-1, 7)), shadow < 0):
                msg = "Shadow margin and opacity predicate disagree; abort protocol"
                raise UnsupportedCurvedGeometryError(msg)
        refine = boundary & (depth < rule.depth)
        topology_refine = np.zeros(len(current), dtype=bool)
        if depth >= rule.depth:
            for field in [mu, *([] if shadow is None else [shadow])]:
                endpoints = field[:, :3] >= 0
                same = np.all(endpoints == endpoints[:, :1], axis=1)
                hidden = np.any((field >= 0) != endpoints[:, :1], axis=1)
                topology_refine |= possible & same & hidden
                if depth == rule.depth+rule.topology_depth and topology_refine.any():
                    msg = "Boundary leaf has unresolved sampled contour topology; increase fixed depth"
                    raise UnsupportedCurvedGeometryError(msg)
        refine |= topology_refine
        take = possible & ~refine
        accepted.append(current[take])
        ancestry.append(parents[take])
        counts.append({"depth": depth, "leaves": len(current), "refined": int(refine.sum()),
                       "accepted": int(take.sum()), "boundary_leaves": int(boundary.sum()),
                       "topology_refined": int(topology_refine.sum())})
        leaves = sum(len(a) for a in accepted)+int(refine.sum())*4
        if leaves*rule.order**2 > rule.max_nodes:
            msg = "Boundary quadrature exceeds explicit node cap; abort protocol"
            raise UnsupportedCurvedGeometryError(msg)
        if not refine.any():
            break
        current, parents = split_triangles(current[refine]), np.repeat(parents[refine], 4)
    leaves, parent = np.concatenate(accepted), np.concatenate(ancestry)

    def limb(v: np.ndarray) -> np.ndarray:
        return geometry.sample(v)[1]@line

    leaves, parent, limb_roots = _clip(leaves, parent, limb, rule.edge_bisections)
    shadow_roots = 0
    if foreground is not None:
        def clear(v: np.ndarray) -> np.ndarray:
            return foreground.shadow(geometry.sample(v)[0], line)[1]

        leaves, parent, shadow_roots = _clip(leaves, parent, clear, rule.edge_bisections)
    nodes = len(leaves)*rule.order**2
    if nodes > rule.max_nodes:
        msg = "Clipped boundary quadrature exceeds explicit node cap; abort protocol"
        raise UnsupportedCurvedGeometryError(msg)
    if not len(leaves):
        return {"mu": np.zeros(0), "area": np.zeros(0), "parents": np.zeros(0, dtype=int),
                "counts": counts, "nodes": 0, "selected_nodes": 0,
                "limb_edge_roots": limb_roots, "shadow_edge_roots": shadow_roots}
    vectors, weights = duffy_nodes(leaves, rule.order)
    _, normal, jacobian = geometry.sample(vectors)
    mu = normal@line
    valid = mu > 0
    # No shadow remasking: the numerical integration domain is the clipped leaf
    # polygon. Reintroducing node masks would defeat the fitted-boundary rule.
    # True negative limb cosines contribute zero, as in original CPU semantics.
    return {"mu": mu[valid], "area": (weights*jacobian)[valid],
            "parents": np.repeat(parent, rule.order**2)[valid], "counts": counts,
            "nodes": nodes, "selected_nodes": int(valid.sum()),
            "limb_edge_roots": limb_roots, "shadow_edge_roots": shadow_roots}
