"""Integrate complete interior Duffy sibling panels on their exact barycentric parent."""

from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from elisa.experimental.curved_geometry import split_triangles
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, SymmetricInteriorRule
from elisa.experimental.shadow_refinement import deeper_shadow_packet


class OpticalFaces:
    """Certify a proposed merge by regenerating its triangle inside one optical face."""

    def __init__(self, triangles):
        self.triangles = triangles
        self.tree = cKDTree(triangles.mean(axis=1))

    def descendants(self, triangles, max_depth):
        centres = triangles.mean(axis=1)
        _, root = self.tree.query(centres)
        current = self.triangles[root]
        found = np.zeros(len(triangles), dtype=bool)
        active = np.arange(len(triangles))
        # Skip the optical root itself. Merge at most one level, strictly within a face.
        for _ in range(max_depth):
            children = split_triangles(current).reshape(-1, 4, 3, 3)
            delta = children.mean(axis=2) - centres[active, None, :]
            nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
            current = children[np.arange(len(active)), nearest]
            identical = np.all(current == triangles[active], axis=(1, 2))
            found[active[identical]] = True
            active = active[~identical]
            if not active.size:
                break
            current = current[~identical]
        return found


# The clipping code replaces its frozen rule when it refines a shadow. Keep the optical
# face index local to this packet across those replacements and concurrent observers.
_faces = ContextVar("duffy_optical_faces")


def compact_panels(triangles, source, faces, max_depth):
    """Replace four distinct, intact dyadic leaves by one panel with the same optical owner."""
    n = len(triangles)
    if n < 4:
        return triangles, source
    # The ordinary splitter emits three corners followed by the central child. Clipping
    # preserves the ordering of intact leaves. Nonconsecutive groups stay as evaluated.
    idx = np.flatnonzero(
        np.all(triangles[:-3, 1] == triangles[3:, 0], axis=1)
        & np.all(triangles[1:-2, 2] == triangles[3:, 1], axis=1)
        & np.all(triangles[2:-1, 0] == triangles[3:, 2], axis=1)
        & (source[1:-2] == source[:-3] + 1)
        & (source[2:-1] == source[:-3] + 2)
        & (source[3:] == source[:-3] + 3)
    )
    if not idx.size:
        return triangles, source
    parents = np.stack((triangles[idx, 0], triangles[idx + 1, 1], triangles[idx + 2, 2]), axis=1)
    quartet = idx[:, None] + np.arange(4)
    exact = np.all(split_triangles(parents).reshape(-1, 4, 3, 3) == triangles[quartet], axis=(1, 2, 3))
    idx, parents = idx[exact], parents[exact]
    if not idx.size:
        return triangles, source
    eligible = faces.descendants(parents, max_depth)
    idx, parents = idx[eligible], parents[eligible]
    if not idx.size:
        return triangles, source
    if np.any(np.diff(idx) < 4):
        raise ArithmeticError("Overlapping interior Duffy siblings")
    # All four accepted leaves lie within the same dyadic parent. A strict ancestor of any
    # one would cover the other three, impossible for four distinct, nonoverlapping leaves.
    keep = np.ones(n, dtype=bool)
    keep[idx[:, None] + np.arange(1, 4)] = False
    panels = triangles[keep]
    panels[np.cumsum(keep)[idx] - 1] = parents
    return panels, source[keep]


@dataclass(frozen=True)
class CompactDuffyRule(CurvedEdgeRule):
    duffy_sibling_levels: int = 1

    def planar_packet(self, triangles, source, budget):
        if self.duffy_sibling_levels != 1:
            raise ValueError("Only a single Duffy sibling merge is supported")
        faces = _faces.get(None)
        if faces is not None:
            triangles, source = compact_panels(
                triangles, source, faces, self.depth + max(self.junction_depth, self.topology_depth)
            )
        return super().planar_packet(triangles, source, budget)


@dataclass(frozen=True)
class CompanionSixPointRule(SymmetricInteriorRule):
    """Record the paired control while preserving the nominal six-point integration."""

    duffy_sibling_levels: int = 1


def compact_shadow_packet(triangles, geometry, line, rule, foreground=None):
    if not isinstance(rule, CompactDuffyRule) or foreground is None:
        return deeper_shadow_packet(triangles, geometry, line, rule, foreground)
    token = _faces.set(OpticalFaces(triangles))
    try:
        return deeper_shadow_packet(triangles, geometry, line, rule, foreground)
    finally:
        _faces.reset(token)
