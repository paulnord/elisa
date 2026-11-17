"""Nested, conforming Roche meshes for finite surface-error studies.

The seed is ELISa's actual fresh candidate mesh. All new edge midpoints are
projected onto the original inner Roche equipotential using CPU Brent roots.
Quarter symmetry and parent lineage are explicit; no field is interpolated
from a previous physical candidate. This is not continuum error certification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

from elisa.binary_system import model

_MAX_LEVELS, _MAX_FACES = 3, 40000
_Q_MIN, _Q_MAX, _SPIN_MIN, _SPIN_MAX = .5, 2., .5, 1.5
_SIGNS = np.array([[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]])


class UnsupportedNestedMeshError(RuntimeError):
    """The requested numerical mesh is unsupported; never substitute a coarse one."""


@dataclass(frozen=True)
class NestedMesh:
    """Uniform nested subdivisions with ancestry for local error indicators.

    The original scalar mesh correction uses the declared effective degree
    seed_degree/2**levels. Its calibration is not a new error bound. Level0 is
    the original mesh exactly; levels1-3 keep each seed face's four descendants
    contiguous and preserve original quarter-face symmetry.
    """

    levels: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.levels, bool) or not isinstance(self.levels, int) or not 0 <= self.levels <= _MAX_LEVELS:
            msg = "Nested refinement levels must be an integer from0 to3"
            raise ValueError(msg)

    def apply(self, system: Any) -> dict[str, Any]:
        """Refine fresh raw mesh/faces before gravity, correction or temperatures."""
        if (system.morphology != "detached" or system.eccentricity != 0
                or system.has_spots() or system.has_pulsations() or not _Q_MIN <= system.mass_ratio <= _Q_MAX):
            msg = "Nested mesh requires circular detached unspotted/nonpulsating q in[.5,2]"
            raise UnsupportedNestedMeshError(msg)
        result = {}
        for name in ("primary", "secondary"):
            star = getattr(system, name)
            if any(getattr(star, key) is not None and np.asarray(getattr(star, key)).size
                   for key in ("areas", "temperatures", "potential_gradient_magnitudes")):
                msg = "Nested refinement requires fresh raw mesh before correction/gravity/thermal state"
                raise UnsupportedNestedMeshError(msg)
            if not _SPIN_MIN <= star.synchronicity <= _SPIN_MAX or not star.symmetry_test():
                msg = "Nested mesh requires original symmetric spin in[.5,1.5]"
                raise UnsupportedNestedMeshError(msg)
            count = int(star.base_symmetry_faces_number)
            if len(star.faces) != 4*count or 4*count*4**self.levels > _MAX_FACES:
                msg = "Nested mesh exceeds40000 symmetric faces per component"
                raise UnsupportedNestedMeshError(msg)
            parents = np.arange(count, dtype=np.int64)
            points = np.array(star.symmetry_points(), copy=True)
            faces = np.array(star.symmetry_faces(star.faces), dtype=np.int64, copy=True)
            seed_degree = float(star.discretization_factor)
            if self.levels:
                # Original stitching identifies seam vertices topologically,
                # even when trig leaves ~1e-17 in a mathematically zero axis.
                # Reuse those identities, never tolerance-merge nearby points.
                inverse = star.inverse_point_symmetry_matrix
                points[inverse[0] == inverse[1], 1] = 0.
                points[inverse[0] == inverse[3], 2] = 0.
            for _ in range(self.levels):
                points, faces = self._subdivide(points, faces, star, name, system.mass_ratio)
                parents = np.repeat(parents, 4)
            if self.levels:
                all_points, inverse = self._mirror(points)
                star.points = all_points
                star.base_symmetry_points_number = len(points)
                star.inverse_point_symmetry_matrix = inverse
                star.base_symmetry_faces = faces
                star.base_symmetry_faces_number = len(faces)
                star.faces = np.concatenate([indices[faces] for indices in inverse])
                star.face_symmetry_vector = np.tile(np.arange(len(faces)), 4)
                star.discretization_factor = seed_degree/2**self.levels
            ancestry = np.concatenate([parents+q*count for q in range(4)])
            ancestry.flags.writeable = False
            result[name] = {"root_faces": 4*count, "parents": ancestry,
                            "seed_degrees": float(np.degrees(seed_degree)),
                            "effective_degrees": float(np.degrees(star.discretization_factor))}
        return result

    @staticmethod
    def _mirror(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values, lookup, inverse = [], {}, []
        for sign in _SIGNS:
            indices = []
            for point in points*sign:
                key = tuple(point)
                if key not in lookup:
                    lookup[key] = len(values)
                    values.append(point)
                indices.append(lookup[key])
            inverse.append(indices)
        if not np.array_equal(inverse[0], np.arange(len(points))):
            msg = "Ambiguous duplicate vertices in nested base quarter"
            raise UnsupportedNestedMeshError(msg)
        return np.asarray(values), np.asarray(inverse, dtype=np.int64)

    @staticmethod
    def _subdivide(points: np.ndarray, faces: np.ndarray, star: Any, component: str, q: float) -> tuple:
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        midpoint = points[unique].mean(axis=1)
        local = midpoint.copy()
        if component == "secondary":
            local[:, 0] = 1-local[:, 0]
        direction = local/np.linalg.norm(local, axis=1)[:, None]
        phi = np.arctan2(direction[:, 1], direction[:, 0])
        theta = np.arccos(np.clip(direction[:, 2], -1, 1))
        coefficients = getattr(model, f"pre_calculate_for_potential_value_{component}")(
            star.synchronicity, q, 1., phi, theta, return_as_tuple=True,
        )
        potential = getattr(model, f"potential_value_{component}")
        try:
            radii = np.array([brentq(lambda r, j=j: potential(r, q, *(c[j] for c in coefficients))
                                    -star.surface_potential, 1e-6, .25, xtol=5e-16)
                              for j in range(len(unique))])
        except ValueError as exc:
            msg = "New Roche rays are outside the validated inner[1e-6,.25] branch"
            raise UnsupportedNestedMeshError(msg) from exc
        new_points = direction*radii[:, None]
        if component == "secondary":
            new_points[:, 0] = 1-new_points[:, 0]
        m = inverse.reshape((-1, 3))+len(points)
        a, b, c = faces.T
        ab, bc, ca = m.T
        children = np.stack([np.stack(v, axis=1) for v in [(a, ab, ca), (ab, b, bc),
                                                          (ca, bc, c), (ab, bc, ca)]], axis=1)
        return np.concatenate([points, new_points]), children.reshape((-1, 3))
