"""Fixed selective Roche refinement with conforming red/green triangles.

A pilot chooses root-face patches once. Every later physical evaluation builds
all fields afresh and verifies the seed topology. No candidate-dependent budget
failure changes the plan, prior or numerical model. CPU defaults are untouched.
"""

# ruff: noqa: SLF001, PLR2004 - explicit quarter symmetry and triangle edge cases
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa import const
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.experimental.gpu_nested_mesh import NestedMesh, UnsupportedNestedMeshError

_NAMES = ("primary", "secondary")
_MAX_FACES = 40000


def topology_id(star: Any) -> str:
    """Topology identity, permitting physical movement but no index reassignment."""
    h = hashlib.sha256(np.asarray(star.faces, dtype=np.int64).tobytes())
    h.update(str((len(star.points), star.base_symmetry_points_number)).encode())
    return h.hexdigest()


def seed_system(binary: Any) -> Any:
    """Build the original raw seed without temperatures, correction or reflection."""
    system = OrbitalPositionContainer.from_binary_system(binary, const.Position(0, 1., 0., 0., 0.))
    system.build_mesh(components_distance=1.)
    system.build_faces(components_distance=1.)
    return system


@dataclass(frozen=True)
class SelectiveMesh(NestedMesh):
    """Refine selected seed-quarter patches from levels-1 to levels (1-3).

    roots and topology are immutable plan metadata. The default density-matched
    effective angle is base_angle*sqrt(base_faces/new_faces), an explicit finite
    convention for the ORIGINAL scalar correction. coarse/fine are diagnostic
    correction ablations, not alternative physical priors or accuracy fixes.
    """

    roots: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    topology: tuple[str, str] = ("", "")
    correction: str = "density"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.levels or self.correction not in ("density", "coarse", "fine"):
            msg = "Selective mesh requires levels1-3 and density/coarse/fine correction"
            raise ValueError(msg)
        if (not isinstance(self.roots, tuple) or len(self.roots) != 2
                or not isinstance(self.topology, tuple) or len(self.topology) != 2
                or any(not isinstance(t, str) or len(t) != 64 for t in self.topology)
                or any(not isinstance(r, tuple)
                       or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in r)
                       or tuple(sorted(set(r))) != r for r in self.roots)):
            msg = "Supply frozen sorted unique quarter-root IDs and seed topology hashes"
            raise ValueError(msg)

    def apply(self, system: Any) -> dict[str, Any]:
        for name, token in zip(_NAMES, self.topology, strict=True):
            if topology_id(getattr(system, name)) != token:
                msg = "Fixed selective plan seed topology changed; abort numerical protocol"
                raise UnsupportedNestedMeshError(msg)
        info = NestedMesh(self.levels-1).apply(system)
        for name, chosen in zip(_NAMES, self.roots, strict=True):
            star = getattr(system, name)
            nseed = info[name]["root_faces"]//4
            if any(i >= nseed for i in chosen):
                msg = "Selective quarter-root ID is outside the seed mesh"
                raise UnsupportedNestedMeshError(msg)
            points = np.array(star.symmetry_points(), copy=True)
            faces = np.array(star.symmetry_faces(star.faces), dtype=np.int64, copy=True)
            parents = info[name]["parents"][:len(faces)]
            old_count, base_angle = len(faces), star.discretization_factor
            if chosen:
                inverse = star.inverse_point_symmetry_matrix
                points[inverse[0] == inverse[1], 1] = 0.
                points[inverse[0] == inverse[3], 2] = 0.
                points, faces, parents = self._split(points, faces, parents, np.isin(parents, chosen),
                                                    star, name, system.mass_ratio)
            if 4*len(faces) > _MAX_FACES:
                msg = "Selective mesh exceeds40000 symmetric faces per component"
                raise UnsupportedNestedMeshError(msg)
            if chosen:
                points, inverse = self._mirror(points)
                star.points, star.inverse_point_symmetry_matrix = points, inverse
                star.base_symmetry_points_number = len(inverse[0])
                star.base_symmetry_faces, star.base_symmetry_faces_number = faces, len(faces)
                star.faces = np.concatenate([indices[faces] for indices in inverse])
                star.face_symmetry_vector = np.tile(np.arange(len(faces)), 4)
            factor = {"coarse": 1., "fine": .5, "density": np.sqrt(old_count/len(faces))}[self.correction]
            star.discretization_factor = base_angle*factor
            ancestry = np.concatenate([parents+q*nseed for q in range(4)])
            ancestry.flags.writeable = False
            info[name].update(parents=ancestry, effective_degrees=float(np.degrees(star.discretization_factor)),
                              selected_quarter_roots=len(chosen), base_faces=4*old_count, faces=len(star.faces),
                              correction=self.correction)
        return info

    @staticmethod
    def _split(points: np.ndarray, faces: np.ndarray, parents: np.ndarray, selected: np.ndarray,
               star: Any, component: str, q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
        marked = np.unique(edges[selected].reshape((-1, 2)), axis=0)
        if not len(marked):
            return points, faces, parents.copy()
        new = NestedMesh._project_edges(points, marked, star, component, q)
        lookup = {tuple(edge): i+len(points) for i, edge in enumerate(marked)}
        all_points = np.concatenate([points, new])
        children, lineage = [], []
        for face, edge, parent in zip(faces, edges, parents, strict=True):
            mid = [lookup.get(tuple(e), -1) for e in edge]
            n = sum(i >= 0 for i in mid)
            a, b, c = face
            ab, bc, ca = mid
            if n == 0:
                patch = [(a, b, c)]
            elif n == 3:
                patch = [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            elif n == 1:
                i = next(j for j in range(3) if mid[j] >= 0)
                a, b, c = np.roll(face, -i)
                m = mid[i]
                patch = [(a, m, c), (m, b, c)]
            else:
                i = next(j for j in range(3) if mid[j] >= 0 and mid[(j+1)%3] >= 0)
                a, b, c = np.roll(face, -i)
                ab, bc = mid[i], mid[(i+1)%3]
                patch = [(ab, b, bc)]
                if np.linalg.norm(all_points[ab]-all_points[c]) <= np.linalg.norm(all_points[a]-all_points[bc]):
                    patch += [(a, ab, c), (ab, bc, c)]
                else:
                    patch += [(a, ab, bc), (a, bc, c)]
            children.extend(patch)
            lineage.extend([parent]*len(patch))
        return all_points, np.asarray(children, dtype=np.int64), np.asarray(lineage, dtype=np.int64)


def choose_plan(binary: Any, coarse: Any, fine: Any, sigma: Any, *, fraction: float = .5,
                correction: str = "density") -> SelectiveMesh:
    """Freeze a pilot's nominal-noise-weighted root indicators, folded by quarter.

    Uses original maximum normalization and all supplied phases/bands, without
    nuisance-noise dilution. Scores describe coupled finite-model changes, not
    independent local truncation errors. Do not refit this plan per proposal.
    """
    if (coarse.signature != fine.signature or fine.levels != coarse.levels+1
            or not np.array_equal(coarse.phases, fine.phases) or not np.isfinite(fraction) or not 0 <= fraction <= 1):
        msg = "Plan needs a matching consecutive pilot pair and fraction in[0,1]"
        raise ValueError(msg)
    system = seed_system(binary)
    scores = [np.zeros(getattr(system, n).base_symmetry_faces_number) for n in _NAMES]
    for band in coarse.fluxes:
        error = np.asarray(sigma[band] if isinstance(sigma, dict) else sigma, dtype=float)
        if not np.isfinite(error).all() or np.any(error <= 0):
            msg = "Plan requires positive finite nominal observational errors"
            raise ValueError(msg)
        weights = []
        for snapshot in (coarse, fine):
            raw = snapshot.totals[band]
            offset = raw.mean()*snapshot.additional_light/(1-snapshot.additional_light)
            weights.append([v/(raw+offset).max() for v in snapshot.groups[band]])
        for i, score in enumerate(scores):
            change = (weights[1][i]-weights[0][i])/np.broadcast_to(error, (len(coarse.phases),))[:, None]
            if change.shape[1] != 4*len(score):
                msg = "Pilot root ancestry does not match requested seed topology"
                raise ValueError(msg)
            scores[i] += np.sum(change.reshape((len(change), 4, len(score)))**2, axis=(0, 1))
    chosen = tuple(tuple(sorted(int(j) for j in np.lexsort((np.arange(len(score)), -score))[
        :int(np.ceil(fraction*len(score)))
    ])) for score in scores)
    return SelectiveMesh(fine.levels, roots=chosen, topology=tuple(topology_id(getattr(system, n)) for n in _NAMES),
                         correction=correction)
