"""Batched device projection, partial-face clipping, coverage and band flux.

CPU Qhull/Path and ELISa's visibility selectors remain a fresh, timed bridge.
This preserves the CPU boundary contract instead of substituting an analytic
eclipse. Device packets hold the actual new projected mesh and hulls; no previous
candidate's coverage, radiance, limb darkening or temperature is reused.
"""

# ruff: noqa: SLF001 - composition of experimental device kernels
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from elisa.binary_system.surface import coverage as cpu_coverage
from elisa.binary_system.utils import move_sys_onpos
from elisa.experimental.gpu_clipping import ConvexTriangleClipper
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator

_COMPONENTS = ("primary", "secondary")
_CHUNK = 8
_MAX_VERTICES, _MAX_FACES = 4096, 8192
_MAX_HULL, _MAX_PARTIAL = 128, 256
_MAX_PHASES = 128


class UnsupportedOccultationError(ValueError):
    """A bounded device evaluation is unsupported; use explicit full CPU fallback."""


@dataclass(frozen=True)
class PreparedOccultationBatch:
    """Immutable projected mesh and fresh CPU hull/selection packet on device."""

    _owner: Any
    _arrays: tuple[Any, ...]

    def evaluate(self) -> dict[str, Any]:
        if not self._owner._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        return self._owner._evaluate(*self._arrays)

    def block_until_ready(self) -> PreparedOccultationBatch:
        for array in self._arrays:
            array.block_until_ready()
        return self


class DeviceEclipseIntegrator:
    """Project fresh corrected CPU-compatible state and integrate changing eclipses.

    Geometry/brightness upload once per physical curve. Projection stays on the
    requested device; projected arrays return to CPU for Qhull/Path visibility.
    Only those new hulls/masks/partial-face indices upload for the remaining
    device work. Eight phases per packet bound clipping buffers. This is a
    hybrid algorithm with a CPU hull bridge, not resident GPU occultation.
    Projection supports at most 128 synthesis phases, 4096 vertices and 8192
    faces per star. Hull/partial capacities are 128/256. Larger requests require
    full CPU fallback; these are resource bounds, not physical-prior exclusions.
    """

    def __init__(self, law: str, *, device: str = "gpu") -> None:
        clipper = ConvexTriangleClipper(device=device)
        flux = SurfaceFluxIntegrator(law, device=device)
        self._jax, self.device = clipper._jax, clipper.device
        import jax.numpy as jnp  # noqa: PLC0415 - optional dependency

        barrier = self._jax.lax.optimization_barrier

        def project(points: Any, normals: Any, azimuth: Any, inclination: Any) -> dict[str, Any]:
            alpha, beta = azimuth + jnp.pi / 2, jnp.pi / 2 - inclination
            ca, sa, cb, sb = jnp.cos(alpha), jnp.sin(alpha), jnp.cos(beta), jnp.sin(beta)

            def rotate(vector: Any) -> Any:
                x, y, z = (vector[None, ..., i] for i in range(3))
                # Retain the two CPU row-vector rotation stages and FP64
                # rounding; corrected points and cached normals are distinct.
                nx = barrier(x * ca[:, None, None]) - barrier(y * sa[:, None, None])
                ny = barrier(x * sa[:, None, None]) + barrier(y * ca[:, None, None])
                nx, ny = barrier(nx), barrier(ny)
                return jnp.stack((barrier(nx * cb) - barrier(z * sb), ny,
                                  barrier(nx * sb) + barrier(z * cb)), axis=-1)

            rotated_normals = rotate(normals)
            return {"points": rotate(points), "normals": rotated_normals, "mu": -rotated_normals[..., 0]}

        self._project = self._jax.jit(project)

        def evaluate(
            points: Any, mu: Any, faces: Any, areas: Any, radiance: Any, coefficients: Any,
            polygons: Any, counts: Any, partial: Any, part_valid: Any, full: Any,
            thresholds: Any, scale: Any,
        ) -> dict[str, Any]:
            rows, width = mu.shape
            component = jnp.arange(rows) % 2
            selected_faces = jnp.take_along_axis(faces[component], partial[..., None], axis=1)
            triangles = self._jax.vmap(lambda p, f: p[f, 1:])(points, selected_faces)
            triangles = barrier(triangles)
            clipped = barrier(clipper._areas(polygons, counts, triangles))
            edge_a, edge_b = triangles[..., 1, :] - triangles[..., 0, :], triangles[..., 2, :] - triangles[..., 0, :]
            determinant = barrier(edge_a[..., 0] * edge_b[..., 1]) - barrier(edge_a[..., 1] * edge_b[..., 0])
            projected_area = barrier(jnp.abs(determinant) / 2)
            cosine = jnp.take_along_axis(mu, partial, axis=1)
            uncovered = barrier(projected_area - clipped["areas"]) / jnp.where(part_valid, cosine, 1.0)
            valid_partial = ~part_valid | ((clipped["status"] == 0) & jnp.isfinite(uncovered) & (cosine > 0))
            uncovered = jnp.where(part_valid & (uncovered >= thresholds[component, None]), uncovered, 0.0)
            coverage = jnp.where(full, areas[component], 0.0)
            coverage += jnp.zeros_like(coverage).at[jnp.arange(rows)[:, None], partial].add(uncovered)
            coverage = barrier(coverage * scale)
            bands = radiance.shape[1]
            cosines = jnp.broadcast_to(mu[:, None], (rows, bands, width)).reshape((-1, width))
            cov = jnp.broadcast_to(coverage[:, None], (rows, bands, width)).reshape((-1, width))
            face_flux, totals = flux._integrate(
                radiance[component].reshape((-1, width)), cosines, cov,
                coefficients[component].reshape((-1, width, coefficients.shape[-1])),
            )
            supported = jnp.all(valid_partial, axis=1) & jnp.all(jnp.isfinite(coverage) & (coverage >= 0), axis=1)
            supported &= jnp.all(jnp.isfinite(totals.reshape((rows, bands))), axis=1)
            return {"coverage": coverage, "intersection_areas": clipped["areas"],
                    "clipping_status": clipped["status"], "row_valid": supported,
                    "face_flux": face_flux.reshape((rows, bands, width)),
                    "component_flux": totals.reshape((rows // 2, 2, bands))}

        self._evaluate = self._jax.jit(evaluate)

    def _snapshot(  # noqa: C901 - explicit host versus device radiance validation
        self, system: Any, bands: tuple[str, ...], device_radiance: dict[str, Any] | None = None,
    ) -> tuple[tuple[Any, ...], list[Any]]:
        if (system.eccentricity != 0 or system.morphology != "detached"
                or system.has_spots() or system.has_pulsations()):
            msg = "Device occultation requires a circular detached unspotted/nonpulsating state"
            raise UnsupportedOccultationError(msg)
        stars = [getattr(system, name) for name in _COMPONENTS]
        vertices, width = max(len(s.points) for s in stars), max(len(s.faces) for s in stars)
        if vertices > _MAX_VERTICES or width > _MAX_FACES:
            msg = "Device occultation mesh exceeds bounded vertex/face capacity; use full CPU"
            raise UnsupportedOccultationError(msg)
        points, normals = np.zeros((2, vertices, 3)), np.zeros((2, width, 3))
        faces = np.zeros((2, width, 3), dtype=np.int64)
        areas = np.zeros((2, width))
        if device_radiance is None:
            radiance = np.zeros((2, len(bands), width))
            ncoeff = len(stars[0].ld_cfs[bands[0]][0])
            coefficients = np.zeros((2, len(bands), width, ncoeff))
        thresholds = np.array([2 * np.pi * s.polar_radius**2 / len(s.faces) / 1e6 for s in stars])
        for comp, star in enumerate(stars):
            points[comp, :len(star.points)], normals[comp, :len(star.faces)] = star.points, star.normals
            faces[comp, :len(star.faces)], areas[comp, :len(star.faces)] = star.faces, star.areas
            if device_radiance is None:
                for band_index, band in enumerate(bands):
                    radiance[comp, band_index, :len(star.faces)] = star.normal_radiance[band]
                    coefficients[comp, band_index, :len(star.faces)] = star.ld_cfs[band]
        arrays = (points, normals, faces, areas, thresholds, np.array(system.semi_major_axis**2))
        if not all(np.isfinite(x).all() for x in arrays):
            msg = "Nonfinite/negative fresh surface state is unsupported for device occultation"
            raise UnsupportedOccultationError(msg)
        points, normals, faces, areas, thresholds, scale = (
            self._jax.device_put(x, self.device, may_alias=False) for x in arrays
        )
        if device_radiance is None:
            if not np.isfinite(radiance).all() or np.any(radiance < 0) or not np.isfinite(coefficients).all():
                msg = "Nonfinite/negative fresh radiance/LD is unsupported for device occultation"
                raise UnsupportedOccultationError(msg)
            radiance, coefficients = (self._jax.device_put(x, self.device, may_alias=False)
                                      for x in (radiance, coefficients))
        else:
            # Keep fresh spectral output on device through all eclipse packets.
            radiance, coefficients = device_radiance["normal_radiance"], device_radiance["coefficients"]
            if (radiance.shape != (2, len(bands), width) or coefficients.shape[:3] != radiance.shape
                    or any(x.device != self.device or x.dtype != np.float64 for x in (radiance, coefficients))):
                msg = "Device radiance/LD must have matching padded shapes, FP64 and device"
                raise ValueError(msg)
        return (points, normals, faces, areas, radiance, coefficients, thresholds, scale), stars

    @staticmethod
    def _selectors(
        projected: dict[str, np.ndarray], stars: list[Any], motion: Any, eclipses: Any, width: int,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        full = np.zeros((len(motion) * 2, width), dtype=bool)
        hulls, partials = [], []
        for index, position in enumerate(motion):
            front = 1 if 0 < position.azimuth < np.pi else 0
            views = []
            for comp, star in enumerate(stars):
                indices = np.flatnonzero(projected["mu"][index, comp, :len(star.faces)] > 0)
                full[2 * index + comp, indices] = True
                views.append(SimpleNamespace(
                    points=projected["points"][index, comp, :len(star.points)],
                    normals=projected["normals"][index, comp, :len(star.faces)],
                    indices=indices, faces=star.faces, equivalent_radius=star.equivalent_radius,
                    discretization_factor=star.discretization_factor,
                ))
            current_hull, current_partial = np.empty((0, 2)), np.empty(0, dtype=int)
            if eclipses[index]:
                cover, under = views[front], views[1 - front]
                visible_points = np.unique(cover.faces[cover.indices])
                boundary = cpu_coverage.get_eclipse_boundary_path(cover.points[visible_points, 1:])
                under_indices = np.unique(under.faces[under.indices])
                args = under.points[under_indices, 1:], under, under_indices, boundary
                similar_size = cpu_coverage.test_size_similarity(cover, under)
                selector = (cpu_coverage.visibility_disimilar_objects if similar_size
                            else cpu_coverage.visibility_similar_objects)
                visible, _, partial = selector(*args)
                full[2 * index + 1 - front, :len(under.faces)] = visible
                current_hull, current_partial = boundary.vertices.copy(), np.flatnonzero(partial)
            for comp in range(2):
                hulls.append(current_hull if comp != front else np.empty((0, 2)))
                partials.append(current_partial if comp != front else np.empty(0, dtype=int))
        return full, hulls, partials

    def __call__(  # noqa: C901, PLR0915 - explicit timed host/device stages
        self, system: Any, motion: Any, in_eclipse: Any, bands: tuple[str, ...], *, diagnostics: bool = False,
        device_radiance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        started = perf_counter()
        if not 1 <= len(motion) <= _MAX_PHASES:
            msg = "Device projection supports up to 128 synthesis phases per curve; use full CPU"
            raise UnsupportedOccultationError(msg)
        surface, stars = self._snapshot(system, bands, device_radiance)
        points, normals, faces, areas, radiance, coefficients, thresholds, scale = surface
        count = len(motion)
        padded_count = (count + _CHUNK - 1) // _CHUNK * _CHUNK
        azimuth = np.zeros(padded_count)
        azimuth[:count] = [p.azimuth for p in motion]
        angles = (self._jax.device_put(azimuth, self.device),
                  self._jax.device_put(np.array(system.inclination), self.device))
        projection = self._project(points, normals, *angles)
        projected = self._jax.tree.map(np.asarray, projection)  # deliberate CPU hull bridge
        projection_done = perf_counter()
        width = areas.shape[1]
        full, hulls, partials = self._selectors(projected, stars, motion, in_eclipse, width)
        hull_width = max([3, *(len(h) for h in hulls)])
        partial_width = max([1, *(len(p) for p in partials)])
        if hull_width > _MAX_HULL or partial_width > _MAX_PARTIAL:
            msg = "Device hull/partial-face capacity exceeded; use full CPU without a prior penalty"
            raise UnsupportedOccultationError(msg)
        # Power-of-two capacities reuse compiles across phases/candidates.
        hull_width, partial_width = 2**(hull_width - 1).bit_length(), 2**(partial_width - 1).bit_length()
        polygons = np.zeros((padded_count * 2, hull_width, 2))
        counts = np.zeros(padded_count * 2, dtype=np.int32)
        partial = np.zeros((padded_count * 2, partial_width), dtype=np.int64)
        part_valid = np.zeros(partial.shape, dtype=bool)
        full = np.pad(full, ((0, (padded_count - count) * 2), (0, 0)))
        for i, (hull, selected) in enumerate(zip(hulls, partials, strict=True)):
            polygons[i, :len(hull)], counts[i] = hull, len(hull)
            partial[i, :len(selected)], part_valid[i, :len(selected)] = selected, True
        selection_done = perf_counter()
        components = {band: np.empty((count, 2)) for band in bands}
        packets, captures, records = [], [], []
        for first in range(0, padded_count, _CHUNK):
            rows = slice(2 * first, 2 * (first + _CHUNK))
            params = tuple(self._jax.device_put(x[rows], self.device, may_alias=False)
                           for x in (polygons, counts, partial, part_valid, full))
            packet = PreparedOccultationBatch(self, (
                projection["points"][first:first + _CHUNK].reshape((-1, points.shape[1], 3)),
                projection["mu"][first:first + _CHUNK].reshape((-1, width)),
                faces, areas, radiance, coefficients, *params, thresholds, scale,
            ))
            result = packet.evaluate()
            if not np.asarray(result["row_valid"]).all():
                msg = "Device partial-face clipping/integration returned unsupported numeric state; use full CPU"
                raise UnsupportedOccultationError(msg)
            totals = np.asarray(result["component_flux"])
            used = min(_CHUNK, count - first)
            for band_index, band in enumerate(bands):
                components[band][first:first + used] = totals[:used, :, band_index]
            if diagnostics:
                packets.append(packet)
                host = self._jax.tree.map(np.asarray, result)
                records.append({"partial_counts": part_valid[rows].sum(axis=1), "hull_counts": counts[rows],
                                "full_visible": full[rows], "partial_indices": partial[rows],
                                "partial_valid": part_valid[rows], "result": host})
                for i in range(used):
                    at = first + i
                    on_pos = move_sys_onpos(system, motion[at], on_copy=True)
                    for comp, name in enumerate(_COMPONENTS):
                        star = getattr(on_pos, name)
                        star.points = projected["points"][at, comp, :len(star.points)].copy()
                        star.normals = projected["normals"][at, comp, :len(star.faces)].copy()
                        star.los_cosines = projected["mu"][at, comp, :len(star.faces)].copy()
                        star.indices = np.flatnonzero(star.los_cosines > 0)
                        star.coverage = host["coverage"][2 * i + comp, :len(star.faces)].copy()
                    captures.append(on_pos)
        return {"components": components, "positions": captures if diagnostics else None,
                "timings": {"device_projection_upload_download": projection_done - started,
                            "cpu_hull_visibility": selection_done - projection_done,
                            "device_clipping_integration_transfers": perf_counter() - selection_done},
                "details": {"hull_bridge": "CPU Qhull/Path/ELISa selectors on fresh device projection",
                            "phase_chunk": _CHUNK, "hull_capacity": hull_width, "partial_capacity": partial_width,
                            "partial_faces": int(part_valid.sum()), "prepared_batches": packets,
                            "records": records} if diagnostics else None}
