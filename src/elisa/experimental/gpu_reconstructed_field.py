"""Conservative C0 reconstruction of fresh emitted-intensity/LD moments.

Cell fields remain original thermal/reflection/radiance evaluations. A positive
curved-area lumped projection maps their intensity moments to shared vertices;
positive gnomonic shape functions then reconstruct a continuous field. Every
moment's surface integral is preserved under the declared order4 area rule.
This is a new finite field approximation, not a continuum or fitting certificate.
"""

# ruff: noqa: PLR0915, PLR2004, SLF001 - bounded experimental kernels
from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from elisa.base.curves.utils import calculate_surface_element_fluxes
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_batched_geometry import MAX_NODES, BatchedRocheGeometry

_EPS = np.finfo(float).eps


def shape_rule() -> tuple:
    x, w = leggauss(4)
    s, t = np.meshgrid((x + 1) / 2, (x + 1) / 2, indexing="ij")
    return np.stack([1 - s - (1 - s) * t, s, (1 - s) * t], axis=-1).reshape(-1, 3), (
        np.outer(w / 2, w / 2) * (1 - s)
    ).reshape(-1)


def incidence(faces: np.ndarray) -> tuple:
    _, compressed = np.unique(faces, return_inverse=True)
    compressed = compressed.ravel()
    faces = compressed.reshape(faces.shape)
    count = np.bincount(compressed)
    if count.max() > 64:
        msg = "Reconstruction vertex incidence exceeds declared64slot resource cap"
        raise UnsupportedCurvedGeometryError(msg)
    slots = np.zeros((len(count), int(count.max())), dtype=np.int32)
    active = np.zeros_like(slots, dtype=bool)
    order = np.argsort(compressed, kind="stable")
    start = np.repeat(np.cumsum(count) - count, count)
    slots[compressed[order], np.arange(len(order)) - start] = order
    active[compressed[order], np.arange(len(order)) - start] = True
    return faces, slots, active


def project_moments(weights: np.ndarray, moments: np.ndarray, slots: np.ndarray, active: np.ndarray) -> tuple:
    w = np.where(active, weights.reshape(-1)[slots], 0.0)
    mass = w.sum(axis=1)
    values = (w[..., None, None] * moments[slots // 3]).sum(axis=1) / mass[:, None, None]
    return values, mass


def reference_barycentric(
    triangles: np.ndarray, condition: np.ndarray, parents: np.ndarray, rays: np.ndarray,
) -> tuple:
    c = np.linalg.solve(triangles[parents].transpose(0, 2, 1), rays[..., None])[..., 0]
    guard = 64 * _EPS * condition[parents] * np.maximum(1.0, np.abs(c).max(axis=1))
    if not np.isfinite(c).all() or np.any(c < -guard[:, None]) or np.any(c.sum(axis=1) <= 0):
        msg = "Reconstruction node escaped its declared angular parent cone"
        raise UnsupportedCurvedGeometryError(msg)
    correction = float(np.maximum(-c, 0).max(initial=0.0))
    # Mathematical positive coordinates: correct only the checked FP64 cone
    # roundoff. Never discard a node, negative physical field or surface patch.
    c = np.maximum(c, 0.0)
    return c / c.sum(axis=1)[:, None], correction


class MomentFieldBackend:
    """Compile geometry/projection/interpolation/reduction once per device/shape."""

    def __init__(self, component: str, law: str, *, device: str, max_nodes: int = MAX_NODES) -> None:
        if component not in ("primary", "secondary") or law not in ("linear", "cosine", "logarithmic", "square_root"):
            msg = "Explicit supported component and LD law required"
            raise ValueError(msg)
        if (
            device not in ("reference", "cpu", "gpu")
            or isinstance(max_nodes, bool)
            or not isinstance(max_nodes, int)
            or not 1 <= max_nodes <= MAX_NODES
        ):
            msg = "Explicit backend and positive bounded node cap required"
            raise ValueError(msg)
        self.component, self.law, self.max_nodes = component, law, max_nodes
        self.mode = device
        self.stats = {}
        if device == "reference":
            return
        geometry = BatchedRocheGeometry(component, device=device, max_nodes=max_nodes, root_unroll=8)
        self.geometry = geometry
        self.jax, self.jnp, self.device = geometry.jax, geometry.jnp, geometry.device
        jax, jnp = self.jax, self.jnp
        barrier = jax.lax.optimization_barrier
        bary, quad = shape_rule()

        def prepare(parameters: Any, triangles: Any, areas: Any, moments: Any, slots: Any, active: Any) -> dict:
            v = jnp.einsum("qj,fjk->fqk", jnp.asarray(bary), triangles)
            sample = geometry.sample_kernel(
                parameters[None],
                v.reshape(1, -1, 3),
                jnp.ones((1, v.shape[0] * 16), dtype=bool),
            )
            jac = sample["area_per_solid_angle"].reshape(v.shape[:2])
            det = jnp.abs(jnp.sum(triangles[:, 0] * jnp.cross(triangles[:, 1], triangles[:, 2]), axis=1))
            w = det[:, None] * jnp.asarray(quad)[None] / jnp.linalg.norm(v, axis=2) ** 3 * jac
            raw = barrier(jnp.einsum("fq,qj->fj", w, jnp.asarray(bary)))
            error = jnp.max(jnp.abs(raw.sum(axis=1) / areas - 1))
            weights = barrier(raw * (areas / raw.sum(axis=1))[:, None])
            selected = jnp.where(active, weights.reshape(-1)[slots], 0.0)
            mass = barrier(selected.sum(axis=1))
            vertex = barrier(jnp.sum(selected[..., None, None] * moments[slots // 3], axis=1) / mass[:, None, None])
            source = jnp.sum(areas[:, None, None] * moments, axis=0)
            result = jnp.sum(mass[:, None, None] * vertex, axis=0)
            scale = jnp.sum(areas[:, None, None] * jnp.abs(moments), axis=0)
            closure = jnp.max(jnp.abs(result - source) / jnp.where(scale > 0, scale, 1.0))
            valid = sample["proposal_valid"][0] & (error <= 3e-13) & (closure <= 3e-13)
            valid &= jnp.all(jnp.isfinite(vertex)) & jnp.all(mass > 0)
            return {"vertex": vertex, "area_error": error, "moment_closure": closure, "valid": valid}

        self._prepare = jax.jit(prepare)

        def nodes(inverse: Any, condition: Any, faces: Any, vertex: Any, parent: Any, rays: Any) -> dict:
            c = barrier(jnp.einsum("nij,nj->ni", inverse[parent], rays))
            guard = 64 * _EPS * condition[parent] * jnp.maximum(1.0, jnp.max(jnp.abs(c), axis=1))
            valid = jnp.all(jnp.isfinite(c)) & jnp.all(c >= -guard[:, None]) & jnp.all(c.sum(axis=1) > 0)
            correction = jnp.max(jnp.maximum(-c, 0.0))
            c = jnp.maximum(c, 0.0)
            c = barrier(c / c.sum(axis=1)[:, None])
            values = barrier(jnp.sum(c[..., None, None] * vertex[faces[parent]], axis=1))
            return {"moments": values, "valid": valid, "roundoff_correction": correction}

        self._nodes = jax.jit(nodes)

        def flux(
            inverse: Any,
            condition: Any,
            faces: Any,
            vertex: Any,
            parent: Any,
            rays: Any,
            mu: Any,
            area: Any,
        ) -> dict:
            values = nodes(inverse, condition, faces, vertex, parent, rays)
            m = values["moments"]
            radiance = m[..., 0]
            coefficients = barrier(m[..., 1:] / jnp.where(radiance > 0, radiance, 1.0)[..., None])
            positive = jnp.maximum(mu, 0.0)[:, None]
            a = coefficients[..., 0]
            factor = barrier(1 - a + a * positive)
            if law == "logarithmic":
                safe = jnp.where(positive > 0, positive, 1.0)
                factor = barrier(factor - coefficients[..., 1] * positive * jnp.log(safe))
            elif law == "square_root":
                factor = barrier(factor - coefficients[..., 1] * (1 - jnp.sqrt(positive)))
            factor = jnp.where(mu[:, None] > 0, factor, 0.0)
            intensity = barrier(radiance * barrier((mu * area)[:, None] * factor))
            return {
                "flux": jnp.sum(intensity, axis=0),
                "valid": values["valid"] & jnp.all(jnp.isfinite(intensity)),
                "roundoff_correction": values["roundoff_correction"],
            }

        self._flux = jax.jit(flux)

    def prepare(self, binary: Any, star: Any, bands: tuple) -> Any:
        start = perf_counter()
        centre = np.array([0.0 if self.component == "primary" else 1.0, 0.0, 0.0])
        rays = star.points - centre
        rays = rays / np.linalg.norm(rays, axis=1)[:, None]
        triangles = rays[star.faces]
        if len(triangles) * 16 > self.max_nodes:
            msg = "Reconstruction area quadrature exceeds node cap before allocation"
            raise UnsupportedCurvedGeometryError(msg)
        condition = np.linalg.cond(triangles)
        if not np.isfinite(condition).all() or condition.max() > 1e5:
            msg = "Reconstruction angular matrix exceeds conditioning cap"
            raise UnsupportedCurvedGeometryError(msg)
        faces, slots, active = incidence(star.faces)
        r = np.stack([star.normal_radiance[b] for b in bands], axis=1)
        c = np.stack([star.ld_cfs[b] for b in bands], axis=1)
        expected = 1 if self.law in ("linear", "cosine") else 2
        if r.shape != (len(triangles), len(bands)) or c.shape != (*r.shape, expected):
            msg = "Reconstruction source LD/radiance shape does not match the declared law"
            raise UnsupportedCurvedGeometryError(
                msg,
            )
        moments = np.concatenate([r[..., None], r[..., None] * c], axis=2)
        if not np.isfinite(moments).all() or np.any(r < 0) or np.any(star.areas <= 0):
            msg = "Unsupported finite nonnegative source radiance/area"
            raise UnsupportedCurvedGeometryError(msg)
        inverse = np.linalg.inv(triangles.transpose(0, 2, 1))
        if self.mode == "reference":
            nodes, weights = duffy_nodes(triangles, 4)
            jac = ConditionedRocheGeometry(binary, self.component).sample(nodes)[2]
            raw = np.einsum("fq,qj->fj", (weights * jac).reshape(-1, 16), shape_rule()[0])
            error = float(np.max(np.abs(raw.sum(axis=1) / star.areas - 1)))
            if error > 3e-13:
                msg = "Reconstruction area disagrees with original curved cell area"
                raise UnsupportedCurvedGeometryError(msg)
            w = raw * (star.areas / raw.sum(axis=1))[:, None]
            vertex, mass = project_moments(w, moments, slots, active)
            total = np.sum(star.areas[:, None, None] * moments, axis=0)
            scale = np.sum(star.areas[:, None, None] * np.abs(moments), axis=0)
            closure = float(
                np.max(abs((mass[:, None, None] * vertex).sum(axis=0) - total) / np.where(scale > 0, scale, 1.0)),
            )
            if not np.isfinite(vertex).all() or closure > 3e-13:
                msg = "Reconstruction moment conservation failed"
                raise UnsupportedCurvedGeometryError(msg)
        else:
            if not self.jax.config.x64_enabled:
                msg = "JAX FP64 must remain enabled"
                raise RuntimeError(msg)
            parameters = np.array([binary.mass_ratio, star.synchronicity, star.surface_potential])
            placed = [
                self.jax.device_put(a, self.device, may_alias=False)
                for a in [parameters, triangles, star.areas, moments, slots, active]
            ]
            out = self._prepare(*placed)
            if not bool(out["valid"]):
                msg = "Device reconstruction geometry/conservation failed outside priors"
                raise UnsupportedCurvedGeometryError(
                    msg,
                )
            vertex = out["vertex"]
            error = float(out["area_error"])
            closure = float(out["moment_closure"])
        control = {
            "area_relative_error": error,
            "moment_integral_relative_error": closure,
            "maximum_condition": float(condition.max()),
            "source_faces": len(triangles),
            "vertices": len(vertex),
            "bands": len(bands),
            "area_order": 4,
            "field_approximation": "conservative positive C0 emitted-intensity moments",
        }
        self.stats["prepare_seconds"] = self.stats.get("prepare_seconds", 0.0) + perf_counter() - start
        return PreparedMomentField(self, triangles, faces, inverse, condition, vertex, bands, control)


class PreparedMomentField:
    def __init__(
        self,
        backend: MomentFieldBackend,
        triangles: Any,
        faces: Any,
        inverse: Any,
        condition: Any,
        vertex: Any,
        bands: tuple,
        control: dict,
    ) -> None:
        self.backend, self.triangles, self.bands, self.control = backend, triangles, bands, control
        self.faces, self.vertex = faces, vertex
        self.arrays = (inverse, condition, faces, vertex)
        if backend.mode != "reference":
            self.arrays = tuple(backend.jax.device_put(a, backend.device) for a in self.arrays)

    def validate(self, parent: Any, rays: Any) -> tuple:
        b = self.backend
        p = np.asarray(parent)
        r = np.asarray(rays, dtype=float)
        if not np.issubdtype(p.dtype, np.integer) or p.ndim != 1 or r.shape != (len(p), 3) or not np.isfinite(r).all():
            msg = "Reconstruction requires finite node rays and integer parents"
            raise UnsupportedCurvedGeometryError(msg)
        if len(p) > b.max_nodes or np.any((p < 0) | (p >= len(self.triangles))):
            msg = "Reconstruction node cap or parent domain exceeded"
            raise UnsupportedCurvedGeometryError(msg)
        return p, r

    def sample(self, parent: Any, rays: Any) -> np.ndarray:
        p, r = self.validate(parent, rays)
        if not len(p):
            return np.empty((0, len(self.bands), self.vertex.shape[-1]))
        b = self.backend
        if b.mode == "reference":
            c, _ = reference_barycentric(self.triangles, self.arrays[1], p, r)
            return np.sum(c[..., None, None] * self.vertex[self.faces[p]], axis=1)
        if not b.jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        out = b._nodes(*self.arrays, b.jax.device_put(p, b.device), b.jax.device_put(r, b.device))
        if not bool(out["valid"]):
            msg = "Device reconstruction node escaped parent cone"
            raise UnsupportedCurvedGeometryError(msg)
        return np.asarray(out["moments"])

    def flux(self, parent: Any, rays: Any, mu: Any, area: Any) -> np.ndarray:
        start = perf_counter()
        p, r = self.validate(parent, rays)
        b = self.backend
        mu, area = np.asarray(mu), np.asarray(area)
        if (
            mu.shape != p.shape
            or area.shape != p.shape
            or not np.isfinite(mu + area).all()
            or np.any(area < 0)
            or np.any(abs(mu) > 1 + 1e-12)
        ):
            msg = "Reconstruction flux needs matching finite physical mu/area"
            raise UnsupportedCurvedGeometryError(msg)
        if not len(p):
            return np.zeros(len(self.bands))
        if b.mode == "reference":
            m = self.sample(p, r)
            radiance = m[..., 0]
            c = m[..., 1:] / np.where(radiance > 0, radiance, 1.0)[..., None]
            star = SimpleNamespace(
                indices=np.arange(len(p)),
                los_cosines=mu,
                coverage=area,
                normal_radiance={band: radiance[:, i] for i, band in enumerate(self.bands)},
                ld_cfs={band: c[:, i] for i, band in enumerate(self.bands)},
            )
            result = np.array([np.sum(calculate_surface_element_fluxes(band, star)) for band in self.bands])
        else:
            if not b.jax.config.x64_enabled:
                msg = "JAX FP64 must remain enabled"
                raise RuntimeError(msg)
            pad = 1 << max(0, len(p) - 1).bit_length()
            if pad > b.max_nodes:
                msg = "Reconstruction padded node cap exceeded"
                raise UnsupportedCurvedGeometryError(msg)
            p = np.pad(p, (0, pad - len(p)), mode="edge")
            r = np.pad(r, ((0, pad - len(r)), (0, 0)), mode="edge")
            mu = np.pad(mu, (0, pad - len(mu)))
            area = np.pad(area, (0, pad - len(area)))
            out = b._flux(*self.arrays, *(b.jax.device_put(a, b.device, may_alias=False) for a in [p, r, mu, area]))
            if not bool(out["valid"]):
                msg = "Device reconstruction flux failed outside priors"
                raise UnsupportedCurvedGeometryError(msg)
            result = np.asarray(out["flux"])
        if not np.isfinite(result).all():
            msg = "Reconstruction flux reduction is nonfinite"
            raise UnsupportedCurvedGeometryError(msg)
        b.stats["integration_calls"] = b.stats.get("integration_calls", 0) + 1
        b.stats["integration_seconds"] = b.stats.get("integration_seconds", 0.0) + perf_counter() - start
        return result
