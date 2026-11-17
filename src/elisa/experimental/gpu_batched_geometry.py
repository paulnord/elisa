"""Batched device Roche roots, normals and curved-area quadrature.

Port of a remaining CPU stage, not a complete light curve or MCMC backend.
Circular separation-one detached inner branches retain the existing ray domain.
Proposals may change q, spin, potential AND every supplied ray/triangle. Inputs
describe the caller's actual angular topology; this module does not freeze or
invent an ELISa mesh. All intermediate arithmetic and area reduction run on the
requested device; checked_host is an explicit validation/download boundary.
"""

from __future__ import annotations

# ruff: noqa: C901, PLR2004, PLR0915 - fixed physical domain and compiled geometry graph
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_likelihood import _jax_device

R_MIN, R_MAX = 1e-6, 0.25
MAX_NODES = 2097152
ROOT_ITERATIONS = 80


class BatchedRocheGeometry:
    """FP64 geometry for proposal rows [q, synchronicity, surface_potential].

    Vectors use the binary-frame axes relative to each star's own centre; the
    secondary's local +x points AWAY from the companion. This is the convention
    of RocheGeometry.sample, not RocheRaySolver.points_local. Points returned
    in binary coordinates add [1,0,0] for the secondary; local_points remain
    conditioned and resident for subsequent kernels.

    Domain q=[.5,2], spin=[.5,1.5], roots strictly inside [1e-6,.25]. An
    unsupported or numerically ambiguous ray has an explicit status and invalid
    proposal mask. These are not physical-prior exclusions. Padding is masked,
    never substituted into physical fluxes. No autodiff/HMC guarantee is made.
    The node cap is a workload cap, not a complete GPU memory bound.
    """

    def __init__(self, component: str, *, device: str = "gpu", max_nodes: int = MAX_NODES,
                 root_unroll: int = 1) -> None:
        if component not in ("primary", "secondary"):
            msg = "component must be primary or secondary"
            raise ValueError(msg)
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= MAX_NODES:
            msg = "max_nodes must be an integer in [1,2097152]"
            raise ValueError(msg)
        if isinstance(root_unroll, bool) or not isinstance(root_unroll, int) or root_unroll not in (1, 2, 4, 8):
            msg = "root_unroll must be1,2,4or8; mathematical iteration count remains80"
            raise ValueError(msg)
        self.component, self.max_nodes = component, max_nodes
        self.jax, self.jnp, self.device = _jax_device(device)
        jax, jnp = self.jax, self.jnp
        primary = component == "primary"
        shift = -1.0 if primary else 1.0
        centre = jax.device_put(np.array([0.0 if primary else 1.0, 0.0, 0.0], dtype=np.float64), self.device)

        def sample(parameters: Any, vectors: Any, active: Any) -> Any:
            q, spin, target = [parameters[:, i : i + 1] for i in range(3)]
            domain = jnp.all(jnp.isfinite(parameters), axis=1)[:, None]
            domain &= (q >= 0.5) & (q <= 2) & (spin >= 0.5) & (spin <= 1.5) & (target > 0)
            # Safe scratch arithmetic does not turn unsupported inputs into a
            # physical model: output masks/statuses still reject those rows.
            q = jnp.where(domain, q, 1.0)
            spin = jnp.where(domain, spin, 1.0)
            target = jnp.where(domain, target, 12.0)
            vector_norm = jnp.linalg.norm(vectors, axis=-1)
            vector_ok = jnp.all(jnp.isfinite(vectors), axis=-1) & (vector_norm > 0)
            direction = jnp.where(
                vector_ok[..., None],
                vectors / jnp.where(vector_ok, vector_norm, 1.0)[..., None],
                jnp.array([1.0, 0.0, 0.0]),
            )
            mass = q if not primary else jnp.ones_like(q)
            other = q if primary else jnp.ones_like(q)
            linear = -q if primary else jnp.ones_like(q)
            rotation = spin**2 * (1 + q)
            extra = jnp.zeros_like(q) if primary else 0.5 - 0.5 * q

            def potential(radius: Any) -> Any:
                separation = jnp.sqrt(1 + radius**2 + 2 * shift * radius * direction[..., 0])
                xy = direction[..., 0] ** 2 + direction[..., 1] ** 2
                return (
                    mass / radius
                    + other / separation
                    + linear * radius * direction[..., 0]
                    + 0.5 * rotation * radius**2 * xy
                    + extra
                )

            low = jnp.full(vector_norm.shape, R_MIN, dtype=jnp.float64)
            high = jnp.full(vector_norm.shape, R_MAX, dtype=jnp.float64)
            flo, fhi = potential(low) - target, potential(high) - target
            endpoint_tolerance = 32 * jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, target)
            ambiguous = (jnp.abs(flo) <= endpoint_tolerance) | (jnp.abs(fhi) <= endpoint_tolerance)
            bracket = (flo > endpoint_tolerance) & (fhi < -endpoint_tolerance) & jnp.isfinite(flo) & jnp.isfinite(fhi)

            def bisect(_iteration: Any, bounds: Any) -> Any:
                lo, hi = bounds
                mid = lo + 0.5 * (hi - lo)
                above = potential(mid) > target
                return jnp.where(above, mid, lo), jnp.where(above, hi, mid)

            low, high = jax.lax.fori_loop(0, ROOT_ITERATIONS, bisect, (low, high), unroll=root_unroll)
            radius = low + 0.5 * (high - low)
            residual = potential(radius) - target
            local = radius[..., None] * direction
            companion = local + jnp.array([shift, 0.0, 0.0])
            companion_norm = jnp.linalg.norm(companion, axis=-1)
            gradient = (
                mass[..., None] * local / radius[..., None] ** 3
                + other[..., None] * companion / companion_norm[..., None] ** 3
            )
            gradient -= rotation[..., None] * local * jnp.array([1.0, 1.0, 0.0])
            gradient += jnp.concatenate((-linear, jnp.zeros_like(q), jnp.zeros_like(q)), axis=1)[:, None, :]
            magnitude = jnp.linalg.norm(gradient, axis=-1)
            radial = jnp.sum(gradient * direction, axis=-1)
            branch = jnp.all(jnp.isfinite(gradient), axis=-1) & jnp.isfinite(magnitude) & (magnitude > 0) & (radial > 0)
            solved = jnp.isfinite(residual) & (jnp.abs(residual) <= 3e-13 * jnp.maximum(1.0, target))
            valid = domain & vector_ok & bracket & solved & branch & active
            status = jnp.where(
                ~domain,
                1,
                jnp.where(
                    ~vector_ok,
                    2,
                    jnp.where(ambiguous, 4, jnp.where(~bracket, 3, jnp.where(~solved, 5, jnp.where(~branch, 6, 0)))),
                ),
            )
            status = jnp.where(active, status, 7).astype(jnp.int32)
            normal = gradient / jnp.where(magnitude > 0, magnitude, 1.0)[..., None]
            jacobian = radius**2 * magnitude / jnp.where(radial > 0, radial, 1.0)

            def output(value: Any) -> Any:
                condition = valid if value.ndim == 2 else valid[..., None]
                enabled = active if value.ndim == 2 else active[..., None]
                return jnp.where(condition, value, jnp.where(enabled, jnp.nan, 0.0))

            return {
                "radii": output(radius),
                "local_points": output(local),
                "points": output(local + centre),
                "normals": output(normal),
                "outward_gradients": output(gradient),
                "area_per_solid_angle": output(jacobian),
                "potential_residual": jnp.where(active, residual, 0.0),
                "radius_interval_width": jnp.where(active, high - low, 0.0),
                "valid": valid,
                "status": status,
                "active": active,
                "proposal_valid": jnp.any(active, axis=1) & jnp.all(valid | ~active, axis=1),
            }

        self.sample_kernel = sample
        self._sample = jax.jit(sample)

        def compact(parameters: Any, vectors: Any, active: Any) -> Any:
            result = sample(parameters, vectors, active)
            return {key: result[key] for key in ("points", "normals", "area_per_solid_angle", "proposal_valid")}

        self._sample_compact = jax.jit(compact)
        self._areas = {}

        for order in (2, 3, 4):
            x, w = leggauss(order)
            s, t = np.meshgrid((x + 1) / 2, (x + 1) / 2, indexing="ij")
            bary = np.stack([1 - s - (1 - s) * t, s, (1 - s) * t], axis=-1).reshape((-1, 3))
            weights = (np.outer(w / 2, w / 2) * (1 - s)).reshape(-1)

            def make_area(barycentric: Any, quadrature_weights: Any, quadrature_size: Any) -> Any:
                # Constants live on the selected device with the compiled graph.
                bb = jax.device_put(barycentric, self.device)
                ww = jax.device_put(quadrature_weights, self.device)

                def integrate(parameters: Any, triangles: Any, face_active: Any) -> Any:
                    vectors = jnp.einsum("qj,pnjk->pnqk", bb, triangles)
                    active = jnp.repeat(face_active, quadrature_size, axis=1)
                    sampled = sample(parameters, vectors.reshape((parameters.shape[0], -1, 3)), active)
                    determinant = jnp.abs(
                        jnp.sum(triangles[:, :, 0] * jnp.cross(triangles[:, :, 1], triangles[:, :, 2]), axis=-1),
                    )
                    vector_norm = jnp.linalg.norm(vectors, axis=-1)
                    weights = determinant[..., None] * ww / jnp.where(vector_norm > 0, vector_norm, 1.0) ** 3
                    jacobian = sampled["area_per_solid_angle"].reshape(weights.shape)
                    node_valid = sampled["valid"].reshape(weights.shape)
                    finite = jnp.isfinite(weights).all(axis=-1) & (determinant > 0)
                    face_valid = finite & node_valid.all(axis=-1) & face_active
                    areas = jnp.sum(weights * jacobian, axis=-1)
                    areas = jnp.where(face_valid, areas, jnp.where(face_active, jnp.nan, 0.0))
                    return {
                        "areas": areas,
                        "total_area": jnp.sum(areas, axis=1),
                        "face_valid": face_valid,
                        "active": face_active,
                        "max_relative_potential_residual": jnp.max(
                            jnp.abs(sampled["potential_residual"]) / jnp.maximum(1.0, parameters[:, 2:3]), axis=1,
                        ),
                        "proposal_valid": jnp.any(face_active, axis=1) & jnp.all(face_valid | ~face_active, axis=1),
                    }

                return jax.jit(integrate)

            self._areas[order] = make_area(bary, weights, order**2)

    def _parameters(self, parameters: Any) -> Any:
        if not self.jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        if len(np.shape(parameters)) != 2 or np.shape(parameters)[1] != 3 or np.shape(parameters)[0] == 0:
            msg = "parameters must have shape (nonzero proposals,3)"
            raise ValueError(msg)
        return self.jnp.asarray(self.jax.device_put(parameters, self.device), dtype=self.jnp.float64)

    def _active(self, active: Any, shape: tuple[int, int]) -> Any:
        if active is None:
            with self.jax.default_device(self.device):
                return self.jnp.ones(shape, dtype=self.jnp.bool_)
        if np.shape(active) not in ((shape[1],), shape):
            msg = "active mask must have node/face shape or match the proposal batch"
            raise ValueError(msg)
        values = self.jnp.asarray(self.jax.device_put(active, self.device), dtype=self.jnp.bool_)
        return self.jnp.broadcast_to(values, shape)

    def sample(self, parameters: Any, vectors: Any, *, active: Any = None, compact: bool = False) -> Any:
        """Device arrays; shared (N,3) or changing per-proposal (P,N,3) rays.

        No array values are read back to the host. Status 0=valid,
        1=parameter domain, 2=bad ray, 3=unbracketed, 4=endpoint ambiguity,
        5=potential residual, 6=invalid branch, 7=inactive padding. A caller
        must check proposal_valid before consuming any physical result.
        """
        parameters = self._parameters(parameters)
        shape = np.shape(vectors)
        if len(shape) not in (2, 3) or shape[-1] != 3 or shape[-2] == 0:
            msg = "vectors must have shape (N,3) or (P,N,3)"
            raise ValueError(msg)
        proposals, nodes = parameters.shape[0], shape[-2]
        if len(shape) == 3 and shape[0] != proposals:
            msg = "per-proposal vector shape must match the parameter rows"
            raise ValueError(msg)
        if proposals * nodes > self.max_nodes:
            msg = "Batch node cap exceeded; chunk before device evaluation"
            raise ValueError(msg)
        vectors = self.jnp.asarray(self.jax.device_put(vectors, self.device), dtype=self.jnp.float64)
        vectors = self.jnp.broadcast_to(vectors, (proposals, nodes, 3))
        kernel = self._sample_compact if compact else self._sample
        return kernel(parameters, vectors, self._active(active, (proposals, nodes)))

    def areas(self, parameters: Any, triangles: Any, *, order: int = 4, active: Any = None) -> Any:
        """Device Duffy areas, with shared or changing per-proposal triangles.

        Triangles have shape (T,3,3) or (P,T,3,3). Areas use SMA squared,
        matching the CPU curved-geometry measure; SI conversion is separate.
        Only supplied patches are integrated: no closed-surface certificate.
        """
        parameters = self._parameters(parameters)
        if isinstance(order, bool) or order not in (2, 3, 4):
            msg = "Duffy order must be 2,3 or 4"
            raise ValueError(msg)
        shape = np.shape(triangles)
        if len(shape) not in (3, 4) or shape[-2:] != (3, 3) or shape[-3] == 0:
            msg = "triangles must have shape (T,3,3) or (P,T,3,3)"
            raise ValueError(msg)
        proposals, faces = parameters.shape[0], shape[-3]
        if len(shape) == 4 and shape[0] != proposals:
            msg = "per-proposal triangles must match parameter rows"
            raise ValueError(msg)
        if proposals * faces * order**2 > self.max_nodes:
            msg = "Batch quadrature cap exceeded; chunk before device evaluation"
            raise ValueError(msg)
        triangles = self.jnp.asarray(self.jax.device_put(triangles, self.device), dtype=self.jnp.float64)
        triangles = self.jnp.broadcast_to(triangles, (proposals, faces, 3, 3))
        return self._areas[order](parameters, triangles, self._active(active, (proposals, faces)))

    def checked_host(self, result: dict[str, Any]) -> Any:
        """Explicit synchronization/download; unsupported rows raise, never -inf."""
        host = self.jax.tree.map(np.asarray, result)
        if not np.all(host["proposal_valid"]):
            rows = np.flatnonzero(~host["proposal_valid"]).tolist()
            msg = f"Unsupported batched geometry rows {rows}; no prior penalty"
            raise UnsupportedCurvedGeometryError(msg)
        return host
