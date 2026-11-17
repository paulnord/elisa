"""CPU-compatible corrected, unspotted Roche surface state on an FP64 device.

Angular placement, radial roots and triangulation are supplied by the CPU mesh
builder for EACH candidate. This adapter preserves ELISa's symmetry and build
order; it does not turn the fixed-direction ray experiment into a whole model.
"""

# ruff: noqa: SLF001 - composition of the experimental device kernels
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa import const, utils
from elisa.binary_system.surface.mesh import _load_correction_factors
from elisa.experimental.gpu_gravity import RocheSurfaceGravity
from elisa.experimental.gpu_mesh import TriangleGeometry
from elisa.experimental.gpu_thermal import GravityDarkeningTemperature

_XYZ, _MESH_NDIM = 3, 2
_Q_MIN, _Q_MAX = 0.5, 2.0
_SPIN_MIN, _SPIN_MAX = 0.5, 1.5


@dataclass(frozen=True)
class PreparedSurfaceState:
    """Immutable inputs; evaluation has no intermediate host transfers."""

    _owner: Any
    _arrays: tuple[Any, ...]

    def evaluate(self) -> dict[str, Any]:
        if not self._owner._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        return self._owner._evaluate(*self._arrays)

    def block_until_ready(self) -> PreparedSurfaceState:
        for array in self._arrays:
            array.block_until_ready()
        return self

    @property
    def nbytes(self) -> int:
        """Uploaded input bytes, excluding outputs/runtime/compiler workspace."""
        return sum(x.nbytes for x in self._arrays)


class CorrectedSurfaceState:
    """Reconstruct the CPU's cached state from its pre-correction mesh.

    Centres, normals and SI gravity are calculated BEFORE point correction.
    Only points and areas are corrected. Intrinsic temperatures then use the
    corrected areas and cached gravity. The CPU's quarter-face mirroring is
    retained, including normal signs and secondary centre x=1. Restrictions of
    RocheSurfaceGravity and TriangleGeometry are unchanged. Unsupported output
    requires CPU fallback or explicit failure, never a physical-prior penalty.

    This API accepts a fresh unspotted detached CPU StarContainer after
    build_mesh/build_faces, BEFORE correct_mesh. Calling it on corrected points
    would apply the correction twice. The hybrid builder controls that sequence.
    """

    def __init__(self, component: str, *, device: str = "gpu") -> None:
        field = RocheSurfaceGravity(component, device=device)
        geometry = TriangleGeometry(device=device)
        thermal = GravityDarkeningTemperature(device=device)
        self._jax, self.device = field._jax, field.device
        self.component = component
        import jax.numpy as jnp  # noqa: PLC0415 - optional dependency

        barrier = self._jax.lax.optimization_barrier
        centre = 0.0 if component == "primary" else 1.0

        def evaluate(
            points: Any, faces: Any, mirror: Any, signs: Any, parameters: Any,
        ) -> dict[str, Any]:
            q, spin, mass, axis, factor, t_eff, beta = (parameters[i:i + 1] for i in range(7))
            vertices = points[None]
            raw = barrier(geometry._geometry(vertices, faces, q * 0 + centre, q * 0, q * 0 + jnp.pi / 2))
            gravity = barrier(field._evaluate(vertices, faces, q, spin, mass, axis))
            # CPU gravity/areas mirror the base patch; normals concatenate the
            # same patch with y/z signs. Face centres use ALL original vertices.
            normal = raw["normals"][0, mirror] * signs
            face_gravity = gravity["face_gravity"][:, mirror]
            com = jnp.array([centre, 0.0, 0.0])
            centered = barrier(points - com)
            corrected = barrier(factor[0] * centered) + com
            corrected = barrier(corrected)
            triangles = corrected[faces]
            areas = 0.5 * jnp.linalg.norm(
                jnp.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1,
            )
            areas = barrier(areas[mirror])
            temperature = thermal._temperatures(face_gravity, areas[None], t_eff, beta)
            temperature = barrier(temperature)
            # The full CPU build explicitly renormalizes even without spots.
            t = temperature["temperatures"][0]
            coefficient = (jnp.sum(areas) * t_eff[0]**4 / jnp.sum(areas * t**4))**0.25
            t = barrier(t * coefficient)
            valid = (
                raw["proposal_valid"][0] & gravity["proposal_valid"][0]
                & temperature["proposal_valid"][0] & jnp.all(jnp.isfinite(corrected))
                & jnp.all(jnp.isfinite(t) & (t > 0))
            )
            return {
                "points": corrected,
                "face_centres": raw["centres"][0],
                "normals": normal,
                "areas": areas,
                "potential_gradient_magnitudes": face_gravity[0],
                "log_g": jnp.log10(face_gravity[0]),
                "temperatures": jnp.where(valid, t, jnp.nan),
                "intrinsic_surface_flux": jnp.where(valid, const.STEFAN_BOLTZMAN_CONST * t**4, jnp.nan),
                "luminosity_ratio": jnp.sum(areas * (t / t_eff[0])**4) / jnp.sum(areas),
                "geometry_valid": raw["proposal_valid"][0],
                "gravity_status": gravity["status"][0],
                "thermal_status": temperature["status"][0],
                "proposal_valid": valid,
            }

        self._evaluate = self._jax.jit(evaluate)

    def prepare(
        self, star: Any, mass_ratio: float, primary_mass_kg: float, semi_major_axis_m: float,
    ) -> PreparedSurfaceState:
        """Snapshot CPU mesh, symmetry and actual detached correction resource."""
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        if not star.symmetry_test():
            msg = "Corrected state requires an unspotted, nonpulsating symmetry mesh"
            raise ValueError(msg)
        points, faces = np.array(star.points, dtype=np.float64, copy=True), np.array(star.faces, copy=True)
        nbase = int(star.base_symmetry_faces_number)
        mirror = np.array(star.face_symmetry_vector, dtype=np.int64, copy=True)
        if (
            points.ndim != _MESH_NDIM or points.shape[1] != _XYZ or not len(points) or not np.isfinite(points).all()
            or faces.shape != (4 * nbase, 3) or nbase <= 0 or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(points))
            or not np.array_equal(mirror, np.tile(np.arange(nbase), 4))
        ):
            msg = "Expected the CPU detached quarter-mirrored triangle mesh"
            raise ValueError(msg)
        factor = utils.discretization_correction_factor(
            star.discretization_factor, _load_correction_factors()["detached"],
        )
        parameters = np.array([
            mass_ratio, star.synchronicity, primary_mass_kg, semi_major_axis_m,
            factor, star.t_eff, star.gravity_darkening,
        ], dtype=np.float64)
        if not np.isfinite(parameters).all() or np.any(parameters[2:6] <= 0):
            msg = "Surface parameters and correction must be finite and positive"
            raise ValueError(msg)
        if not (_Q_MIN <= mass_ratio <= _Q_MAX and _SPIN_MIN <= star.synchronicity <= _SPIN_MAX):
            msg = "Unsupported q/spin domain for corrected Roche state"
            raise ValueError(msg)
        GravityDarkeningTemperature.validate_parameters([star.t_eff], [star.gravity_darkening], 1)
        signs = np.repeat(np.array([[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]]), nbase, axis=0)
        return PreparedSurfaceState(
            self, tuple(self._jax.device_put(x, self.device, may_alias=False)
                        for x in (points, faces, mirror, signs, parameters)),
        )
