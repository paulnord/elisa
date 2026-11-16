"""Device-resident changing Roche rays, triangle geometry and effective gravity.

This composes the tested physics primitives without intermediate host copies.
Optional intrinsic temperatures are recomputed from changing gravity/areas.
It does not supply candidate-dependent topology, mesh correction or occultation.
No stale brightness/coverage template is accepted here.
"""

# ruff: noqa: SLF001 - package-internal composition of device kernels
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa.experimental import gpu_roche as roche
from elisa.experimental.gpu_gravity import RocheSurfaceGravity
from elisa.experimental.gpu_mesh import TriangleGeometry
from elisa.experimental.gpu_thermal import GravityDarkeningTemperature

_XYZ = 3
_FACE_NDIM = 2


@dataclass(frozen=True)
class PreparedRocheBatch:
    """Immutable device parameter snapshot sharing the owner's fixed topology."""

    _owner: Any
    _parameters: tuple[Any, ...]
    _thermal: bool = False

    def evaluate(self) -> dict[str, Any]:
        """Run all physics on device, returning asynchronous device arrays.

        Check proposal_valid and the per-stage masks before consuming results.
        There are no NumPy conversions, validation copies or host transfers in
        this method. Initial compilation and final downloads remain caller costs.
        """
        self._owner._check_precision()
        evaluate = self._owner._evaluate_thermal if self._thermal else self._owner._evaluate
        return evaluate(*self._parameters, *self._owner._topology)

    def block_until_ready(self) -> PreparedRocheBatch:
        """Synchronize input transfers without evaluating or compiling physics."""
        for array in (*self._parameters, *self._owner._topology):
            array.block_until_ready()
        return self

    @property
    def nbytes(self) -> int:
        """Parameter bytes only; excludes shared topology, outputs and workspace."""
        return sum(x.nbytes for x in self._parameters)

    @property
    def proposals(self) -> int:
        """Return the number of independent physical parameter rows."""
        return self._parameters[0].shape[0]


class RocheGeometryPipeline:
    """Compile changing radii -> binary-frame vertices -> geometry and gravity.

    Directions/topology are immutable host snapshots transferred once. Each
    prepared proposal supplies q, spin, potential, primary mass (kg), SMA (m),
    azimuth and inclination (radians). q=M2/M1 for both components. Ray and
    conditioning restrictions are inherited unchanged from the three kernels.
    Unsupported masks mean CPU fallback/explicit failure, not a physical prior.

    Geometry areas are SMA**2; gravity is m/s**2. No mesh correction is applied.
    Centres/normals and gravity therefore describe the same supplied ray mesh,
    not the final cached state of a fully built CPU container. This is a bounded
    fixed-direction/topology primitive, not a complete changing stellar model.
    Storage scales as proposals times vertices/faces; callers must chunk.
    """

    def __init__(self, component: str, phi: Any, theta: Any, faces: Any, *, device: str = "gpu") -> None:
        ph, th = (np.array(x, dtype=np.float64, copy=True) for x in (phi, theta))
        f = np.array(faces, copy=True)
        if any(x.ndim != 1 or not x.size or not np.isfinite(x).all() for x in (ph, th)) or ph.shape != th.shape:
            msg = "Directions must be matching nonempty finite one-dimensional arrays"
            raise ValueError(msg)
        if np.any((ph < 0) | (ph > 2 * np.pi) | (th < 0) | (th > np.pi)):
            msg = "Angles require phi in [0,2pi] and theta in [0,pi]"
            raise ValueError(msg)
        if f.ndim != _FACE_NDIM or f.shape[-1] != _XYZ or not f.shape[0] or not np.issubdtype(f.dtype, np.integer):
            msg = "Faces must be nonempty integer triangle indices"
            raise ValueError(msg)
        if np.any(f < 0) or np.any(f >= ph.size) or np.any(np.diff(np.sort(f, axis=1), axis=1) == 0):
            msg = "Face index out of bounds or repeated within a triangle"
            raise ValueError(msg)
        rays = roche.RocheRaySolver(component, device=device)
        geometry = TriangleGeometry(device=device)
        gravity = RocheSurfaceGravity(component, device=device)
        self._jax, self.device = rays._jax, rays.device
        self._topology = tuple(self._jax.device_put(x, self.device, may_alias=False) for x in (ph, th, f))
        centre = 0.0 if component == "primary" else 1.0

        def evaluate(
            q: Any,
            spin: Any,
            potential: Any,
            mass: Any,
            axis: Any,
            az: Any,
            inc: Any,
            phi: Any,
            theta: Any,
            faces: Any,
        ) -> dict[str, Any]:
            # Keep the standalone stages' FP64 rounding boundaries while
            # retaining every array on device. Unrestricted fusion can move
            # ray arithmetic into sensitive small-triangle normal differences.
            ray = self._jax.lax.optimization_barrier(rays._solve(q, spin, potential, phi, theta))
            points = ray["points_local"]
            if component == "secondary":
                points = points.at[..., 0].set(1 - points[..., 0])
            points = self._jax.lax.optimization_barrier(points)
            mesh = geometry._geometry(points, faces, q * 0 + centre, az, inc)
            field = gravity._evaluate(points, faces, q, spin, mass, axis)
            return {
                "rays": ray,
                "points_binary": points,
                "geometry": mesh,
                "gravity": field,
                "proposal_valid": ray["proposal_valid"] & mesh["proposal_valid"] & field["proposal_valid"],
            }

        self._evaluate = self._jax.jit(evaluate)
        thermal = GravityDarkeningTemperature(device=device)

        def evaluate_thermal(
            q: Any,
            spin: Any,
            potential: Any,
            mass: Any,
            axis: Any,
            az: Any,
            inc: Any,
            t_eff: Any,
            beta: Any,
            phi: Any,
            theta: Any,
            faces: Any,
        ) -> dict[str, Any]:
            base = self._jax.lax.optimization_barrier(
                evaluate(q, spin, potential, mass, axis, az, inc, phi, theta, faces),
            )
            temperature = thermal._temperatures(base["gravity"]["face_gravity"], base["geometry"]["areas"], t_eff, beta)
            return dict(
                base, thermal=temperature, proposal_valid=base["proposal_valid"] & temperature["proposal_valid"],
            )

        self._evaluate_thermal = self._jax.jit(evaluate_thermal)

    def _check_precision(self) -> None:
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)

    @property
    def topology_nbytes(self) -> int:
        """Shared direction/connectivity bytes, excluding runtime workspace."""
        return sum(x.nbytes for x in self._topology)

    def block_until_ready(self) -> RocheGeometryPipeline:
        """Synchronize the one-time topology upload."""
        for array in self._topology:
            array.block_until_ready()
        return self

    def prepare(
        self,
        q: Any,
        synchronicity: Any,
        potential: Any,
        primary_mass_kg: Any,
        semi_major_axis_m: Any,
        azimuth: Any,
        inclination: Any,
    ) -> PreparedRocheBatch:
        """Validate/copy host parameters and transfer an immutable device batch.

        Every parameter must be a matching nonempty one-dimensional array.
        Invalid roots remain explicit unsupported outputs; they are not silently
        clipped, converted to zero flux or treated as impossible physical fits.
        """
        self._check_precision()
        arrays = tuple(
            np.array(x, dtype=np.float64, copy=True)
            for x in (q, synchronicity, potential, primary_mass_kg, semi_major_axis_m, azimuth, inclination)
        )
        if any(x.ndim != 1 or not x.size or not np.isfinite(x).all() or x.shape != arrays[0].shape for x in arrays):
            msg = "Parameters must be matching nonempty finite one-dimensional arrays"
            raise ValueError(msg)
        qq, spin, pp, mass, axis, az, inc = arrays
        if np.any((qq < roche._Q_MIN) | (qq > roche._Q_MAX) | (spin < roche._S_MIN) | (spin > roche._S_MAX)):
            msg = "Unsupported q or synchronicity domain"
            raise ValueError(msg)
        if np.any(pp <= 0) or np.any(mass <= 0) or np.any(axis <= 0):
            msg = "Potential, primary mass (kg) and semi-major axis (m) must be positive"
            raise ValueError(msg)
        if np.any(np.abs(az) > 2 * np.pi) or np.any((inc < 0) | (inc > np.pi)):
            msg = "Azimuth must lie in [-2pi,2pi] and inclination in [0,pi]"
            raise ValueError(msg)
        return PreparedRocheBatch(self, tuple(self._jax.device_put(x, self.device, may_alias=False) for x in arrays))

    def prepare_thermal(
        self,
        q: Any,
        synchronicity: Any,
        potential: Any,
        primary_mass_kg: Any,
        semi_major_axis_m: Any,
        azimuth: Any,
        inclination: Any,
        t_eff: Any,
        gravity_darkening: Any,
    ) -> PreparedRocheBatch:
        """Add fresh intrinsic gravity-darkened temperatures on the same device.

        No irradiation, atmosphere, LD or eclipse coverage is implied. The
        returned surface_flux is emitted bolometric W/m**2, not observed flux.
        """
        batch = self.prepare(q, synchronicity, potential, primary_mass_kg, semi_major_axis_m, azimuth, inclination)
        t, b = GravityDarkeningTemperature.validate_parameters(t_eff, gravity_darkening, batch.proposals)
        params = (*batch._parameters, *(self._jax.device_put(x, self.device, may_alias=False) for x in (t, b)))
        return PreparedRocheBatch(self, params, _thermal=True)

    def __call__(
        self,
        q: Any,
        synchronicity: Any,
        potential: Any,
        primary_mass_kg: Any,
        semi_major_axis_m: Any,
        azimuth: Any,
        inclination: Any,
    ) -> dict[str, Any]:
        """Host validation/transfer plus fused evaluation; no output download."""
        return self.prepare(
            q,
            synchronicity,
            potential,
            primary_mass_kg,
            semi_major_axis_m,
            azimuth,
            inclination,
        ).evaluate()
