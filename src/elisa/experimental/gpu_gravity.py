"""Effective gravity on supplied changing Roche vertices, in ELISa's units.

This bounded primitive does not update topology, coverage, temperature or flux.
Unsupported outputs require explicit failure or CPU fallback, never an implicit
physical-prior rejection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from elisa import const
from elisa.experimental.gpu_likelihood import _jax_device

_XYZ = 3
_FACE_NDIM = 2
_MIN_RADIUS, _MAX_RADIUS = 1e-6, 0.25
_Q_MIN, _Q_MAX = 0.5, 2.0
_SPIN_MIN, _SPIN_MAX = 0.5, 1.5


class RocheSurfaceGravity:
    """Evaluate outward potential gradients and corner-averaged SI gravity.

    Coordinates are in the co-rotating BINARY frame, in SMA units: primary
    centre x=0, secondary centre x=1. q=M2/M1 for either component, circular
    separation=1, q in [0.5,2], synchronicity in [0.5,1.5]. Only vertices at
    own-centre radii in [1e-6,0.25] are supported. This does not certify that
    vertices lie on a closed/equipotential surface; check the ray/mesh validity
    separately. Topology is supplied and is not certified by this primitive.

    Both components use G*M_PRIMARY/a**2; no extra secondary q factor belongs
    here. ELISa's outward gradient is minus the gradient of its dimensionless
    potential; its magnitude defines effective acceleration. Face gravity is
    the mean of the three corner MAGNITUDES, not the magnitude of their mean.
    log_g uses log10(m/s**2), matching ELISa's internal SI convention.

    CPU build_surface_gravity runs BEFORE correct_mesh. To reproduce a cached
    CPU container, supply that same pre-correction snapshot. Re-evaluation at
    corrected points is a different calculation. Changed gravity alone cannot
    justify reusing old thermal/brightness/eclipse state for a physical fit.
    """

    def __init__(self, component: str, *, device: str = "gpu") -> None:
        if component not in ("primary", "secondary"):
            msg = "Component must be primary or secondary"
            raise ValueError(msg)
        self.component = component
        self._jax, jnp, self.device = _jax_device(device)
        centre = 0.0 if component == "primary" else 1.0

        def evaluate(vertices: Any, faces: Any, q: Any, spin: Any, mass: Any, axis: Any) -> dict[str, Any]:
            own = vertices - jnp.array([centre, 0.0, 0.0])
            radius = jnp.linalg.norm(own, axis=-1)
            admitted = jnp.all(jnp.isfinite(vertices), axis=-1) & (radius >= _MIN_RADIUS) & (radius <= _MAX_RADIUS)
            # Invalid vertices do not enter singular force calculations. Their
            # outputs remain explicitly NaN/unsupported rather than zero flux.
            safe = jnp.where(admitted[..., None], vertices, jnp.array([centre, 0.0, 0.1]))
            r3 = jnp.linalg.norm(safe, axis=-1) ** 3
            rhat3 = jnp.linalg.norm(safe - jnp.array([1.0, 0.0, 0.0]), axis=-1) ** 3
            x, y, z = safe[..., 0], safe[..., 1], safe[..., 2]
            qq, rotation = q[:, None], (spin**2 * (1 + q))[:, None]
            common = 1 / r3 + qq / rhat3
            dx = -x / r3 + qq * (1 - x) / rhat3
            dx = dx + rotation * x - qq if component == "primary" else dx - rotation * (1 - x) + 1
            gradient = -jnp.stack((dx, -y * (common - rotation), -z * common), axis=-1)
            scaling = const.G * mass / axis**2
            acceleration = scaling[:, None, None] * gradient
            magnitude = jnp.linalg.norm(acceleration, axis=-1)
            numerical = jnp.all(jnp.isfinite(gradient), axis=-1) & jnp.isfinite(magnitude) & (magnitude > 0)
            valid = admitted & numerical
            status = jnp.where(~admitted, 1, jnp.where(numerical, 0, 2))
            face_valid = jnp.all(valid[:, faces], axis=-1)
            face_gravity = jnp.mean(magnitude[:, faces], axis=-1)
            face_gravity = jnp.where(face_valid, face_gravity, jnp.nan)
            return {
                "outward_gradients": jnp.where(valid[..., None], gradient, jnp.nan),
                "vertex_gravity": jnp.where(valid, magnitude, jnp.nan),
                "face_gravity": face_gravity,
                "log_g": jnp.log10(face_gravity),
                "valid_vertices": valid,
                "valid_faces": face_valid,
                "proposal_valid": jnp.all(valid, axis=1),
                "status": status,
            }

        self._evaluate = self._jax.jit(evaluate)

    def __call__(
        self,
        vertices: Any,
        faces: Any,
        q: Any,
        synchronicity: Any,
        primary_mass_kg: Any,
        semi_major_axis_m: Any,
    ) -> dict[str, Any]:
        """Snapshot host inputs and return FP64 device arrays and validity masks.

        Vertices have shape (proposals, vertices, 3); faces are common integer
        (faces, 3) connectivity. Each parameter is a (proposals,) array.
        Status per vertex: 0=supported, 1=outside the spatial/finite-input
        domain, 2=numerical force/scaling failure. No radius is silently clipped.
        Nonfinite vertices are flagged; malformed shapes/parameters raise.
        Callers must check proposal_valid before consuming any physical result.
        """
        if not self._jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        v, f = np.array(vertices, dtype=np.float64, copy=True), np.array(faces, copy=True)
        params = [
            np.array(x, dtype=np.float64, copy=True) for x in (q, synchronicity, primary_mass_kg, semi_major_axis_m)
        ]
        if v.ndim != _XYZ or v.shape[-1] != _XYZ or not v.shape[0] or not v.shape[1]:
            msg = "Vertices must have nonempty (proposals, vertices, 3) shape"
            raise ValueError(msg)
        if f.ndim != _FACE_NDIM or f.shape[-1] != _XYZ or not f.shape[0] or not np.issubdtype(f.dtype, np.integer):
            msg = "Faces must be nonempty integer triangle indices"
            raise ValueError(msg)
        if np.any(f < 0) or np.any(f >= v.shape[1]):
            msg = "Face index out of bounds"
            raise ValueError(msg)
        if any(p.shape != (v.shape[0],) or not np.isfinite(p).all() for p in params):
            msg = "Parameters must be finite arrays matching the proposal count"
            raise ValueError(msg)
        qq, spin, mass, axis = params
        if np.any((qq < _Q_MIN) | (qq > _Q_MAX) | (spin < _SPIN_MIN) | (spin > _SPIN_MAX)):
            msg = "Unsupported domain: q in [0.5,2], synchronicity in [0.5,1.5]"
            raise ValueError(msg)
        if np.any(mass <= 0) or np.any(axis <= 0):
            msg = "Primary mass in kg and semi-major axis in m must be positive"
            raise ValueError(msg)
        return self._evaluate(*(self._jax.device_put(x, self.device, may_alias=False) for x in (v, f, *params)))
