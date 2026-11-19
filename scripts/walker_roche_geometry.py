"""FP64 host bisection for the physical walker benchmark's curved eclipse observer."""

import numpy as np
from numba import njit

from elisa.binary_system import model
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, _R_MAX, _R_MIN


@njit(cache=True, fastmath=False, error_model="numpy")
def _radii(mass, other, target, b, c, d, e, f):
    """Run the same 56 updates in FP64. Tile independent rays for the host vector units."""
    radii = np.empty(len(b))
    low = np.empty(256)
    high = np.empty(256)
    for start in range(0, len(b), 256):
        size = min(256, len(b) - start)
        low[:size] = _R_MIN
        high[:size] = _R_MAX
        for _ in range(56):
            for j in range(size):
                i = start + j
                mid = (low[j] + high[j]) / 2
                squared = mid * mid
                value = mass / mid + other / np.sqrt(b[i] + squared - c[i] * mid) - d[i] * mid + e[i] * squared + f
                take = value > target
                low[j] = mid if take else low[j]
                high[j] = high[j] if take else mid
        for j in range(size):
            radii[start + j] = (low[j] + high[j]) / 2
    return radii


class WalkerRocheGeometry(ConditionedRocheGeometry):
    """Preserve the reference angles, signed brackets, gradient, stationary shadow and area."""

    def sample(self, vectors):
        rays = np.asarray(vectors) / np.linalg.norm(vectors, axis=1)[:, None]
        local = rays.copy()
        primary = self.component == "primary"
        if not primary:
            local[:, 0] *= -1
        q, spin, target = self.binary.mass_ratio, self.star.synchronicity, self.star.surface_potential
        coefficients = getattr(model, f"pre_calculate_for_potential_value_{self.component}")(
            spin,
            q,
            1.0,
            np.arctan2(local[:, 1], local[:, 0]),
            np.arccos(np.clip(local[:, 2], -1, 1)),
            return_as_tuple=True,
        )
        potential = getattr(model, f"potential_value_{self.component}")
        if np.any(potential(_R_MIN, q, *coefficients) <= target) or np.any(
            potential(_R_MAX, q, *coefficients) >= target
        ):
            raise UnsupportedCurvedGeometryError("Curved rays are outside the inner[1e-6,.25] bracket")
        # In the secondary frame f is the constant (1-q)/2 from the ELISa potential.
        offset = 0.0 if primary else (1 - q) / 2
        if not primary and np.any(coefficients[4] != offset):
            raise UnsupportedCurvedGeometryError("Unexpected secondary potential constant")
        radius = _radii(self.mass, q if primary else 1.0, target, *coefficients[:4], offset)
        points = self.centre + radius[:, None] * rays
        g = calculate_potential_gradient(1.0, self.component, points, spin, q)
        norm, radial = np.linalg.norm(g, axis=1), np.einsum("ij,ij->i", g, rays)
        if not np.isfinite(g).all() or np.any(radial <= 0):
            raise UnsupportedCurvedGeometryError("Curved surface is not on the supported star-shaped branch")
        return points, g / norm[:, None], radius * radius * norm / radial
