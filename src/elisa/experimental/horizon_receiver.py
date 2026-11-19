"""Higher-order primary receiver means near the finite companion's illumination boundary."""

import numpy as np

from elisa.binary_system.surface.faces import visibility_test
from elisa.experimental.area_reflection import _BARY
from elisa.experimental.gpu_batched_geometry import MAX_NODES
from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.primary_cubic import CurvedHeatingLightCurve
from elisa.experimental.receiver_heating import SampledReceiverReflection, receiver_samples


_SIX = SymmetricInteriorRule(3, 1)
_SIX_BARY = _SIX.interior_nodes(np.eye(3)[None])[0]


class HorizonReceiverReflection(SampledReceiverReflection):
    """Retain three heating samples for reconstruction and integrate the primary penumbra with six."""

    def _surface(self, binary, system, name, level, xlim):
        original = super()._surface(binary, system, name, level, xlim)
        if name != "primary":
            return original
        star = system.primary
        n = star.base_symmetry_faces_number
        vertices = star.points[star.faces[:n]]
        # Horizon selection uses the projected companion diameter in separation units.
        # It only chooses a quadrature rule. Every ray still uses the original visibility cut.
        width = 2 * star.polar_radius * system.secondary.polar_radius
        x = vertices[:, :, 0]
        use_six = (x.max(axis=1) >= xlim) & (x.min(axis=1) <= xlim + width)
        which = np.flatnonzero(use_six)
        if not len(which):
            return original
        if 3 * n + 6 * len(which) > MAX_NODES:
            raise UnsupportedReflectionError("Primary receiver refinement exceeds the node budget")
        directions = vertices[which] / np.linalg.norm(vertices[which], axis=-1, keepdims=True)
        rays, solid = _SIX.interior_nodes(directions)
        points, normals, jac = self._sample(binary, name, rays)
        area = (solid * jac).reshape(-1, 6)
        if np.any(area <= 0) or not np.isfinite(area).all():
            raise UnsupportedReflectionError("Invalid six-point receiver area")
        keep = visibility_test(points, xlim, name)
        weights = (area / area.sum(axis=1, keepdims=True)).ravel()
        # Nodes excluded by the original visibility test deposit zero; their area remains in the denominator.
        return original._replace(
            receiver=np.concatenate((original.receiver, points[keep])),
            normal=np.concatenate((original.normal, normals[keep])),
            parent=np.r_[original.parent, np.repeat(which, 6)[keep]],
            fraction=np.r_[np.where(use_six[original.parent], 0.0, original.fraction), weights[keep]],
        )

    @staticmethod
    def _receiver_record(star, name, receiver, values, means):
        if name != "primary":
            return receiver_samples(star, name, receiver, values, means)
        parent, weights, values = map(np.asarray, (receiver.parent, receiver.fraction, values))
        n = star.base_symmetry_faces_number
        directions = star.points / np.linalg.norm(star.points, axis=1)[:, None]
        tri = directions[star.faces[:n]][parent]
        bary = np.linalg.solve(tri.transpose(0, 2, 1), np.asarray(receiver.receiver)[..., None])[..., 0]
        bary /= bary.sum(axis=1, keepdims=True)
        rules = np.vstack((_BARY, _SIX_BARY))
        error = np.max(abs(bary[:, None, :] - rules), axis=2)
        slots = error.argmin(axis=1)
        if (np.max(np.min(error, axis=1)) > 3e-12 or len(np.unique(9 * parent + slots)) != len(parent)
                or np.any(values < 0) or not np.isfinite(values).all()):
            raise UnsupportedReflectionError("Invalid refined receiver slot or deposited energy")
        native = slots < 3
        samples = np.zeros((n, 3))
        samples[parent[native], slots[native]] = values[native] / star.t_eff**4
        mean = np.asarray(means) / star.t_eff**4
        rebuilt = np.bincount(parent, weights=weights * (values / star.t_eff**4), minlength=n)
        if not np.allclose(rebuilt, mean, rtol=3e-13, atol=0):
            raise UnsupportedReflectionError("Refined receiver mean failed independent reduction")
        return dict(
            receiver_deposits=(samples, mean),
            receiver_samples_per_face=np.bincount(np.bincount(parent[native], minlength=n), minlength=4).tolist(),
            six_point_receiver_rays=int(np.sum(~native)),
            faces_with_six_point_rays=len(np.unique(parent[~native])),
        )


class HorizonReceiverLightCurve(CurvedHeatingLightCurve):
    solver_type = HorizonReceiverReflection
    rule_prefix = "horizon_mean_primary_cubic"
    field_contract = "Three primary reconstruction samples with a six-point curved-area mean near the illumination boundary"
