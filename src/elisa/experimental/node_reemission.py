"""Re-emit the secondary's deposited heat at the physical nodes used to receive it."""

import numpy as np
from scipy.spatial import cKDTree

from elisa.binary_system.surface.faces import visibility_test
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.horizon_receiver import HorizonReceiverLightCurve, HorizonReceiverReflection


def emission_rows(surface, xlim):
    """Match every mirrored emitter to a quarter-surface receiver; the final index denotes zero heat."""
    def folded(points):
        return np.column_stack((points[:, 0], abs(points[:, 1:])))

    n = len(surface.receiver)
    rows = np.full(len(surface.emitter), n, dtype=np.int32)
    active = visibility_test(surface.emitter, xlim, "secondary")
    distance, ix = cKDTree(folded(surface.receiver)).query(folded(surface.emitter[active]))
    if np.any(distance > 3e-12) or not np.array_equal(surface.parent[ix], surface.visible_base[active]):
        raise UnsupportedReflectionError("Mirrored secondary emitter misses its physical receiver node")
    rows[active] = ix
    area = surface.visible_area.reshape(-1, 3)
    fractions = (area / area.sum(axis=1, keepdims=True)).ravel()
    if not np.allclose(fractions[active], surface.fraction[ix], rtol=3e-13, atol=0):
        raise UnsupportedReflectionError("Receiver and emitter quadrature disagree on secondary energy weights")
    return rows


class NodeReemissionReflection(HorizonReceiverReflection):
    def _prepare_pair(self, surfaces, system, limits, level):
        rows = emission_rows(surfaces[1], limits["secondary"])
        self._emission_rows = rows if self.mode == "reference" else self.jax.device_put(rows, self.device)
        result, details = super()._prepare_pair(surfaces, system, limits, level)
        details[1]["mapped_reemission_nodes"] = int(np.sum(rows < len(surfaces[1].parent)))
        return result, details

    def _sweep_update(self, order, last):
        xp = np if self.mode == "reference" else self.geometry["secondary"].jnp
        secondary_nodes = None
        step = 0

        def capture(receiver, donor, deposited):
            nonlocal secondary_nodes, step
            name = order[step % 2]
            if name == "primary":
                # Keep intrinsic gravity darkening face-centred, as in the original iterations.
                intrinsic = donor.intrinsic[donor.visible_base]
                deposited = (xp.zeros_like(intrinsic) if secondary_nodes is None else
                             xp.concatenate((secondary_nodes, xp.zeros_like(secondary_nodes[:1])))[self._emission_rows])
                donor = donor._replace(intrinsic=intrinsic, visible_base=xp.arange(len(intrinsic)))
            values, mean = self._sampled_update(receiver, donor, deposited)
            if name == "secondary":
                secondary_nodes = values
            step += 1
            last.append((receiver, values, mean))
            return mean

        return capture


class NodeReemissionLightCurve(HorizonReceiverLightCurve):
    solver_type = NodeReemissionReflection
    rule_prefix = "secondary_node_reemission"
    field_contract = "Re-emit each secondary receiver sample from its corresponding curved emitting node"
