"""Reconstruct the final irradiation sweep from the actual three samples on each receiving face."""

from collections import deque
from copy import copy

import numpy as np

from elisa.experimental.area_reflection import _BARY, _group_sum
from elisa.experimental.gpu_curved_edges import SymmetricInteriorRule
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.quadratic_reflection import QuadraticReflection
from elisa.experimental.quadratic_source import QuadraticSourceLightCurve
from elisa.experimental.shadow_refinement import deeper_shadow_packet


def receiver_samples(star, name, receiver, values, means):
    """Map surviving physical rays back to their original slots. Excluded rays deposit zero."""
    origin = np.array([float(name == "secondary"), 0.0, 0.0])
    dirs = star.points - origin
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    n = star.base_symmetry_faces_number
    parent = np.asarray(receiver.parent)
    tri = dirs[star.faces[:n]]
    points = np.asarray(receiver.receiver) - origin
    coords = np.linalg.solve(tri[parent].transpose(0, 2, 1), points[..., None])[..., 0]
    coords /= coords.sum(axis=1)[:, None]
    slot = coords.argmax(axis=1)
    if not np.allclose(coords, _BARY[slot], atol=3e-12, rtol=0) or len(np.unique(3 * parent + slot)) != len(slot):
        raise UnsupportedReflectionError("Physical receiver ray does not identify one unique cubature node")

    sampled = np.zeros((n, 3))
    sampled[parent, slot] = np.asarray(values) / star.t_eff**4
    mean = np.asarray(means) / star.t_eff**4
    weighted = np.bincount(parent, weights=sampled[parent, slot] * np.asarray(receiver.fraction), minlength=n)
    if not np.isfinite(sampled).all() or np.any(sampled < 0) or not np.allclose(weighted, mean, rtol=3e-13, atol=0):
        raise UnsupportedReflectionError("Receiver heat samples fail nonnegativity or the original face mean")
    counts = np.bincount(parent, minlength=n)
    return dict(receiver_samples_per_face=np.bincount(counts, minlength=4).tolist(), receiver_deposits=(sampled, mean))


class SampledReceiverReflection(QuadraticReflection):
    """Carry one heat value per native receiver ray through the existing bounded solver."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        reference = self.mode == "reference"
        xp = np if reference else self.geometry["primary"].jnp
        update = self._reference_update if reference else self._update

        def sampled_update(receiver, donor, deposited):
            index = receiver.parent
            per_ray = receiver._replace(
                parent=xp.arange(len(index)),
                intrinsic=receiver.intrinsic[index],
                absorption=receiver.absorption[index],
                fraction=xp.ones_like(receiver.fraction),
            )
            values = update(per_ray, donor, deposited)
            mean = _group_sum(xp, index, len(receiver.intrinsic), values * receiver.fraction)
            return values, mean

        self._sampled_update = sampled_update if reference else self.jax.jit(sampled_update)

    def evaluate(self, binary, system, level, iterations):
        if self.receiver_points != 3:
            raise UnsupportedReflectionError("Within-face heating requires three receiver rays")
        if iterations == 0:
            return super().evaluate(binary, system, level, iterations)

        # The existing cooler-first driver expects a face mean after each half-iteration.
        # Keep its inputs and order. Retain just the two latest unreduced fields on this call.
        worker = copy(self)
        last = deque(maxlen=2)

        def capture(receiver, donor, deposited):
            values, mean = self._sampled_update(receiver, donor, deposited)
            last.append((receiver, values, mean))
            return mean

        if self.mode == "reference":
            worker._reference_update = capture
        else:
            worker._update = capture
        temperatures, stats = QuadraticReflection.evaluate(worker, binary, system, level, iterations)
        order = sorted(("primary", "secondary"), key=lambda k: getattr(system, k).t_eff)
        for name, data in zip(order, last, strict=True):
            stats[name].update(receiver_samples(getattr(system, name), name, *data))
        return temperatures, stats


class ReceiverHeatingLightCurve(QuadraticSourceLightCurve):
    """Use sampled deposits and optionally refine the angular parents crossing the shadow."""

    solver_type = SampledReceiverReflection
    rule_prefix = "sampled_receiver_heat"
    field_contract = "Conservative six-node thermal emission using the final three deposited heat samples"

    def __init__(self, passbands, rule, *, selective_shadow=False, **kwargs):
        if selective_shadow:
            self.rule_prefix = "shadow_sampled_receiver_heat"
        if isinstance(rule, SymmetricInteriorRule):
            self.rule_prefix = "symmetric_" + self.rule_prefix
        super().__init__(passbands, rule, **kwargs)
        if selective_shadow:
            self._packet = deeper_shadow_packet

    def _source_data(self, backend, model, star, diagnostics):
        stats = dict(diagnostics.get("reflection_quadrature", {}))
        deposits = stats.pop("receiver_deposits", None)
        if deposits is None and stats.get("pairs_per_sweep_both_stars", 0):
            raise UnsupportedReflectionError("Missing final receiver heat samples")
        return backend.six_nodes(model, star, diagnostics["intrinsic_temperatures"], deposits=deposits), stats
