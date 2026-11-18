"""Angular moments for linear limb darkening with explicitly resolved donor horizons."""

from __future__ import annotations

import numpy as np
from numba import njit

from elisa.binary_system.surface import faces
from elisa.experimental.area_reflection import AreaReflection, _BLOCK, _group_sum
from elisa.experimental.gpu_reflection import UnsupportedReflectionError


def _centres(group, weights, positions):
    total = np.bincount(group, weights=weights)
    return np.column_stack([np.bincount(group, weights=weights * v) / total for v in positions.T])


def donor_layout(positions, normals, area, ancestry, receivers):
    """Split groups unless every child has the same strict sign throughout the receiver box.

    Test both the native source position and the group centroid. The extrema of
    the unnormalized direction cosine are linear support functions of the box.
    Singleton emitters always retain their own position, normal and horizon.
    """
    _, group = np.unique(ancestry, return_inverse=True)
    count = np.bincount(group)
    centroid = _centres(group, area, positions)
    lo, hi = receivers.min(axis=0), receivers.max(axis=0)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    extent = abs(normals) @ half
    at_face = np.sum(normals * (mid - positions), axis=1)
    at_group = np.sum(normals * (mid - centroid[group]), axis=1)
    # All coordinates use binary separation units. Account for products, sums
    # and subtraction of nearby coordinates before certifying either strict sign.
    scale = 1 + np.sum(abs(normals) * (abs(mid) + half + np.maximum(abs(positions), abs(centroid[group]))), axis=1)
    guard = 64 * np.finfo(float).eps * scale
    positive = np.minimum(at_face, at_group) - extent > guard
    negative = np.maximum(at_face, at_group) + extent < -guard
    all_positive = np.bincount(group, weights=positive) == count
    all_negative = np.bincount(group, weights=negative) == count
    split = ~(all_positive | all_negative) & (count > 1)
    labels = np.where(split[group], len(count) + np.arange(len(group)), group)
    _, inverse, size = np.unique(labels, return_inverse=True, return_counts=True)
    centre = _centres(inverse, area, positions)
    outward = _centres(inverse, area, normals)
    single = size[inverse] == 1
    centre[inverse[single]], outward[inverse[single]] = positions[single], normals[single]
    if not np.isfinite(centre).all() or not np.isfinite(outward).all():
        raise UnsupportedReflectionError("Nonfinite directional group geometry")
    return (
        inverse,
        centre,
        outward,
        {
            "initial_groups": len(count),
            "split_horizon_groups": int(split.sum()),
            "faces_in_split_groups": int(count[split].sum()),
            "positive_merged_groups": int(np.sum(all_positive & (count > 1))),
            "negative_merged_groups": int(np.sum(all_negative & (count > 1))),
        },
    )


def _moments(xp, donor, heat):
    """Luminosity-weighted vector and six independent quadratic coefficients."""
    light = (donor.intrinsic + heat)[donor.visible_base] * donor.visible_area
    linear = light * donor.prefix_ld[0]
    x, y, z = donor.native_normals.T
    weights = [(light - linear) * n for n in (x, y, z)]
    weights.extend(linear * p for p in (x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z))
    return xp.stack([_group_sum(xp, donor.group, len(donor.emitter), w) for w in weights], axis=1)


@njit(cache=True)
def _cpu_incident(receivers, normals, emitters, outward, moments):
    # Scalar CPU quadrature. The group normal supplies only a visibility sign.
    values = np.zeros(len(receivers))
    for i in range(len(receivers)):
        for j in range(len(emitters)):
            x = receivers[i, 0] - emitters[j, 0]
            y = receivers[i, 1] - emitters[j, 1]
            z = receivers[i, 2] - emitters[j, 2]
            r2 = x * x + y * y + z * z
            inv = 1.0 / np.sqrt(r2)
            x, y, z = x * inv, y * inv, z * inv
            mu = -(x * normals[i, 0] + y * normals[i, 1] + z * normals[i, 2])
            front = x * outward[j, 0] + y * outward[j, 1] + z * outward[j, 2]
            if mu > 0 and front > 0:
                m = moments[j]
                flux = (
                    x * m[0]
                    + y * m[1]
                    + z * m[2]
                    + x * x * m[3]
                    + y * y * m[4]
                    + z * z * m[5]
                    + x * y * m[6]
                    + x * z * m[7]
                    + y * z * m[8]
                )
                values[i] += mu * flux / r2
    return values


class AngularReflection(AreaReflection):
    """Linear/cosine groups preserve native directional moments on a shared ray.

    For non-polynomial laws use individual native sources under the same pair
    limit. A group's centroid is its area-weighted Cartesian position.
    """

    def _initial_width(self, level):
        return 1

    def _prepare_pair(self, surfaces, system, limits, level):
        prepared, details = [], []
        width = 4 ** max(level - self.donor_level, 0)
        for i, name in enumerate(("primary", "secondary")):
            donor = surfaces[i]
            record = dict(
                initial_groups=len(donor.emitter),
                split_horizon_groups=0,
                faces_in_split_groups=0,
                positive_merged_groups=0,
                negative_merged_groups=0,
            )
            if width > 1 and self.law in ("linear", "cosine"):
                star = getattr(system, name)
                if star.base_symmetry_faces_number % width:
                    raise UnsupportedReflectionError("Invalid donor ancestry")
                visible = np.flatnonzero(faces.visibility_test(star.face_centres, limits[name], name))
                group, pos, normal, record = donor_layout(
                    donor.emitter, donor.outward, donor.visible_area, visible // width, surfaces[1 - i].receiver
                )
                if len(pos) < len(donor.emitter):
                    donor = donor._replace(emitter=pos, outward=normal, group=group, native_normals=donor.outward)
            prepared.append(donor)
            details.append(record)
        return prepared, details

    def _reference_update(self, receiver, donor, deposited):
        if donor.native_normals is None:
            return super()._reference_update(receiver, donor, deposited)
        values = _cpu_incident(
            receiver.receiver, receiver.normal, donor.emitter, donor.outward, _moments(np, donor, deposited)
        )
        return receiver.absorption * np.bincount(
            receiver.parent, weights=values * receiver.fraction, minlength=len(receiver.intrinsic)
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode == "reference":
            return
        jax, xp = self.jax, self.geometry["primary"].jnp
        native_update = self._update

        @jax.jit
        def update(receiver, donor, deposited):
            if donor.native_normals is None:
                return native_update(receiver, donor, deposited)
            moments = _moments(xp, donor, deposited)
            size = len(receiver.receiver)
            pad = ((0, (-size) % _BLOCK), (0, 0))
            points, normal = (xp.pad(a, pad, mode="edge") for a in (receiver.receiver, receiver.normal))

            def tile(i, output):
                start = i * _BLOCK
                p, n = (jax.lax.dynamic_slice_in_dim(a, start, _BLOCK) for a in (points, normal))
                delta = p[:, None, :] - donor.emitter
                r2 = xp.sum(delta * delta, axis=-1)
                u = delta / xp.sqrt(r2)[..., None]
                x, y, z = xp.moveaxis(u, -1, 0)
                mu = xp.maximum(-xp.sum(n[:, None, :] * u, axis=-1), 0)
                front = xp.sum(donor.outward * u, axis=-1) > 0
                m = moments.T
                flux = (
                    x * m[0]
                    + y * m[1]
                    + z * m[2]
                    + x * x * m[3]
                    + y * y * m[4]
                    + z * z * m[5]
                    + x * y * m[6]
                    + x * z * m[7]
                    + y * z * m[8]
                )
                values = xp.sum(xp.where(front, mu * flux / r2, 0), axis=1)
                return jax.lax.dynamic_update_slice_in_dim(output, values, start, axis=0)

            incident = jax.lax.fori_loop(0, len(points) // _BLOCK, tile, xp.zeros(len(points)))[:size]
            return receiver.absorption * _group_sum(
                xp, receiver.parent, len(receiver.intrinsic), incident * receiver.fraction
            )

        self._update = update
