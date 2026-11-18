"""Irradiation from curved emitting patches with receiver-specific horizon resolution."""

import numpy as np
from numba import njit

from elisa.binary_system.surface import faces
from elisa.experimental import area_reflection
from elisa.experimental.angular_reflection import donor_layout
from elisa.experimental.area_reflection import AreaReflection, _BARY
from elisa.experimental.gpu_reflection import UnsupportedReflectionError
from elisa.experimental.spatial_reflection import _Spatial, _correction


@njit(cache=True)
def _classify(points, normal, centres, outward, spread, tilt, projected, size, limit):
    """Bound incidence at both ends for every source node and its centroid ray."""
    n, g = len(points), len(centres)
    lit = np.zeros((n, (g + 7) // 8), np.uint8)
    opened = np.zeros_like(lit)
    pairs = 0
    # Coordinates are in separation units on the supported inner Roche branch.
    guard = 256 * np.finfo(np.float64).eps
    for i in range(n):
        for j in range(g):
            incident, emission, width = 0.0, 0.0, 0.0
            cone = projected[j]
            for k in range(3):
                ray = points[i, k] - centres[j, k]
                incident -= normal[i, k] * ray
                width += abs(normal[i, k]) * spread[j, k]
                emission += outward[j, k] * ray
                cone += abs(ray) * tilt[j, k]
            if incident < -width - guard or emission < -cone - guard:
                continue
            bit = 1 << (j % 8)
            if incident > width + guard and emission > cone + guard:
                lit[i, j // 8] |= bit
            else:
                pairs += size[j]
                if pairs > limit:
                    return lit, opened, -1
                opened[i, j // 8] |= bit
    return lit, opened, pairs


@njit(cache=True)
def _indices(opened, start, ordered, count):
    rows, nodes = np.empty(count, np.int32), np.empty(count, np.int32)
    cursor = 0
    for i in range(len(opened)):
        for byte in range(opened.shape[1]):
            bits = opened[i, byte]
            if not bits:
                continue
            for k in range(8):
                if bits & (1 << k):
                    j = 8 * byte + k
                    for face in range(start[j], start[j + 1]):
                        rows[cursor], nodes[cursor] = i, ordered[face]
                        cursor += 1
    if cursor != count:
        raise ArithmeticError("Emitter horizon index size changed")
    return rows, nodes


def _plan(receiver, source, donor, remaining):
    group = donor.group
    centre, outward = donor.emitter, donor.outward
    offset = source.emitter - centre[group]
    spread, tilt = np.zeros_like(centre), np.zeros_like(centre)
    np.maximum.at(spread, group, abs(offset))
    np.maximum.at(tilt, group, abs(source.outward - outward[group]))
    projected = np.zeros(len(centre))
    np.maximum.at(projected, group, abs(np.sum(source.outward * offset, axis=1)))
    size = np.bincount(group)
    lit, opened, count = _classify(
        receiver.receiver, receiver.normal, centre, outward, spread, tilt, projected, size, remaining
    )
    if count < 0:
        raise UnsupportedReflectionError("Curved emitter horizon quadrature exceeds the reflection pair cap")
    rows, nodes = _indices(opened, np.r_[0, np.cumsum(size)], np.argsort(group, kind="stable"), count)
    return _Spatial(centre, outward, np.arange(len(group)), group, offset, source.emitter, rows, nodes, lit)


class EmittingAreaReflection(AreaReflection):
    """Three positive curved-area samples per original visible emitting face.

    Lit groups use their directional and first-order spatial moments. A group
    touching either tangent plane contributes exclusively through its native
    quadrature nodes. Non-polynomial LD uses individual nodes throughout.
    """

    def _initial_width(self, level):
        return 1

    def _surface(self, binary, system, name, level, xlim):
        base = super()._surface(binary, system, name, level, xlim)
        star = getattr(system, name)
        selected = faces.visibility_test(star.face_centres, xlim, name)
        directions = star.points - np.array([float(name == "secondary"), 0.0, 0.0])
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        rays = (_BARY @ directions[star.faces[selected]]).reshape(-1, 3)
        points, normals, jac = self._sample(binary, name, rays)
        area = (jac / np.linalg.norm(rays, axis=1) ** 3).reshape(-1, 3)
        weights = area / area.sum(axis=1)[:, None]
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise UnsupportedReflectionError("Invalid emitting-face quadrature weights")
        return base._replace(
            emitter=points,
            outward=normals,
            group=np.arange(len(points)),
            visible_base=np.repeat(base.visible_base, 3),
            visible_area=(base.visible_area[:, None] * weights).ravel(),
            prefix_ld=np.repeat(base.prefix_ld, 3, axis=1),
        )

    def _prepare_pair(self, surfaces, system, limits, level):
        prepared, details = [], []
        width = 4 ** max(level - self.donor_level, 0)
        for i, name in enumerate(("primary", "secondary")):
            source = surfaces[i]
            count = len(source.emitter) // 3
            donor = source
            record = dict(original_emitting_faces=count, emitting_area_nodes=3 * count, extra_native_pairs=0, split_horizon_groups=0)
            if self.law in ("linear", "cosine"):
                star = getattr(system, name)
                visible = np.flatnonzero(faces.visibility_test(star.face_centres, limits[name], name))
                if star.base_symmetry_faces_number % width:
                    raise UnsupportedReflectionError("Invalid curved emitting-face ancestry")
                group, centre, outward, stats = donor_layout(
                    source.emitter,
                    source.outward,
                    source.visible_area,
                    np.repeat(visible // width, 3),
                    surfaces[1 - i].receiver,
                    leaf=np.repeat(np.arange(count), 3),
                )
                donor = source._replace(group=group, emitter=centre, outward=outward, native_normals=source.outward)
                record.update(stats)
            prepared.append(donor)
            details.append(record)
        remaining = area_reflection._MAX_PAIRS - sum(
            len(s.receiver) * len(d.emitter) for s, d in (prepared, prepared[::-1])
        )
        if remaining < 0:
            raise UnsupportedReflectionError("Curved emitting surfaces exceed the reflection pair cap")
        for i, donor in enumerate(prepared):
            if donor.native_normals is not None:
                plan = _plan(surfaces[1 - i], surfaces[i], donor, remaining)
                extra = len(plan.horizon_row)
                remaining -= extra
                prepared[i] = donor._replace(spatial=plan)
                details[i]["extra_native_pairs"] = extra
        return prepared, details

    def _reference_update(self, receiver, donor, deposited):
        if donor.spatial is None:
            return super()._reference_update(receiver, donor, deposited)
        return _correction(None, np, receiver, donor, deposited)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode == "reference":
            return
        native = self._update
        jax, xp = self.jax, self.geometry["primary"].jnp

        @jax.jit
        def update(receiver, donor, deposited):
            if donor.spatial is None:
                return native(receiver, donor, deposited)
            return _correction(jax, xp, receiver, donor, deposited)

        self._update = update
