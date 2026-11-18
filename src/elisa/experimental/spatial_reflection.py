"""First spatial derivative of directional donor emission and native receiver horizons."""

from typing import NamedTuple

import numpy as np
from numba import njit

from elisa.experimental import area_reflection
from elisa.experimental.angular_reflection import AngularReflection, _moments
from elisa.experimental.area_reflection import _BLOCK, _group_sum
from elisa.experimental.gpu_reflection import UnsupportedReflectionError


class _Spatial(NamedTuple):
    centres: np.ndarray
    outward: np.ndarray
    members: np.ndarray
    group: np.ndarray
    offset: np.ndarray
    native_positions: np.ndarray
    horizon_row: np.ndarray
    horizon_face: np.ndarray
    lit: np.ndarray | None = None


@njit(cache=True)
def _horizons(points, normals, centres, extent, starts, ordered, positions, rows, faces, fill, limit):
    """Count first, then index native children only where a support box touches the tangent plane."""
    used = 0
    opened = 0
    guard = 128 * np.finfo(np.float64).eps
    for i in range(len(points)):
        for j in range(len(centres)):
            side = 0.0
            width = 0.0
            for k in range(3):
                side += normals[i, k] * (centres[j, k] - points[i, k])
                width += abs(normals[i, k]) * extent[j, k]
            if abs(side) > width + guard:
                continue
            # A box can straddle the plane while all its actual sources are
            # strictly on one side. Retain only crossings or near-tangencies.
            split = False
            for t in range(starts[j], starts[j + 1]):
                child = ordered[t]
                native = 0.0
                for k in range(3):
                    native += normals[i, k] * (positions[child, k] - points[i, k])
                if abs(native) <= guard or abs(side) <= guard or (native > 0) != (side > 0):
                    split = True
                    break
            if split:
                opened += 1
                count = starts[j + 1] - starts[j]
                if used + count > limit:
                    return -1, opened
                if fill:
                    for t in range(count):
                        rows[used + t] = i
                        faces[used + t] = ordered[starts[j] + t]
                used += count
    return used, opened


def _layout(receiver, native, donor, remaining):
    counts = np.bincount(donor.group)
    merged = np.flatnonzero(counts > 1)
    members = np.flatnonzero(counts[donor.group] > 1)
    group = np.searchsorted(merged, donor.group[members])
    centres = donor.emitter[merged]
    pos = native.emitter[members]
    offset = pos - centres[group]
    extent = np.zeros_like(centres)
    np.maximum.at(extent, group, abs(offset))
    order = np.argsort(group, kind="stable")
    args = (receiver.receiver, receiver.normal, centres, extent, np.r_[0, np.cumsum(counts[merged])], order, pos)
    stub = np.empty(0, dtype=np.int32)
    count, opened = _horizons(*args, stub, stub, False, remaining)
    if count < 0:
        raise UnsupportedReflectionError("Resolved receiver horizons exceed the reflection pair cap")
    rows, children = np.empty(count, np.int32), np.empty(count, np.int32)
    if count and _horizons(*args, rows, children, True, count) != (count, opened):
        raise ArithmeticError("Receiver horizon layout changed between counting and filling")
    return _Spatial(centres, donor.outward[merged], members, group, offset, pos, rows, children), opened


def _spatial_moments(xp, plan, normal, coeff, light):
    """Pack three rows of nine displacement/normal moments, then four trace terms.

    For native offset d, unit donor normal n, LD a and current luminosity L,
    the rows represent sums of L(1-a)d n and La d nn. The traces multiply
    (n·d) by L(1-a) or 2La n. Off-diagonal quadratic entries include both orders.
    """
    x, y, z = normal.T
    ld = light * coeff
    plain = light - ld
    angle = [plain * k for k in (x, y, z)] + [ld * k for k in (x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z)]
    dot = xp.sum(normal * plan.offset, axis=1)
    weights = [d * k for d in plan.offset.T for k in angle]
    weights.extend([dot * plain, *(2 * dot * ld * n for n in (x, y, z))])
    return xp.stack([_group_sum(xp, plan.group, len(plan.centres), w) for w in weights], axis=1)


@njit(cache=True)
def _cpu_gradient(points, normals, centres, outward, moments, angular=None, lit=None):
    total = np.zeros(len(points))
    for i, h in enumerate(normals):
        for j, centre in enumerate(centres):
            if lit is not None and not (lit[i, j // 8] & (1 << (j % 8))):
                continue
            dx = points[i, 0] - centre[0]
            dy = points[i, 1] - centre[1]
            dz = points[i, 2] - centre[2]
            inv = 1 / np.sqrt(dx * dx + dy * dy + dz * dz)
            x, y, z = dx * inv, dy * inv, dz * inv
            u = (x, y, z)
            mu = -(x * h[0] + y * h[1] + z * h[2])
            if mu <= 0 or x * outward[j, 0] + y * outward[j, 1] + z * outward[j, 2] <= 0:
                continue
            m = moments[j]
            value = -mu * (m[27] + x * m[28] + y * m[29] + z * m[30])
            for k in range(3):
                p = k * 9
                linear = x * m[p] + y * m[p + 1] + z * m[p + 2]
                quadratic = (
                    x * x * m[p + 3]
                    + y * y * m[p + 4]
                    + z * z * m[p + 5]
                    + x * y * m[p + 6]
                    + x * z * m[p + 7]
                    + y * z * m[p + 8]
                )
                value += (4 * mu * u[k] + h[k]) * linear + (5 * mu * u[k] + h[k]) * quadratic
            term = value * inv**3
            if angular is not None:
                a = angular[j]
                flux = (
                    x * a[0]
                    + y * a[1]
                    + z * a[2]
                    + x * x * a[3]
                    + y * y * a[4]
                    + z * z * a[5]
                    + x * y * a[6]
                    + x * z * a[7]
                    + y * z * a[8]
                )
                term += mu * flux * inv**2
            total[i] += term
    return total


def _device_gradient(jax, xp, receiver, plan, moments, angular=None):
    size = len(receiver.receiver)
    pad = ((0, (-size) % _BLOCK), (0, 0))
    p, n = (xp.pad(a, pad, mode="edge") for a in (receiver.receiver, receiver.normal))

    packed = None if plan.lit is None else xp.pad(plan.lit, pad, mode="edge")

    def tile(i, result):
        start = i * _BLOCK
        points, h = (jax.lax.dynamic_slice_in_dim(a, start, _BLOCK)[:, None, :] for a in (p, n))
        ray = points - plan.centres
        inv = 1 / xp.sqrt(xp.sum(ray * ray, axis=-1))
        u = ray * inv[..., None]
        x, y, z = xp.moveaxis(u, -1, 0)
        mu = -xp.sum(h * u, axis=-1)
        m = moments.T
        value = -mu * (m[27] + x * m[28] + y * m[29] + z * m[30])
        for k in range(3):
            v = m[9 * k : 9 * k + 9]
            linear = x * v[0] + y * v[1] + z * v[2]
            quad = x * x * v[3] + y * y * v[4] + z * z * v[5] + x * y * v[6] + x * z * v[7] + y * z * v[8]
            value += (4 * mu * u[..., k] + h[..., k]) * linear + (5 * mu * u[..., k] + h[..., k]) * quad
        front = (mu > 0) & (xp.sum(plan.outward * u, axis=-1) > 0)
        term = value * inv**3
        if angular is not None:
            a = angular.T
            flux = (
                x * a[0]
                + y * a[1]
                + z * a[2]
                + x * x * a[3]
                + y * y * a[4]
                + z * z * a[5]
                + x * y * a[6]
                + x * z * a[7]
                + y * z * a[8]
            )
            term += mu * flux * inv**2
            bits = jax.lax.dynamic_slice_in_dim(packed, start, _BLOCK)
            idx = xp.arange(len(plan.centres))
            front = (bits[:, idx // 8] & (1 << (idx % 8))) != 0
        values = xp.sum(xp.where(front, term, 0), axis=1)
        return jax.lax.dynamic_update_slice_in_dim(result, values, start, axis=0)

    return jax.lax.fori_loop(0, len(p) // _BLOCK, tile, xp.zeros(len(p)))[:size]


def _horizon_kernel(xp, receiver, plan, normals, coeff, row, f):
    """Replace the shared-position Taylor term by each native, clipped interaction."""
    h, n, a, d = receiver.normal[row], normals[f], coeff[f], plan.offset[f]
    points = receiver.receiver[row]

    def kernel(source):
        ray = points - source
        inv = 1 / xp.sqrt(xp.sum(ray * ray, axis=1))
        u = ray * inv[:, None]
        mu = -xp.sum(h * u, axis=1)
        b = xp.sum(n * u, axis=1)
        k = b * (1 - a + a * b)
        return xp.where((mu > 0) & (b > 0), mu * k * inv**2, 0), inv, u, mu, b, k

    native = kernel(plan.native_positions[f])[0]
    if plan.lit is not None:
        # The packed mask already excluded this entire group from the moment sum.
        return native
    common, inv, u, mu, b, k = kernel(plan.centres[plan.group[f]])
    prime = 1 - a + 2 * a * b
    derivative = (
        mu * (3 * k + b * prime) * xp.sum(u * d, axis=1)
        + k * xp.sum(d * h, axis=1)
        - mu * prime * xp.sum(n * d, axis=1)
    ) * inv**3
    return native - (common + xp.where((mu > 0) & (b > 0), derivative, 0))


# Bound temporary ray/normal arrays even when many native horizon pairs fit the pair cap.
_HORIZON_TILE = 65536


def _resolved_horizons(jax, xp, receiver, plan, normal, coeff, light):
    count = len(plan.horizon_row)
    rows, faces = plan.horizon_row, plan.horizon_face
    size = len(receiver.receiver)

    def weights(r, f):
        return _horizon_kernel(xp, receiver, plan, normal, coeff, r, f) * light[f]

    if count <= _HORIZON_TILE:
        return _group_sum(xp, rows, size, weights(rows, faces))
    result = xp.zeros(size)
    if xp is np:
        for start in range(0, count, _HORIZON_TILE):
            r, f = rows[start : start + _HORIZON_TILE], faces[start : start + _HORIZON_TILE]
            np.add.at(result, r, weights(r, f))
        return result

    def tile(i, out):
        r, f = (jax.lax.dynamic_slice_in_dim(a, i * _HORIZON_TILE, _HORIZON_TILE) for a in (rows, faces))
        return out.at[r].add(weights(r, f))

    chunks, tail = divmod(count, _HORIZON_TILE)
    result = jax.lax.fori_loop(0, chunks, tile, result)
    if tail:
        result = result.at[rows[-tail:]].add(weights(rows[-tail:], faces[-tail:]))
    return result


def _correction(jax, xp, receiver, donor, deposited):
    plan = donor.spatial
    m = plan.members
    normal, coeff = donor.native_normals[m], donor.prefix_ld[0, m]
    light = (donor.intrinsic + deposited)[donor.visible_base[m]] * donor.visible_area[m]
    moments = _spatial_moments(xp, plan, normal, coeff, light)
    angular = None if plan.lit is None else _moments(xp, donor, deposited)
    if xp is np:
        values = _cpu_gradient(
            receiver.receiver, receiver.normal, plan.centres, plan.outward, moments, angular, plan.lit
        )
    else:
        values = _device_gradient(jax, xp, receiver, plan, moments, angular)
    if len(plan.horizon_row):
        values += _resolved_horizons(jax, xp, receiver, plan, normal, coeff, light)
    return receiver.absorption * _group_sum(xp, receiver.parent, len(receiver.intrinsic), values * receiver.fraction)


class SpatialReflection(AngularReflection):
    """Preserve directional moments and resolve individual positions through first order."""

    def _prepare_pair(self, surfaces, system, limits, level):
        prepared, details = super()._prepare_pair(surfaces, system, limits, level)
        base = sum(len(s.receiver) * len(d.emitter) for s, d in (prepared, prepared[::-1]))
        remaining = area_reflection._MAX_PAIRS - base
        if remaining < 0:
            raise UnsupportedReflectionError("Directional groups exceed the reflection pair cap")
        for i, donor in enumerate(prepared):
            if donor.native_normals is None:
                continue
            plan, opened = _layout(surfaces[1 - i], surfaces[i], donor, remaining)
            extra = len(plan.horizon_row)
            details[i].update(receiver_horizon_groups=opened, extra_native_pairs=extra)
            remaining -= extra
            prepared[i] = donor._replace(spatial=plan)
        return prepared, details

    def _reference_update(self, receiver, donor, deposited):
        base = super()._reference_update(receiver, donor, deposited)
        if donor.spatial is None:
            return base
        return base + _correction(None, np, receiver, donor, deposited)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode == "reference":
            return
        base = self._update
        jax, xp = self.jax, self.geometry["primary"].jnp

        @jax.jit
        def update(receiver, donor, deposited):
            value = base(receiver, donor, deposited)
            if donor.spatial is None:
                return value
            return value + _correction(jax, xp, receiver, donor, deposited)

        self._update = update
