"""Second-order emitting-position moments for groups strictly inside both horizons."""

from collections import defaultdict
from typing import NamedTuple

import numpy as np
from numba import njit

from elisa.experimental.area_reflection import _BLOCK, _group_sum
from elisa.experimental.emitting_area import EmittingAreaReflection
from elisa.experimental.paired_reflection import PairedEmittingAreaLightCurve


def _multiply(left, right):
    product = defaultdict(float)
    for (a, b, c), value in left.items():
        for (i, j, k), other in right.items():
            product[a + i, b + j, c + k] += value * other
    return product


def _coefficients(terms, degree, size):
    poly = defaultdict(float)
    for factor, polynomial in terms:
        for powers, coefficient in polynomial.items():
            poly[powers] += factor * coefficient
    return [
        np.broadcast_to(poly[i, j, k], size)
        for i in range(degree, -1, -1)
        for j in range(degree - i, -1, -1)
        for k in range(degree - i - j, -1, -1)
    ]


def position_moments(offset, normal, a):
    """Expand the quadratic correction as mu*P4(u) + h·Q3(u), divided by r^4.

    u points from group centre to receiver, mu = -h·u and h is the receiver normal.
    t = u·offset and b = u·normal. The two linear-LD terms expand the inverse
    distance factors with powers 2 and 5/2, respectively.
    """
    powers = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    one = {(0, 0, 0): 1.0}
    t, b = dict(zip(powers, offset.T, strict=True)), dict(zip(powers, normal.T, strict=True))
    tb, bb = _multiply(t, b), _multiply(b, b)
    tt_b, t_bb, tt_bb = _multiply(t, tb), _multiply(t, bb), _multiply(tb, tb)
    nd = np.sum(normal * offset, axis=1)
    dd = np.sum(offset * offset, axis=1)
    plain = 1 - a
    p = [
        (-4 * plain * nd, t),
        (-2 * plain * dd, b),
        (12 * plain, tt_b),
        (a * nd**2, one),
        (-10 * a * nd, tb),
        (-2.5 * a * dd, bb),
        (17.5 * a, tt_bb),
    ]
    columns = _coefficients(p, 4, len(a))
    q = _coefficients([(-plain * nd, one), (4 * plain, tb), (-2 * a * nd, b), (5 * a, t_bb)], 3, len(a))
    for coordinate in offset.T:
        columns.extend(coordinate * v for v in q)
    return np.column_stack(columns)


class _Quadratic(NamedTuple):
    base: object
    coefficients: np.ndarray


@njit(cache=True)
def _sum_moments(group, coefficients, light, size):
    out = np.zeros((size, coefficients.shape[1]))
    for i, g in enumerate(group):
        for k in range(coefficients.shape[1]):
            out[g, k] += coefficients[i, k] * light[i]
    return out


# Fixed descending monomial order is shared by the host coefficients and both evaluators.
def _horner(m, x, y, z, degree, index):
    result = 0.0
    for i in range(degree, -1, -1):
        row = 0.0
        for j in range(degree - i, -1, -1):
            col = m[index]
            index += 1
            for _ in range(degree - i - j):
                col = col * z + m[index]
                index += 1
            row = row * y + col
        result = result * x + row
    return result


_cpu_poly = njit(cache=True)(_horner)


@njit(cache=True)
def _cpu_values(points, normals, centres, lit, moments):
    values = np.zeros(len(points))
    for i, h in enumerate(normals):
        for j in range(len(centres)):
            if not (lit[i, j // 8] & (1 << (j % 8))):
                continue
            dx, dy, dz = (points[i, 0] - centres[j, 0], points[i, 1] - centres[j, 1], points[i, 2] - centres[j, 2])
            inv = 1 / np.sqrt(dx * dx + dy * dy + dz * dz)
            x, y, z = dx * inv, dy * inv, dz * inv
            m = moments[j]
            total = -(h[0] * x + h[1] * y + h[2] * z) * _cpu_poly(m, x, y, z, 4, 0)
            for k in range(3):
                total += h[k] * _cpu_poly(m, x, y, z, 3, 35 + 20 * k)
            values[i] += total * inv**4
    return values


def _device_values(jax, xp, receiver, base, moments):
    size = len(receiver.receiver)
    pad = ((0, (-size) % _BLOCK), (0, 0))
    points, normals, lit = (xp.pad(v, pad, mode="edge") for v in (receiver.receiver, receiver.normal, base.lit))
    idx = xp.arange(len(base.centres))

    def tile(i, out):
        start = i * _BLOCK
        p, h, mask = (jax.lax.dynamic_slice_in_dim(v, start, _BLOCK) for v in (points, normals, lit))
        ray = p[:, None, :] - base.centres
        inv = 1 / xp.sqrt(xp.sum(ray**2, axis=2))
        u = ray * inv[..., None]
        x, y, z = (u[..., k] for k in range(3))
        value = -xp.sum(h[:, None, :] * u, axis=2) * _horner(moments.T, x, y, z, 4, 0)
        for k in range(3):
            value += h[:, k, None] * _horner(moments.T, x, y, z, 3, 35 + 20 * k)
        keep = (mask[:, idx // 8] & (1 << (idx % 8))) != 0
        return jax.lax.dynamic_update_slice_in_dim(out, xp.sum(xp.where(keep, value * inv**4, 0), axis=1), start, 0)

    return jax.lax.fori_loop(0, len(points) // _BLOCK, tile, xp.zeros(len(points)))[:size]


def _second(jax, xp, receiver, donor, heat, plan):
    base, coeff = plan
    light = (donor.intrinsic + heat)[donor.visible_base] * donor.visible_area
    if xp is np:
        moments = _sum_moments(donor.group, coeff, light, len(base.centres))
        values = _cpu_values(receiver.receiver, receiver.normal, base.centres, base.lit, moments)
    else:
        moments = xp.zeros((len(base.centres), coeff.shape[1])).at[donor.group].add(light[:, None] * coeff)
        values = _device_values(jax, xp, receiver, base, moments)
    return receiver.absorption * _group_sum(xp, receiver.parent, len(receiver.intrinsic), values * receiver.fraction)


class QuadraticReflection(EmittingAreaReflection):
    """Reuse the bounded interaction layout and the exact original near-horizon contributions."""

    def _prepare_pair(self, *args):
        surfaces, details = super()._prepare_pair(*args)
        for i, donor in enumerate(surfaces):
            if donor.spatial is not None:
                coefficients = position_moments(donor.spatial.offset, donor.native_normals, donor.prefix_ld[0])
                surfaces[i] = donor._replace(spatial=_Quadratic(donor.spatial, coefficients))
        return surfaces, details

    def _reference_update(self, receiver, donor, heat):
        if donor.spatial is None:
            return super()._reference_update(receiver, donor, heat)
        plain = donor._replace(spatial=donor.spatial.base)
        return super()._reference_update(receiver, plain, heat) + _second(
            None, np, receiver, plain, heat, donor.spatial
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode != "reference":
            first = self._update
            xp = self.geometry["primary"].jnp

            @self.jax.jit
            def update(receiver, donor, heat):
                if donor.spatial is None:
                    return first(receiver, donor, heat)
                plain = donor._replace(spatial=donor.spatial.base)
                return first(receiver, plain, heat) + _second(self.jax, xp, receiver, plain, heat, donor.spatial)

            self._update = update


class PairedQuadraticLightCurve(PairedEmittingAreaLightCurve):
    solver_type = QuadraticReflection
    rule_prefix = "quadratic_emitting_area"
    field_contract = "Second-order emitter positions in fully lit groups and resolved curved nodes near both horizons"
