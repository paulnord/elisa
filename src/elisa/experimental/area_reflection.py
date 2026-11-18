"""Finite-area receiver quadrature and independent spatial grouping of emitting faces."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from elisa import ld, settings, utils
from elisa.binary_system.surface import faces
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.gpu_batched_geometry import MAX_NODES, BatchedRocheGeometry
from elisa.experimental.gpu_reflection import DeviceReflection, UnsupportedReflectionError, _MAX_ITERATIONS
from elisa.experimental.gpu_reflection_tiled import _MAX_FACES, _MAX_PAIRS
from elisa.numba_functions import reflection_effect as reflection

_NAMES = ("primary", "secondary")
_BLOCK = 32
# Symmetric degree-two triangle rule before applying the curved area Jacobian.
_BARY = np.eye(3) / 2 + np.ones((3, 3)) / 6


class _Surface(NamedTuple):
    receiver: Any
    normal: Any
    parent: Any
    fraction: Any
    emitter: Any
    outward: Any
    group: Any
    visible_base: Any
    visible_area: Any
    prefix_ld: Any
    intrinsic: Any
    absorption: Any


def _group_sum(xp: Any, index: Any, count: int, values: Any) -> Any:
    return (
        xp.bincount(index, weights=values, minlength=count)
        if xp is np
        else xp.bincount(
            index,
            weights=values,
            length=count,
        )
    )


def _emission(xp: Any, donor: _Surface, deposited: Any) -> tuple:
    """Aggregate luminosity and luminosity-weighted original directional LD."""
    index, count = donor.group, len(donor.emitter)
    light = (donor.intrinsic + deposited)[donor.visible_base] * donor.visible_area
    total = _group_sum(xp, index, count, light)
    coeff = xp.stack([_group_sum(xp, index, count, light * c) / total for c in donor.prefix_ld])
    return total, coeff


def _cpu_update(receiver: _Surface, donor: _Surface, deposited: np.ndarray, law: str) -> np.ndarray:
    light, coeff = _emission(np, donor, deposited)
    incident = np.empty(len(receiver.parent))
    for start in range(0, len(incident), _BLOCK):
        rows = slice(start, start + _BLOCK)
        distance, join = utils.calculate_distance_matrix(
            receiver.receiver[rows], donor.emitter, return_join_vector_matrix=True
        )
        ga = np.maximum(reflection.gamma_primary(receiver.normal[rows], join), 0)
        gb = np.maximum(reflection.gamma_secondary(donor.outward, join), 0)
        directional = ld.limb_darkening_factor(coefficients=coeff.T, cos_theta=gb.T, limb_darkening_law=law).T
        incident[rows] = ((ga * gb / distance**2 * directional) @ light) * receiver.fraction[rows]
    return receiver.absorption * np.bincount(receiver.parent, weights=incident, minlength=len(receiver.intrinsic))


class AreaReflection:
    """Recompute every receiving mean and donor luminosity in cooler-first iteration order.

    Receiver LD and intrinsic T⁴ remain parent-cell fields. A donor level cap
    groups only original visible descendants. Their total emitting area and
    current luminosity/LD moments contribute on a common ray per ancestor.
    """

    def __init__(self, law: str, *, device: str, receiver_points: int = 3, donor_level: int = 2) -> None:
        if (
            type(receiver_points) is not int
            or receiver_points not in (1, 3)
            or type(donor_level) is not int
            or not 0 <= donor_level <= 3
        ):
            raise ValueError("Use one or three receiver samples and an integer donor level from 0 to 3")
        self.receiver_points, self.donor_level = receiver_points, donor_level
        self.law, self.mode = law, device
        if device == "reference":
            return
        self.geometry = {n: BatchedRocheGeometry(n, device=device) for n in _NAMES}
        primitives = DeviceReflection(law, device=device)
        jax, xp = self.geometry["primary"].jax, self.geometry["primary"].jnp
        self.jax, self.device = jax, self.geometry["primary"].device

        def update(receiver: _Surface, donor: _Surface, deposited: Any) -> Any:
            light, coeff = _emission(xp, donor, deposited)
            size = len(receiver.parent)
            padding = (-size) % _BLOCK
            points = xp.pad(receiver.receiver, ((0, padding), (0, 0)), mode="edge")
            normal = xp.pad(receiver.normal, ((0, padding), (0, 0)), mode="edge")

            def tile(i: Any, output: Any) -> Any:
                start = i * _BLOCK
                _, _, mu, q = primitives._geometry(
                    jax.lax.dynamic_slice_in_dim(points, start, _BLOCK),
                    donor.emitter,
                    jax.lax.dynamic_slice_in_dim(normal, start, _BLOCK),
                    donor.outward,
                )
                values = xp.matmul(q * primitives._directional(mu, coeff), light, precision=jax.lax.Precision.HIGHEST)
                return jax.lax.dynamic_update_slice_in_dim(output, values, start, axis=0)

            values = jax.lax.fori_loop(0, len(points) // _BLOCK, tile, xp.zeros(len(points)))[:size]
            return receiver.absorption * _group_sum(
                xp, receiver.parent, len(receiver.intrinsic), values * receiver.fraction
            )

        self._update = jax.jit(update)

    def _sample(self, binary: Any, name: str, rays: np.ndarray) -> tuple:
        if self.mode == "reference":
            return ConditionedRocheGeometry(binary, name).sample(rays)
        star = getattr(binary, name)
        result = self.geometry[name].sample([[binary.mass_ratio, star.synchronicity, star.surface_potential]], rays)
        if not bool(result["proposal_valid"][0]):
            raise UnsupportedReflectionError("Unsupported Roche roots in reflection quadrature")
        return tuple(np.asarray(result[k][0]) for k in ("points", "normals", "area_per_solid_angle"))

    def _surface(self, binary: Any, system: Any, name: str, level: int, xlim: float) -> _Surface:
        star = getattr(system, name)
        n = int(star.base_symmetry_faces_number)
        mirror = np.asarray(star.face_symmetry_vector)
        if (
            len(star.faces) > _MAX_FACES
            or not 0 < n * self.receiver_points <= MAX_NODES
            or not star.symmetry_test()
            or not np.array_equal(mirror, np.tile(np.arange(n), 4))
        ):
            raise UnsupportedReflectionError("Receiver integration exceeds the symmetric surface budget")
        origin = np.array([float(name == "secondary"), 0.0, 0.0])
        centre, normal = star.face_centres, star.normals
        if self.receiver_points == 1:
            points, normals, frac = centre[:n], normal[:n], np.ones(n)
        else:
            directions = star.points - origin
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            coords = (_BARY @ directions[star.faces[:n]]).reshape(-1, 3)
            points, normals, jac = self._sample(binary, name, coords)
            # Constant triangle determinant and common cubature weight cancel in dA / area.
            w = (jac / np.linalg.norm(coords, axis=1) ** 3).reshape(n, self.receiver_points)
            frac = (w / w.sum(axis=1)[:, None]).ravel()
            if not np.isfinite(frac).all() or np.any(frac <= 0):
                raise UnsupportedReflectionError("Invalid curved receiver area weights")
        recv = faces.visibility_test(points, xlim, name)
        vis = np.flatnonzero(faces.visibility_test(centre, xlim, name))
        if not len(vis) or not np.any(recv):
            raise UnsupportedReflectionError("Empty original reflection visibility selection")
        width = 4 ** max(level - self.donor_level, 0)
        if n % width:
            raise UnsupportedReflectionError("Donor grouping lost nested quarter-face ancestry")
        # NestedMesh keeps descendants contiguous, including after each of its four mirrors.
        _, group = np.unique(vis // width, return_inverse=True)
        area = star.areas[vis]
        if width == 1:
            emit, outward = centre[vis], normal[vis]
        else:
            # Positive area centroid of only the visible descendants, projected back onto the Roche surface.
            rays = np.column_stack([np.bincount(group, weights=area * v) for v in (centre[vis] - origin).T])
            emit, outward, _ = self._sample(binary, name, rays)
        coefficients = ld.get_bolometric_ld_coefficients(
            star.temperatures, star.log_g, star.metallicity, star.limb_darkening_coefficients
        )
        integral = ld.calculate_integrated_limb_darkening_factor(self.law, coefficients)
        if (
            not np.isfinite(coefficients).all()
            or not np.all(integral > 0)
            or not 0 <= star.albedo <= 1
            or not np.all(area > 0)
            or not np.all(star.temperatures > 0)
        ):
            raise UnsupportedReflectionError("Invalid reflection LD, intrinsic temperature, area or albedo")
        return _Surface(
            points[recv],
            normals[recv],
            np.repeat(np.arange(n), self.receiver_points)[recv],
            frac[recv],
            emit,
            outward,
            group,
            mirror[vis],
            area,
            coefficients[:, : len(vis)],
            star.temperatures[:n] ** 4,
            star.albedo / integral[:n],
        )

    def evaluate(self, binary: Any, system: Any, level: int, iterations: int) -> tuple[dict, dict]:
        if (
            type(level) is not int or not 0 <= level <= 3
            or system.morphology != "detached"
            or system.eccentricity != 0
            or system.has_spots()
            or system.has_pulsations()
            or not isinstance(iterations, (int, np.integer))
            or not 0 <= iterations <= _MAX_ITERATIONS
            or settings.LIMB_DARKENING_LAW != self.law
        ):
            raise UnsupportedReflectionError("Unsupported reflection geometry, law or iteration count")
        limits = faces.faces_visibility_x_limits(system.primary.polar_radius, system.secondary.polar_radius, 1.0)
        surfaces = [self._surface(binary, system, n, level, limits[n]) for n in _NAMES]
        pairs = sum(len(a.parent) * len(b.emitter) for a, b in (surfaces, surfaces[::-1]))
        if pairs > _MAX_PAIRS:
            raise UnsupportedReflectionError("Integrated receiver/emitter quadrature exceeds the reflection pair cap")
        stats = {
            n: {
                "receiver_samples": len(s.parent),
                "donor_faces": len(s.visible_base),
                "donor_groups": len(s.emitter),
                "pairs_per_sweep_both_stars": pairs,
            }
            for n, s in zip(_NAMES, surfaces, strict=True)
        }
        if self.mode != "reference":
            if not self.jax.config.x64_enabled:
                raise RuntimeError("Area reflection requires FP64")
            surfaces = [self.jax.device_put(s, self.device) for s in surfaces]
        heat = [s.intrinsic * 0.0 for s in surfaces]
        order = (0, 1) if system.primary.t_eff <= system.secondary.t_eff else (1, 0)
        for _ in range(iterations):
            for i in order:
                args = surfaces[i], surfaces[1 - i], heat[1 - i]
                heat[i] = _cpu_update(*args, self.law) if self.mode == "reference" else self._update(*args)
        out = {}
        for n, s, h in zip(_NAMES, surfaces, heat, strict=True):
            star = getattr(system, n)
            checked = np.asarray(h)
            if not np.isfinite(checked).all() or np.any(checked < 0):
                raise UnsupportedReflectionError("Invalid area-integrated deposited heat")
            base = star.temperatures[: len(h)]
            out[n] = (base * (1 + h / s.intrinsic) ** 0.25)[star.face_symmetry_vector]
            stats[n]["deposited_T4_area_mean_over_Teff4"] = float(
                np.average(
                    checked / star.t_eff**4,
                    weights=star.areas[: len(h)],
                )
            )
        return out, stats
