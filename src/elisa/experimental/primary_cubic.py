"""Conservative cubic temperature and passband moments on primary Roche faces."""

import numpy as np

from elisa import const
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_reconstructed_field import shape_rule
from elisa.experimental.quadratic_source import QuadraticSourceBackend, receiver_slope
from elisa.experimental.receiver_curvature import curvature_stencil, heat_basis, heat_controls
from elisa.experimental.receiver_heating import ReceiverHeatingLightCurve


_VERTICES = np.eye(3)
_NEXT = np.roll(_VERTICES, -1, axis=0)
CUBIC_NODES = np.vstack((_VERTICES, (2 * _VERTICES + _NEXT) / 3, (_VERTICES + 2 * _NEXT) / 3, np.full(3, 1 / 3)))


def lagrange(xp, bary):
    """Cardinal cubics in vertex, oriented edge-pair and centroid order."""
    other = xp.roll(bary, -1, axis=-1)
    edge = 4.5 * bary * other
    return xp.concatenate(
        (
            0.5 * bary * (3 * bary - 1) * (3 * bary - 2),
            edge * (3 * bary - 1),
            edge * (3 * other - 1),
            27 * xp.prod(bary, axis=-1, keepdims=True),
        ),
        axis=-1,
    )


def bernstein(xp, bary):
    other = xp.roll(bary, -1, axis=-1)
    return xp.concatenate(
        (bary**3, 3 * bary**2 * other, 3 * bary * other**2, 6 * xp.prod(bary, axis=-1, keepdims=True)), axis=-1
    )


_TO_CONTROLS = np.linalg.inv(bernstein(np, CUBIC_NODES))


def cubic_density(xp, weights, gravity, node_gravity, beta, intrinsic, deposits, stencil=None):
    """Preserve the intrinsic and deposited curved-area means on each parent independently."""
    rule = xp.asarray(shape_rule()[0])
    fraction = weights / weights.sum(axis=1, keepdims=True)
    mass = fraction @ lagrange(xp, rule)
    scale = weights.sum() / (weights * gravity**beta).sum()
    density = scale * node_gravity**beta
    density += (intrinsic - (mass * density).sum(axis=1))[:, None]
    limited = 0
    if stencil is not None:
        controls, limited_faces = heat_controls(xp, *deposits, stencil, fraction @ heat_basis(xp, rule))
        density += controls @ heat_basis(xp, xp.asarray(CUBIC_NODES)).T
        limited = xp.sum(limited_faces)
    elif deposits is not None:
        slope, limiter = receiver_slope(xp, *deposits, fraction @ rule)
        density = density + deposits[1][:, None] + slope @ xp.asarray(CUBIC_NODES.T)
        limited = xp.sum(limiter < 1)
    return density, (mass * density).sum(axis=1), limited


class CubicPrimaryBackend(QuadraticSourceBackend):
    """Independent CPU gravity and Lagrange evaluation, device Roche roots and Bernstein moments."""

    curved_heating = False
    reference_basis = staticmethod(lagrange)
    device_basis = staticmethod(bernstein)

    @staticmethod
    def control_values(values):
        return np.einsum("ij,fj...->fi...", _TO_CONTROLS, values)

    def __init__(self, component, law, **kwargs):
        if component != "primary":
            raise UnsupportedCurvedGeometryError("Cubic source validation covers the primary only")
        super().__init__(component, law, **kwargs)
        if self.mode == "reference":
            return
        xp = self.jnp
        rule, qw = shape_rule()

        def thermal(parameters, tri, beta, intrinsic, deposits, stencil=None):
            directions = xp.einsum("ij,fjk->fik", xp.asarray(np.vstack((rule, CUBIC_NODES))), tri)
            rays = directions.reshape(-1, 3)
            state = self.geometry.sample_kernel(parameters[None], rays[None], xp.ones((1, len(rays)), dtype=bool))
            gravity = xp.linalg.norm(state["outward_gradients"][0], axis=-1).reshape(-1, 26)
            jac = state["area_per_solid_angle"][0].reshape(-1, 26)[:, :16]
            det = xp.abs(xp.sum(tri[:, 0] * xp.cross(tri[:, 1], tri[:, 2]), axis=1))
            weights = jac * det[:, None] * xp.asarray(qw) / xp.linalg.norm(directions[:, :16], axis=-1) ** 3
            t4, mean, limited = cubic_density(
                xp, weights, gravity[:, :16], gravity[:, 16:], beta, intrinsic, deposits, stencil
            )
            return t4, mean, limited, gravity[:, 16:], weights.sum(axis=1), state["proposal_valid"][0]

        self._cubic_thermal = self.jax.jit(thermal)

    def emission_nodes(self, model, star, intrinsic, *, deposits=None):
        tri = self.thermal_triangles(star, 26)
        base = star.base_symmetry_faces_number
        faces = tri[:base]
        target = (star.temperatures[:base] / star.t_eff) ** 4
        intrinsic = np.asarray(intrinsic)
        if (
            intrinsic.shape != star.temperatures.shape
            or not np.isfinite(intrinsic).all()
            or np.any(intrinsic <= 0)
            or not np.array_equal(intrinsic, intrinsic[:base][star.face_symmetry_vector])
        ):
            raise UnsupportedCurvedGeometryError(
                "Cubic source requires positive, quarter-symmetric intrinsic temperatures"
            )
        intrinsic = (intrinsic[:base] / star.t_eff) ** 4
        if deposits is not None:
            values, mean = map(np.asarray, deposits)
            if (
                values.shape != (base, 3)
                or mean.shape != (base,)
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or not np.isfinite(mean).all()
                or np.any(mean < 0)
            ):
                raise UnsupportedCurvedGeometryError("Invalid cubic receiver samples")
        if not np.allclose(intrinsic + (0 if deposits is None else mean), target, rtol=3e-13, atol=0):
            raise UnsupportedCurvedGeometryError("Cubic source does not match the reflected face temperatures")
        stencil = curvature_stencil(star.faces, tri, base) if self.curved_heating and deposits is not None else None
        if self.mode == "reference":
            geom = ConditionedRocheGeometry(model, self.component)
            q, w = duffy_nodes(faces, 4)
            points, _, jac = geom.sample(q)
            rays = np.einsum("ij,fjk->fik", CUBIC_NODES, faces).reshape(-1, 3)

            def gravity(points):
                gradient = calculate_potential_gradient(
                    1.0, self.component, points, star.synchronicity, model.mass_ratio
                )
                return np.linalg.norm(gradient, axis=-1)

            node_gravity = gravity(geom.sample(rays)[0]).reshape(base, 10)
            weights = (w * jac).reshape(base, 16)
            density, fitted, limited = cubic_density(
                np,
                weights,
                gravity(points).reshape(base, 16),
                node_gravity,
                star.gravity_darkening,
                intrinsic,
                deposits,
                stencil,
            )
            area, valid = weights.sum(axis=1), True
        else:
            parameters = [model.mass_ratio, star.synchronicity, star.surface_potential]
            density, fitted, limited, node_gravity, area, valid = self._cubic_thermal(
                *(
                    self.jax.device_put(np.asarray(x), self.device)
                    for x in (parameters, faces, star.gravity_darkening, intrinsic)
                ),
                None if deposits is None else self.jax.device_put(deposits, self.device),
                None if stencil is None else self.jax.device_put(stencil, self.device),
            )
        host_density = np.asarray(density)
        closure = float(np.max(np.abs(np.asarray(fitted) / target - 1)))
        area_error = float(np.max(np.abs(np.asarray(area) / star.areas[:base] - 1)))
        if (
            not bool(valid)
            or not (closure <= 3e-13 and area_error <= 3e-13)
            or not np.isfinite(host_density).all()
            or not np.all(self.control_values(host_density) > 0)
        ):
            raise UnsupportedCurvedGeometryError("Cubic source failed curved area, temperature closure or positivity")
        scale = const.G * model.primary.mass / model.semi_major_axis**2
        control = dict(
            area_relative_error=area_error,
            cell_T4_relative_error=closure,
            area_order=4,
            source_faces=len(tri),
            quarter_spectral_samples=10 * base,
            receiver_limited_faces=int(limited),
        )
        return tri, star.t_eff * density**0.25, np.log10(scale * np.asarray(node_gravity)), control


class PrimaryCubicLightCurve(ReceiverHeatingLightCurve):
    rule_prefix = "sampled_receiver_heat_primary_cubic"
    field_contract = "Cubic primary and quadratic secondary thermal and passband fields with sampled receiver heating"

    @staticmethod
    def field_backend(component, law, **kwargs):
        backend = CubicPrimaryBackend if component == "primary" else QuadraticSourceBackend
        return backend(component, law, **kwargs)


class CurvedHeatingBackend(CubicPrimaryBackend):
    curved_heating = True


class CurvedHeatingLightCurve(PrimaryCubicLightCurve):
    rule_prefix = "sampled_receiver_heat_primary_cubic_curved_heating"
    field_contract = "Conservative quadratic primary irradiation with cubic thermal and passband fields"

    @staticmethod
    def field_backend(component, law, **kwargs):
        backend = CurvedHeatingBackend if component == "primary" else QuadraticSourceBackend
        return backend(component, law, **kwargs)
