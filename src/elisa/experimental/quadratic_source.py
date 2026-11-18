"""Six physical spectra per curved face, with conservative quadratic thermal reconstruction."""

from copy import copy

import numpy as np

from elisa import const
from elisa.binary_system.curves.utils import prep_surface_params
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_atmosphere import check_ld_boundaries
from elisa.experimental.gpu_nodal_source import _NodalBackend
from elisa.experimental.gpu_reconstructed_field import PreparedMomentField, reference_barycentric, shape_rule
from elisa.experimental.quadratic_reflection import PairedQuadraticLightCurve


def shape_values(xp, bary):
    """Quadratic Lagrange shape functions in vertex, then cyclic edge order."""
    return xp.concatenate((bary * (2 * bary - 1), 4 * bary * xp.roll(bary, -1, axis=-1)), axis=-1)


def bernstein_edges(values):
    """Edge control values of the same triangular quadratic, given its six samples."""
    corners = values[:, :3]
    return 2 * values[:, 3:] - 0.5 * (corners + np.roll(corners, -1, axis=1))


def receiver_slope(xp, samples, heat_mean, mass):
    """Invert the symmetric three-point triangle rule and preserve the curved-area mean.

    The inverse has twice the sample differences. Limit their amplitude when extrapolation
    to an angular corner would make deposited power negative.
    """
    span = 2 * samples
    slope = span - (mass * span).sum(axis=1)[:, None]
    lower = xp.maximum(0, -xp.min(slope, axis=1))
    scale = xp.minimum(1.0, heat_mean / xp.where(lower > 0, lower, 1.0))
    scale = xp.where(lower > 0, scale, 1.0)
    return slope * scale[:, None], scale


def correct_midpoints(xp, corner_density, gcorner, gmid, gquad, weights, beta):
    density, mean, _ = _correct_midpoints(xp, corner_density, gcorner, gmid, gquad, weights, beta, None)
    return density, mean


def _correct_midpoints(xp, corner_density, gcorner, gmid, gquad, weights, beta, deposits):
    basis = xp.asarray(shape_values(np, shape_rule()[0]))
    mass = (weights @ basis) / weights.sum(axis=1)[:, None]
    scale = weights.sum() / (weights * gquad**beta).sum()
    limited = 0
    if deposits is not None:
        linear_mass = (weights @ xp.asarray(shape_rule()[0])) / weights.sum(axis=1)[:, None]
        slope, limiter = receiver_slope(xp, *deposits, linear_mass)
        corner_density = corner_density + slope
        limited = xp.sum(limiter < 1)
    vertex = gcorner**beta
    delta = scale * (gmid**beta - 0.5 * (vertex + xp.roll(vertex, -1, axis=1)))
    midpoint = 0.5 * (corner_density + xp.roll(corner_density, -1, axis=1))
    density = xp.concatenate((corner_density, midpoint + delta), axis=1)
    density -= (mass[:, 3:] * delta).sum(axis=1)[:, None]
    return density, (mass * density).sum(axis=1), limited


class _QuadraticField(PreparedMomentField):
    def sample(self, parent, rays):
        p, r = self.validate(parent, rays)
        if self.backend.mode != "reference" or not len(p):
            return super().sample(parent, rays)
        # The reference solves the cone per ray and uses signed Lagrange shape functions.
        c, _ = reference_barycentric(self.triangles, self.arrays[1], p, r)
        return np.einsum("nk,nkbc->nbc", self.backend.reference_basis(np, c), self.vertex[self.faces[p]])


class QuadraticSourceBackend(_NodalBackend):
    reference_basis = staticmethod(shape_values)

    @staticmethod
    def device_basis(xp, bary):
        return xp.concatenate((bary * bary, 2 * bary * xp.roll(bary, -1, axis=-1)), axis=-1)

    @staticmethod
    def control_values(values):
        return np.concatenate((values[:, :3], bernstein_edges(values)), axis=1)

    def __init__(self, component, law, **kwargs):
        if law != "linear":
            raise UnsupportedCurvedGeometryError("Quadratic optical reconstruction requires linear limb darkening")
        super().__init__(component, law, **kwargs)
        if self.mode == "reference":
            return
        xp = self.jnp
        bary, qw = shape_rule()

        def correct(parameters, tri, corner_density, gvertex, beta, deposits):
            mid = 0.5 * (tri + xp.roll(tri, -1, axis=1))
            points = xp.einsum("qj,fjk->fqk", xp.asarray(bary), tri)
            rays = xp.concatenate((mid, points), axis=1).reshape(-1, 3)
            sample = self.geometry.sample_kernel(parameters[None], rays[None], xp.ones((1, len(rays)), dtype=bool))
            gravity = xp.linalg.norm(sample["outward_gradients"][0], axis=-1).reshape(-1, 19)
            jac = sample["area_per_solid_angle"][0].reshape(-1, 19)[:, 3:]
            det = xp.abs(xp.sum(tri[:, 0] * xp.cross(tri[:, 1], tri[:, 2]), axis=1))
            weights = jac * det[:, None] * xp.asarray(qw) / xp.linalg.norm(points, axis=-1) ** 3
            density, mean, limited = _correct_midpoints(
                xp, corner_density, gvertex, gravity[:, :3], gravity[:, 3:], weights, beta, deposits
            )
            return density, mean, gravity[:, :3], sample["proposal_valid"][0], limited

        def nodes(inverse, condition, faces, controls, parent, rays):
            c = xp.einsum("nij,nj->ni", inverse[parent], rays)
            guard = 64 * np.finfo(float).eps * condition[parent] * xp.maximum(1.0, xp.abs(c).max(axis=1))
            valid = xp.all(xp.isfinite(c)) & xp.all(c >= -guard[:, None]) & xp.all(c.sum(axis=1) > 0)
            c = xp.maximum(c, 0)
            c = c / c.sum(axis=1)[:, None]
            basis = self.device_basis(xp, c)
            m = xp.einsum("nk,nkbc->nbc", basis, controls[faces[parent]])
            return {"moments": m, "valid": valid}

        def flux(inverse, condition, faces, controls, parent, rays, mu, area):
            state = nodes(inverse, condition, faces, controls, parent, rays)
            rad, ra = state["moments"][..., 0], state["moments"][..., 1]
            intensity = rad - (1 - mu[:, None]) * ra
            return {
                "flux": xp.sum((xp.maximum(mu, 0) * area)[:, None] * intensity, axis=0),
                "valid": state["valid"] & xp.all(rad > 0) & xp.all(xp.isfinite(intensity)),
            }

        self._correct = self.jax.jit(correct)
        self._nodes = self.jax.jit(nodes)
        self._flux = self.jax.jit(flux)

    def six_nodes(self, model, star, intrinsic, *, deposits=None):
        tri, corner, log_g, control = super().thermal_nodes(model, star, intrinsic if deposits is None else None)
        base = star.base_symmetry_faces_number
        if deposits is not None:
            raw, deposited_mean = (np.asarray(v) for v in deposits)
            if (
                raw.shape != (base, 3)
                or deposited_mean.shape != (base,)
                or not np.isfinite(raw).all()
                or not np.isfinite(deposited_mean).all()
                or np.any(raw < 0)
                or np.any(deposited_mean < 0)
            ):
                raise UnsupportedCurvedGeometryError("Invalid three-node heating")
            closed = (np.asarray(intrinsic[:base]) / star.t_eff) ** 4 + deposited_mean
            if not np.allclose(closed, (star.temperatures[:base] / star.t_eff) ** 4, rtol=3e-13, atol=0):
                raise UnsupportedCurvedGeometryError("Receiver samples belong to a different thermal field")
        faces = tri[:base]
        corner_density = (np.asarray(corner).reshape(-1, 3) / star.t_eff) ** 4
        scale = const.G * model.primary.mass / model.semi_major_axis**2
        gvertex = 10 ** log_g.reshape(-1, 3) / scale
        if self.mode == "reference":
            geom = ConditionedRocheGeometry(model, self.component)
            mid = 0.5 * (faces + np.roll(faces, -1, axis=1))
            nodes, weights = duffy_nodes(faces, 4)
            points, _, jac = geom.sample(nodes)

            def magnitude(points):
                return np.linalg.norm(
                    calculate_potential_gradient(1.0, self.component, points, star.synchronicity, model.mass_ratio),
                    axis=1,
                )

            gmid = magnitude(geom.sample(mid.reshape(-1, 3))[0]).reshape(-1, 3)
            density, mean, limited = _correct_midpoints(
                np,
                corner_density,
                gvertex,
                gmid,
                magnitude(points).reshape(-1, 16),
                (weights * jac).reshape(-1, 16),
                star.gravity_darkening,
                deposits,
            )
        else:
            parameters = [model.mass_ratio, star.synchronicity, star.surface_potential]
            density, mean, gmid, valid, limited = self._correct(
                *(
                    self.jax.device_put(np.asarray(v), self.device)
                    for v in (parameters, faces, corner_density, gvertex, star.gravity_darkening)
                ),
                None if deposits is None else self.jax.device_put(deposits, self.device),
            )
            if not bool(valid):
                raise UnsupportedCurvedGeometryError("Unsupported quadratic source midpoint geometry")
            mean, gmid = np.asarray(mean), np.asarray(gmid)
        closure = float(np.max(np.abs(mean / (star.temperatures[:base] / star.t_eff) ** 4 - 1)))
        if not closure <= 3e-13 or not np.all(density > 0) or not np.all(bernstein_edges(density) > 0):
            raise UnsupportedCurvedGeometryError("Quadratic thermal field failed T^4 closure or positivity")
        if deposits is not None:
            control["receiver_limited_faces"] = int(limited)
        control.update(quadratic_T4_relative_error=closure, quarter_spectral_samples=int(6 * base))
        return (
            tri,
            star.t_eff * density**0.25,
            np.concatenate((log_g.reshape(-1, 3), np.log10(scale * gmid)), axis=1),
            control,
        )

    emission_nodes = six_nodes

    def optical_field(self, tri, values, bands, control):
        # Each moment is (normal radiance, normal radiance * linear LD coefficient).
        controls = self.control_values(values)
        if (
            not np.isfinite(controls).all()
            or not np.all(controls[..., 0] > 0)
            or np.any(controls[..., 0] - controls[..., 1] < 0)
        ):
            raise UnsupportedCurvedGeometryError(
                "Interpolated spectrum cannot certify nonnegative intensity across the face"
            )
        condition = np.linalg.cond(tri)
        if not np.isfinite(condition).all() or np.any(condition > 1e5):
            raise UnsupportedCurvedGeometryError("Ill-conditioned optical face")
        array = values if self.mode == "reference" else controls
        faces = np.tile(np.arange(len(array) * array.shape[1]).reshape(len(array), array.shape[1]), (4, 1))
        return _QuadraticField(
            self,
            tri,
            faces,
            np.linalg.inv(tri.transpose(0, 2, 1)),
            condition,
            array.reshape(-1, *array.shape[2:]),
            bands,
            control,
        )


class QuadraticSourceLightCurve(PairedQuadraticLightCurve):
    """Resolve intrinsic source curvature and passband emission on the fixed irradiation grid."""

    field_backend = QuadraticSourceBackend
    rule_prefix = "quadratic_spectral_source"
    field_contract = "Six-node conservative thermal and optical fields with two iterations of curved donor irradiation"

    def _source_data(self, backend, model, star, diagnostics):
        return backend.six_nodes(model, star, diagnostics["intrinsic_temperatures"]), diagnostics.get(
            "reflection_quadrature", {}
        )

    def _prepare_field(self, model, system, observer, variant):
        engine = self.engines[variant.key]
        data, reflection = {}, {}
        for name, b in self.moment_backends.items():
            star = getattr(system, name)
            if star.atmosphere != "bb":
                raise UnsupportedCurvedGeometryError("Quadratic source validation requires a blackbody atmosphere")
            data[name], reflection[name] = self._source_data(b, model, star, engine.last_diagnostics[name])
        return self._optical_fields(system, observer, engine, data, reflection)

    def _optical_fields(self, system, observer, engine, data, reflection):
        spectra = {name: [] for name in data}
        # Use every source node once, with at most three spectra per face in each Planck batch.
        batches = (max(data[n][1].shape[1] for n in data) + 2) // 3
        slots = {n: np.array_split(np.arange(data[n][1].shape[1]), batches) for n in data}
        for batch in range(batches):
            sampled = copy(system)
            temperatures = {}
            for name, (_, temp, log_g, _) in data.items():
                star = getattr(system, name)
                view = copy(star)
                indices = slots[name][batch]
                columns = slice(int(indices[0]), int(indices[-1]) + 1)
                temperatures[name] = temp[:, columns].ravel()
                view.temperatures = np.asarray(temperatures[name])
                view.log_g = log_g[:, columns].ravel()
                view.faces = np.repeat(star.faces[: star.base_symmetry_faces_number], len(indices), axis=0)
                view.base_symmetry_faces_number = len(view.faces)
                view.face_symmetry_vector = np.arange(len(view.faces))
                setattr(sampled, name, view)
            check_ld_boundaries(sampled, observer.passband)
            if self.device == "reference" or self.radiance == "cpu":
                prep_surface_params(
                    sampled,
                    return_values=False,
                    write_to_containers=True,
                    passband=observer.passband,
                    left_bandwidth=observer.left_bandwidth,
                    right_bandwidth=observer.right_bandwidth,
                )
            else:
                engine.helper._prepare_radiance(sampled, observer, temperatures, diagnostics=True)
            for name in data:
                view = getattr(sampled, name)
                r = np.stack([view.normal_radiance[b] for b in self.passbands], axis=1)
                a = np.stack([view.ld_cfs[b][:, 0] for b in self.passbands], axis=1)
                spectra[name].append(np.stack((r, r * a), axis=-1).reshape(-1, len(slots[name][batch]), len(self.passbands), 2))
        return {
            name: b.optical_field(
                data[name][0],
                np.concatenate(spectra[name], axis=1),
                self.passbands,
                {**data[name][-1], **reflection[name]},
            )
            for name, b in self.moment_backends.items()
        }
