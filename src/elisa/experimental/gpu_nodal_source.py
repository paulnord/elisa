"""Face-local gravity-darkened passband fields on curved eclipse packets."""

# ruff: noqa: PLR2004, SLF001 - explicit finite-model quadrature and owned kernels
from __future__ import annotations

from copy import copy
from functools import partial
from time import perf_counter
from typing import Any

import numpy as np

from elisa import const
from elisa.binary_system.curves.utils import prep_surface_params
from elisa.binary_system.surface.gravity import calculate_potential_gradient
from elisa.experimental.area_reflection import AreaReflection
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_atmosphere import check_ld_boundaries, check_table_transitions
from elisa.experimental.gpu_reconstructed_field import MomentFieldBackend, PreparedMomentField, shape_rule
from elisa.experimental.gpu_reconstructed_observer import ReconstructedObserverLightCurve, partitioned_packet
from elisa.experimental.receiver_irradiation import receiver_slopes, receiver_stencil


class _NodalBackend(MomentFieldBackend):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.mode == "reference":
            return
        xp = self.jnp
        barycentric, weights = shape_rule()

        def thermal(parameters: Any, tri: Any, areas: Any, face_t: Any, physics: Any, heating: tuple) -> tuple:
            teff, beta, gravity_scale = physics
            coords = xp.einsum("qj,fjk->fqk", xp.asarray(barycentric), tri)
            all_rays = xp.concatenate([coords.reshape(-1, 3), tri.reshape(-1, 3)])
            result = self.geometry.sample_kernel(
                parameters[None],
                all_rays[None],
                xp.ones((1, len(all_rays)), dtype=bool),
            )
            n = coords.shape[0] * 16
            norms = xp.linalg.norm(result["outward_gradients"][0], axis=1)
            det = xp.abs(xp.sum(tri[:, 0] * xp.cross(tri[:, 1], tri[:, 2]), axis=1))
            w = (
                result["area_per_solid_angle"][0, :n].reshape(-1, 16)
                * xp.asarray(weights)
                * det[:, None]
                / xp.linalg.norm(coords, axis=2) ** 3
            )
            area = w.sum(axis=1)
            shape_mass = w @ xp.asarray(barycentric) / area[:, None]
            g = norms[n:].reshape(-1, 3)
            corner = g**beta
            scale = area.sum() / xp.sum(w * norms[:n].reshape(-1, 16) ** beta)
            t4 = (face_t[:, None] / teff) ** 4 + scale * (corner - (shape_mass * corner).sum(axis=1)[:, None])
            span = limited = 0.0
            if heating:
                intrinsic, neighbours, transform = heating
                heat = (face_t / teff) ** 4 - (intrinsic / teff) ** 4
                slope, limiter = receiver_slopes(xp, heat, neighbours, transform, shape_mass)
                t4 = t4 + slope
                span = xp.max(xp.ptp(slope, axis=1))
                limited = xp.sum(limiter < 1)
            closure = xp.max(xp.abs((shape_mass * t4).sum(axis=1) / (face_t / teff) ** 4 - 1))
            error = xp.max(xp.abs(area / areas - 1))
            valid = result["proposal_valid"][0] & xp.all(t4 > 0) & xp.all(shape_mass > 0)
            return teff * t4.ravel() ** 0.25, xp.log10(gravity_scale * g.ravel()), error, closure, valid, span, limited

        self._thermal_nodes = self.jax.jit(thermal)

    def thermal_nodes(self, model: Any, star: Any, intrinsic: Any = None) -> tuple:
        base = int(star.base_symmetry_faces_number)
        if base <= 0 or base * 19 > self.max_nodes or len(star.faces) != 4 * base:
            msg = "Nodal thermal source exceeds its symmetric quadrature budget"
            raise UnsupportedCurvedGeometryError(msg)
        # Face-local corner order matters when a mirrored face is integrated.
        expected = np.concatenate([idx[star.faces[:base]] for idx in star.inverse_point_symmetry_matrix])
        mapping = np.tile(np.arange(base), 4)
        if (
            not np.array_equal(star.faces, expected)
            or not np.array_equal(star.face_symmetry_vector, mapping)
            or not np.allclose(star.temperatures, star.temperatures[:base][mapping], rtol=3e-13, atol=0)
        ):
            msg = "Nodal source requires verified quarter symmetry of corners and deposited heat"
            raise UnsupportedCurvedGeometryError(msg)
        rays = star.points - np.array([0.0 if self.component == "primary" else 1.0, 0.0, 0.0])
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        tri = rays[star.faces]
        faces = tri[:base]
        scale = const.G * model.primary.mass / model.semi_major_axis**2
        heating = ()
        span = limited = 0.0
        if intrinsic is not None:
            intrinsic = np.asarray(intrinsic)
            if (
                intrinsic.shape != star.temperatures.shape
                or not np.isfinite(intrinsic).all()
                or np.any(intrinsic <= 0)
                or np.any(intrinsic > star.temperatures)
                or not np.array_equal(intrinsic, intrinsic[:base][mapping])
            ):
                msg = "Receiver reconstruction requires positive symmetric intrinsic temperatures and nonnegative deposits"
                raise UnsupportedCurvedGeometryError(msg)
            heating = (intrinsic[:base], *receiver_stencil(star.faces, tri, base))
        if self.mode == "reference":
            geom = ConditionedRocheGeometry(model, self.component)
            points, weights = duffy_nodes(faces, 4)
            sample, _, jac = geom.sample(points)
            vertices = geom.sample(faces.reshape(-1, 3))[0]

            def gravity(x: Any) -> np.ndarray:
                return np.linalg.norm(
                    calculate_potential_gradient(1.0, self.component, x, star.synchronicity, model.mass_ratio),
                    axis=1,
                )

            w = (weights * jac).reshape(base, 16)
            area = w.sum(axis=1)
            mass = (w @ shape_rule()[0]) / area[:, None]
            g = gravity(vertices).reshape(base, 3)
            power = g**star.gravity_darkening
            polar_scale = area.sum() / np.sum(w * gravity(sample).reshape(base, 16) ** star.gravity_darkening)
            mean = (star.temperatures[:base] / star.t_eff) ** 4
            density = mean[:, None] + polar_scale * (power - np.sum(mass * power, axis=1)[:, None])
            if heating:
                heat = mean - (heating[0] / star.t_eff) ** 4
                slope, limiter = receiver_slopes(np, heat, *heating[1:], mass)
                density += slope
                span, limited = np.max(np.ptp(slope, axis=1)), np.sum(limiter < 1)
            valid = np.all(density > 0) and np.all(mass > 0)
            error = np.max(np.abs(area / star.areas[:base] - 1))
            closure = np.max(np.abs(np.sum(mass * density, axis=1) / mean - 1))
            if not valid:
                msg = "Nonpositive corner emission or area mass"
                raise UnsupportedCurvedGeometryError(msg)
            temp = star.t_eff * density.ravel() ** 0.25
            log_g = np.log10(scale * g.ravel())
        else:
            if not self.jax.config.x64_enabled:
                msg = "Nodal source requires FP64"
                raise RuntimeError(msg)
            args = [
                [model.mass_ratio, star.synchronicity, star.surface_potential],
                faces,
                star.areas[:base],
                star.temperatures[:base],
                [star.t_eff, star.gravity_darkening, scale],
            ]
            temp, log_g, error, closure, valid, span, limited = self._thermal_nodes(
                *(self.jax.device_put(np.asarray(x), self.device) for x in args),
                tuple(self.jax.device_put(x, self.device) for x in heating),
            )
        if not bool(valid) or not float(error) <= 3e-13 or not float(closure) <= 3e-13:
            msg = "Nodal source failed curved area, positivity or per-cell T^4 closure"
            raise UnsupportedCurvedGeometryError(msg)
        control = {
            "area_relative_error": float(error),
            "cell_T4_relative_error": float(closure),
            "quarter_corner_samples": 3 * base,
            "source_faces": len(tri),
            "area_order": 4,
        }
        if heating:
            control.update(maximum_deposit_T4_span=float(span), positivity_limited_faces=int(limited))
        return tri, temp, np.asarray(log_g), control

    def interpolate_field(self, triangles: Any, spectral: Any, bands: tuple, control: dict) -> PreparedMomentField:
        condition = np.linalg.cond(triangles)
        if not np.isfinite(condition).all() or condition.max() > 1e5:
            msg = "Nodal source angular parent exceeds conditioning cap"
            raise UnsupportedCurvedGeometryError(msg)
        r = np.stack([spectral.normal_radiance[b] for b in bands], axis=1)
        c = np.stack([spectral.ld_cfs[b] for b in bands], axis=1)
        if not np.isfinite(r).all() or np.any(r <= 0) or not np.isfinite(c).all():
            msg = "Nonfinite nodal spectrum or limb-darkening coefficients"
            raise UnsupportedCurvedGeometryError(msg)
        moments = np.concatenate([r[..., None], r[..., None] * c], axis=-1)
        face_corners = np.tile(np.arange(len(r)).reshape(-1, 3), (4, 1))
        control.update(
            maximum_condition=float(condition.max()),
            max_within_face_relative_intensity_span=float(
                np.max(np.ptp(r.reshape(-1, 3, len(bands)), axis=1) / r.reshape(-1, 3, len(bands)).mean(axis=1)),
            ),
            max_within_face_LD_span=float(np.max(np.ptp(c.reshape(-1, 3, *c.shape[1:]), axis=1))),
        )
        return PreparedMomentField(
            self,
            triangles,
            face_corners,
            np.linalg.inv(triangles.transpose(0, 2, 1)),
            condition,
            moments,
            bands,
            control,
        )


class NodalSourceLightCurve(ReconstructedObserverLightCurve):
    """Fit a linear angular intensity field to three physical corner samples per face.

    The intrinsic slope follows Roche gravity. The curved-area mean of the
    face's post-irradiation T^4 is preserved. Irradiation is deposited per face.
    The optical spectrum and LD integral are evaluated afresh at each corner.
    """

    reconstruct_heating = False
    field_backend = _NodalBackend
    field_rule = "face_local_spectral_nodes"
    field_contract = "Face-local passband intensity/LD moments with curved-area conservation of thermal emission"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, area_thermal=True, **kwargs)

    def _prepare_field(self, model: Any, system: Any, observer: Any, variant: Any) -> dict:
        start = perf_counter()
        sampled = copy(system)
        geometry, temperatures, controls = {}, {}, {}
        for name, backend in self.moment_backends.items():
            star = getattr(system, name)
            intrinsic = (
                self.engines[variant.key].last_diagnostics[name]["intrinsic_temperatures"]
                if self.reconstruct_heating
                else None
            )
            tri, temp, log_g, control = backend.thermal_nodes(model, star, intrinsic)
            geometry[name], temperatures[name], controls[name] = tri, temp, control
            # Each row denotes one corner of the corresponding base face. All
            # base samples are distinct until the outer, verified mirror map.
            view = copy(star)
            view.temperatures, view.log_g = np.asarray(temp), log_g
            view.faces = np.repeat(star.faces[: star.base_symmetry_faces_number], 3, axis=0)
            view.base_symmetry_faces_number = len(temp)
            view.face_symmetry_vector = np.arange(len(temp))
            setattr(sampled, name, view)
        check_ld_boundaries(sampled, observer.passband)
        if sampled.primary.atmosphere == "ck04":
            check_table_transitions(sampled)
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
            self.engines[variant.key].helper._prepare_radiance(sampled, observer, temperatures, diagnostics=True)
        fields = {
            name: backend.interpolate_field(geometry[name], getattr(sampled, name), self.passbands, controls[name])
            for name, backend in self.moment_backends.items()
        }
        for item in fields.values():
            item.control["both_stars_preparation_seconds"] = perf_counter() - start
        return fields


class ReceiverSourceLightCurve(NodalSourceLightCurve):
    """Recover illumination gradients from each fresh extended-source reflection solution."""

    reconstruct_heating = True
    field_rule = "face_local_spectral_and_irradiation_nodes"
    field_contract = "Conservative receiver heating gradients from original iterated face deposits"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._packet = partitioned_packet

    def _prepare(self, model: Any, observer: Any) -> tuple:
        for engine in self.engines.values():
            engine.keep_states = True
        return super()._prepare(model, observer)

class AreaReflectedLightCurve(ReceiverSourceLightCurve):
    """Integrate a receiver-cell mean within each iteration before rebuilding its spectra."""

    field_contract = "Area-averaged iterated irradiation, conservative local thermal and passband fields"

    def __init__(self, *args: Any, receiver_points: int = 3, donor_level: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        solver = AreaReflection(self.law, device=self.device, receiver_points=receiver_points, donor_level=donor_level)
        self.field_rule = f"area_reflection_receiver{receiver_points}_donor_cap{donor_level}"
        for variant in self.variants:
            self.engines[variant.key].area_reflection = partial(solver.evaluate, level=variant.level)

    def _prepare_field(self, model: Any, system: Any, observer: Any, variant: Any) -> dict:
        fields = super()._prepare_field(model, system, observer, variant)
        for name, field in fields.items():
            stats = self.engines[variant.key].last_diagnostics[name]
            if "reflection_quadrature" in stats:
                field.control.update(stats["reflection_quadrature"])
        return fields
