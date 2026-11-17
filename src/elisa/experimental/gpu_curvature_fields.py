"""Curvature-consistent finite Roche fields and complete shared-grid curves.

Cell centres lie on the actual equipotential; normals/gravity are evaluated
there and cell areas integrate the curved patch. Original thermal/reflection/
radiance formulas use this geometry, with no scalar mesh correction. This is
a new opt-in finite model, not a change to original CPU/default semantics.
"""

# ruff: noqa: SLF001, PLR2004 - explicit composition of owned experimental kernels
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from elisa import const, settings
from elisa.base.surface.temperature import calculate_effective_temperatures, renormalize_temperatures
from elisa.binary_system.curves.utils import prep_surface_params
from elisa.binary_system.surface.gravity import (
    calculate_polar_potential_gradient_magnitude,
    calculate_potential_gradient,
)
from elisa.experimental.boundary_geometry import BoundaryRule, ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_gravity import RocheSurfaceGravity
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_reference import cpu_reflect
from elisa.experimental.gpu_reflection_tiled import TiledDeviceReflection
from elisa.experimental.gpu_selective_mesh import seed_system
from elisa.experimental.gpu_shared_fields import SharedFieldLightCurve, SharedFieldResult
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator
from elisa.experimental.gpu_thermal import GravityDarkeningTemperature

_NAMES = ("primary", "secondary")
_SIGNS = np.array([[1., 1., 1.], [1., -1., 1.], [1., -1., -1.], [1., 1., -1.]])


@dataclass(frozen=True)
class CurvatureFieldSpec:
    """Fixed finite field level; no scalar-correction parameter or prior cut."""

    level: int

    def __post_init__(self) -> None:
        if isinstance(self.level, bool) or not isinstance(self.level, int) or not 0 <= self.level <= 3:
            msg = "Curvature field level must be integer0-3"
            raise ValueError(msg)

    @property
    def key(self) -> str:
        return f"curvature_field{self.level}"

    @property
    def correction_level(self) -> int:
        """Explicit not-applicable marker in the inherited finite provenance."""
        return -1


def curvature_geometry(binary: Any, level: int) -> tuple[Any, dict, dict]:
    """Fresh original angular topology, true cell centres and curved cell areas.

    Only the original base quarter is evaluated, then exactly mirrored. The
    cell collocation ray is the normalized mean of its angular corner rays.
    Order4 Duffy quadrature integrates dA/dOmega over each angular cell.
    Geometry/area convergence remains separate from field/implementation parity.
    """
    CurvatureFieldSpec(level)
    geometry = {name: ConditionedRocheGeometry(binary, name) for name in _NAMES}
    system = seed_system(binary)
    NestedMesh(level).apply(system)
    cells, control = {}, {}
    for name, x in zip(_NAMES, (0., 1.), strict=True):
        star = getattr(system, name)
        nbase = int(star.base_symmetry_faces_number)
        d = star.points-np.array([x, 0, 0])
        d /= np.linalg.norm(d, axis=1)[:, None]
        triangles = d[star.faces]
        points, _, _ = geometry[name].sample(triangles[:nbase].mean(axis=1))
        nodes, weights = duffy_nodes(triangles[:nbase], 4)
        _, _, jacobian = geometry[name].sample(nodes)
        area = (weights*jacobian).reshape((nbase, 16)).sum(axis=1)
        residual = np.abs(geometry[name].potential(points)/star.surface_potential-1)
        if np.any(area <= 0) or not np.isfinite(area).all() or np.any(residual > 3e-13):
            msg = "Curvature field geometry failed area/equipotential gates"
            raise UnsupportedCurvedGeometryError(msg)
        cells[name] = {"points": points, "areas": area, "triangles": triangles}
        control[name] = {"faces": len(star.faces), "curved_area_SMA2": float(4*area.sum()),
                         "max_collocation_potential_relative_residual": float(residual.max()),
                         "area_order": 4, "scalar_correction": False}
    return system, cells, control


class _PointState:
    """Existing device point gravity and thermal law on true Roche cell centres."""

    def __init__(self, component: str, *, device: str) -> None:
        gravity = RocheSurfaceGravity(component, device=device)
        thermal = GravityDarkeningTemperature(device=device)
        self.jax, self.device = gravity._jax, gravity.device
        import jax.numpy as jnp  # noqa: PLC0415 - optional device dependency

        barrier = self.jax.lax.optimization_barrier

        def evaluate(points: Any, areas: Any, mirror: Any, signs: Any, parameters: Any) -> dict:
            q, spin, mass, axis, teff, beta = (parameters[i:i+1] for i in range(6))
            # Connectivity selects existing point-gravity entries only. Its
            # corner averages are discarded; no degenerate geometric area or
            # normal is ever computed from these repeated selector indices.
            indices = jnp.repeat(jnp.arange(len(points))[:, None], 3, axis=1)
            g = barrier(gravity._evaluate(points[None], indices, q, spin, mass, axis))
            field = g["outward_gradients"][0]
            normal = barrier(field/jnp.linalg.norm(field, axis=1)[:, None])
            force = barrier(g["vertex_gravity"][0, mirror])
            a = barrier(areas[mirror])
            intrinsic = barrier(thermal._temperatures(force[None], a[None], teff, beta))
            t = intrinsic["temperatures"][0]
            coefficient = (jnp.sum(a)*teff[0]**4/jnp.sum(a*t**4))**.25
            t = barrier(t*coefficient)
            valid = g["proposal_valid"][0] & intrinsic["proposal_valid"][0]
            valid &= jnp.all(jnp.isfinite(t) & (t > 0))
            return {"face_centres": points[mirror]*signs, "normals": normal[mirror]*signs,
                    "areas": a, "potential_gradient_magnitudes": force, "log_g": jnp.log10(force),
                    "temperatures": t, "proposal_valid": valid,
                    "intrinsic_luminosity_ratio": jnp.sum(a*(t/teff[0])**4)/jnp.sum(a)}

        self.evaluate = self.jax.jit(evaluate)

    def __call__(self, binary: Any, star: Any, cell: dict) -> dict:
        if not self.jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        points, area = cell["points"], cell["areas"]
        base = len(area)
        mirror = np.asarray(star.face_symmetry_vector, dtype=np.int64)
        if not np.array_equal(mirror, np.tile(np.arange(base), 4)):
            msg = "Curvature fields require original quarter-face ancestry"
            raise UnsupportedCurvedGeometryError(msg)
        GravityDarkeningTemperature.validate_parameters([star.t_eff], [star.gravity_darkening], 1)
        parameters = np.array([binary.mass_ratio, star.synchronicity, binary.primary.mass,
                               binary.semi_major_axis, star.t_eff, star.gravity_darkening])
        if not np.isfinite(parameters).all() or np.any(parameters[2:5] <= 0):
            msg = "Curvature field physical scaling is nonfinite/nonpositive"
            raise UnsupportedCurvedGeometryError(msg)
        values = (points, area, mirror, np.repeat(_SIGNS, base, axis=0), parameters)
        return self.evaluate(*(self.jax.device_put(a, self.device, may_alias=False) for a in values))


def _cpu_state(binary: Any, star: Any, name: str, cell: dict) -> dict:
    """Unchanged original CPU gradient and thermal functions on new geometry."""
    GravityDarkeningTemperature.validate_parameters([star.t_eff], [star.gravity_darkening], 1)
    points, areas = cell["points"], cell["areas"]
    gradient = calculate_potential_gradient(1., name, points, star.synchronicity, binary.mass_ratio)
    norm = np.linalg.norm(gradient, axis=1)
    base_gravity = norm*(const.G*binary.primary.mass/binary.semi_major_axis**2)
    star.face_centres = np.concatenate([points*sign for sign in _SIGNS])
    star.normals = np.concatenate([gradient/norm[:, None]*sign for sign in _SIGNS])
    star.areas = star.mirror_face_values(areas)
    star.potential_gradient_magnitudes = star.mirror_face_values(base_gravity)
    star.log_g = np.log10(star.potential_gradient_magnitudes)
    star.temperatures = calculate_effective_temperatures(star, base_gravity)
    renormalize_temperatures(star)
    if (not np.isfinite(star.temperatures).all() or np.any(star.temperatures <= 0)
            or not np.isfinite(base_gravity).all() or np.any(base_gravity <= 0)):
        msg = "Curvature CPU gravity/thermal state unsupported; abort protocol"
        raise UnsupportedCurvedGeometryError(msg)
    return {"intrinsic_luminosity_ratio": float(np.sum(star.areas*(star.temperatures/star.t_eff)**4)
                                               /np.sum(star.areas))}


class _CurvatureEngine:
    def __init__(self, passbands: list[str], spec: CurvatureFieldSpec, *, device: str, radiance: str) -> None:
        self.spec, self.device, self.radiance = spec, device, radiance
        self.keep_states = False
        self.last_diagnostics = {}
        self.gravity = {} if device == "reference" else {n: _PointState(n, device=device) for n in _NAMES}
        self.helper = None if device == "reference" else HybridBinaryLightCurve(
            passbands, device=device, reflection="tiled", coverage="cpu", fallback="raise", radiance=radiance,
        )
        self.reflection = None if device == "reference" else TiledDeviceReflection(
            settings.LIMB_DARKENING_LAW, device=device,
        )

    def _fields(self, binary: Any, observer: Any) -> tuple[Any, dict]:
        system, cells, controls = curvature_geometry(binary, self.spec.level)
        states, intrinsic = {}, {}
        for name in _NAMES:
            star = getattr(system, name)
            star.polar_potential_gradient_magnitude = calculate_polar_potential_gradient_magnitude(
                1., binary.mass_ratio, star.polar_radius, name, star.synchronicity,
            )
            if self.device == "reference":
                detail = _cpu_state(binary, star, name, cells[name])
            else:
                state = self.gravity[name](binary, star, cells[name])
                if not bool(state["proposal_valid"]):
                    msg = "Curvature device gravity/thermal state unsupported; abort protocol"
                    raise UnsupportedCurvedGeometryError(msg)
                states[name] = state
                for key in ("face_centres", "normals", "areas", "potential_gradient_magnitudes", "log_g",
                            "temperatures"):
                    setattr(star, key, np.array(state[key], copy=True))
                detail = {"intrinsic_luminosity_ratio": float(state["intrinsic_luminosity_ratio"])}
            intrinsic[name] = {**controls[name], **detail,
                               "intrinsic_temperatures": star.temperatures.copy() if self.keep_states else None}
        if self.device == "reference":
            system = cpu_reflect(system, settings.REFLECTION_EFFECT_ITERATIONS)
        elif settings.REFLECTION_EFFECT and settings.REFLECTION_EFFECT_ITERATIONS > 0:
            result = self.reflection.prepare(system, settings.REFLECTION_EFFECT_ITERATIONS,
                                             device_states=states).evaluate()
            if not bool(result["proposal_valid"]):
                msg = "Curvature device reflection unsupported; abort protocol"
                raise UnsupportedCurvedGeometryError(msg)
            for name, temperatures in zip(_NAMES, result["temperatures"], strict=True):
                states[name]["temperatures"] = temperatures
                getattr(system, name).temperatures = np.array(temperatures, copy=True)
        system.flat_it()
        if self.device != "reference" and self.radiance == "device":
            self.helper._prepare_radiance(system, observer, {n: states[n]["temperatures"] for n in _NAMES},
                                         diagnostics=True)
        else:
            prep_surface_params(system, return_values=False, write_to_containers=True,
                                passband=observer.passband, left_bandwidth=observer.left_bandwidth,
                                right_bandwidth=observer.right_bandwidth)
        self.last_diagnostics = intrinsic
        return system, {name: cells[name]["triangles"] for name in _NAMES}


class CurvatureFieldLightCurve(SharedFieldLightCurve):
    """Complete curved-cell field batches on a common fresh clipped quadrature.

    Cell fields are finite collocation values, constant per angular parent.
    Original CPU formulas are the independent implementation oracle, applied
    to this new geometry. No original faceted whole-curve equivalence, continuum
    field solution, reflected energy closure or fit/MCMC qualification follows.
    """

    def __init__(self, passbands: list[str], rule: BoundaryRule, *, levels: tuple[int, ...] = (1, 2),
                 integration_level: int = 2, device: str = "gpu", radiance: str = "device") -> None:
        if (not isinstance(levels, tuple) or not 1 <= len(levels) <= 4 or len(set(levels)) != len(levels)
                or not isinstance(rule, BoundaryRule) or device not in ("reference", "cpu", "gpu")
                or radiance not in ("cpu", "device") or not passbands or len(passbands) > 8
                or len(set(passbands)) != len(passbands)):
            msg = "Supply1-4unique field levels, BoundaryRule,1-8bands and explicit backend/radiance"
            raise ValueError(msg)
        variants = tuple(CurvatureFieldSpec(v) for v in levels)
        if (isinstance(integration_level, bool) or not isinstance(integration_level, int)
                or not max(levels) <= integration_level <= 3):
            msg = "Curvature common level must be integer max(field levels)..3"
            raise ValueError(msg)
        self.passbands, self.rule, self.variants = tuple(passbands), rule, variants
        self.integration_level, self.device, self.radiance = integration_level, device, radiance
        self.law = settings.LIMB_DARKENING_LAW
        self.engines = {v.key: _CurvatureEngine(passbands, v, device=device, radiance=radiance) for v in variants}
        self.integrator = None if device == "reference" else SurfaceFluxIntegrator(self.law, device=device)

    def __call__(self, binary: Any, phases: Any, *, diagnostics: bool = False, **kwargs: Any) -> SharedFieldResult:
        started = perf_counter()
        for engine in self.engines.values():
            engine.keep_states = diagnostics
        result = super().__call__(binary, phases, diagnostics=diagnostics, **kwargs)
        detail = {**result.diagnostics, "field_geometry": {k: e.last_diagnostics for k, e in self.engines.items()},
                  "contract": "Curved Roche cell centres/normals/areas, collocated gravity and area-normalized "
                  "intrinsic temperatures; original reflection/LD/radiance formulas on new finite geometry. "
                  "Shared CPU quadrature, no scalar correction, no continuum/fit certificate."}
        curves = {k: replace(v, rule={**v.rule, "curvature_fields": 1}) for k, v in result.curves.items()}
        return replace(result, curves=curves, diagnostics=detail,
                       timings={**result.timings, "total": perf_counter()-started})
