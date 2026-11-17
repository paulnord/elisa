"""Batched fresh device geometry composed into complete reflected curves.

CPU topology, observer geometry/clipping and LD/table bridges remain explicit.
Geometry rows may change every physical parameter and angular ray. Grouping by
shape and bounded proposal waves do not freeze topology or cache predictions.
"""

# ruff: noqa: C901, PLR0912, PLR0915, SLF001, PLR2004 - explicit composition of owned experimental kernels
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np

from elisa import const, settings
from elisa.binary_system.curves.utils import prep_surface_params
from elisa.binary_system.surface.gravity import calculate_polar_potential_gradient_magnitude
from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from elisa.experimental.gpu_batched_geometry import MAX_NODES, BatchedRocheGeometry
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_curved_edges import CurvedEdgeLightCurve, CurvedEdgeRule
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_reflection_tiled import TiledDeviceReflection
from elisa.experimental.gpu_selective_mesh import seed_system
from elisa.experimental.gpu_thermal import GravityDarkeningTemperature

_NAMES = ("primary", "secondary")
_SIGNS = np.array([[1., 1., 1.], [1., -1., 1.], [1., -1., -1.], [1., 1., -1.]])
_STATE_KEYS = ("face_centres", "normals", "areas", "potential_gradient_magnitudes", "log_g", "temperatures")


def _signature(binary: Any) -> str:
    return hashlib.sha256(json.dumps(binary.to_json(), sort_keys=True, default=str).encode()).hexdigest()


def _angular_inputs(binary: Any, level: int) -> tuple:
    # Constructor checks preserve the original physical/numerical domain.
    for name in _NAMES:
        ConditionedRocheGeometry(binary, name)
    system = seed_system(binary)
    NestedMesh(level).apply(system)
    inputs = {}
    for name, x in zip(_NAMES, (0., 1.), strict=True):
        star = getattr(system, name)
        base = int(star.base_symmetry_faces_number)
        mirror = np.asarray(star.face_symmetry_vector)
        if not np.array_equal(mirror, np.tile(np.arange(base), 4)):
            msg = "Resident geometry requires original quarter-face ancestry"
            raise UnsupportedCurvedGeometryError(msg)
        d = star.points-np.array([x, 0., 0.])
        d /= np.linalg.norm(d, axis=1)[:, None]
        triangles = d[star.faces]
        inputs[name] = {"rays": triangles[:base].mean(axis=1), "triangles": triangles,
                        "parameters": np.array([binary.mass_ratio, star.synchronicity, star.surface_potential]),
                        "physics": np.array([binary.primary.mass, binary.semi_major_axis,
                                             star.t_eff, star.gravity_darkening])}
    return system, inputs


class _ResidentField:
    """Pure device root/area/force/thermal composition for one shape group."""

    def __init__(self, component: str, *, device: str) -> None:
        self.geometry = BatchedRocheGeometry(component, device=device)
        self.jax, self.jnp, self.device = self.geometry.jax, self.geometry.jnp, self.geometry.device
        thermal = GravityDarkeningTemperature(device=device)
        jnp, barrier = self.jnp, self.jax.lax.optimization_barrier

        def compose(parameters: Any, rays: Any, triangles: Any, physics: Any) -> dict:
            proposals, base = rays.shape[:2]
            mask = jnp.ones((proposals, base), dtype=bool)
            sampled = self.geometry.sample_kernel(parameters, rays, mask)
            integrated = self.geometry._areas[4](parameters, triangles, mask)
            mirror = jnp.tile(jnp.arange(base), 4)
            signs = jnp.repeat(jnp.asarray(_SIGNS), base, axis=0)
            mass, axis, teff, beta = (physics[:, i] for i in range(4))
            gradient = barrier(sampled["outward_gradients"])
            force = barrier(jnp.linalg.norm(gradient, axis=-1)*(const.G*mass/axis**2)[:, None])[:, mirror]
            areas = barrier(integrated["areas"][:, mirror])
            intrinsic = thermal._temperatures(force, areas, teff, beta)
            t = barrier(intrinsic["temperatures"])
            coefficient = (jnp.sum(areas, axis=1)*teff**4/jnp.sum(areas*t**4, axis=1))**.25
            t = barrier(t*coefficient[:, None])
            valid = sampled["proposal_valid"] & integrated["proposal_valid"] & intrinsic["proposal_valid"]
            valid &= jnp.all(jnp.isfinite(t) & (t > 0), axis=1)
            return {"face_centres": barrier(sampled["points"][:, mirror]*signs),
                    "normals": barrier(sampled["normals"][:, mirror]*signs),
                    "areas": areas, "potential_gradient_magnitudes": force, "log_g": jnp.log10(force),
                    "temperatures": t, "proposal_valid": valid,
                    "intrinsic_luminosity_ratio": jnp.sum(areas*(t/teff[:, None])**4, axis=1)/jnp.sum(areas, axis=1),
                    "curved_area_SMA2": jnp.sum(areas, axis=1),
                    "max_collocation_potential_relative_residual": jnp.max(
                        jnp.abs(sampled["potential_residual"])/parameters[:, 2:3], axis=1)}
        self.kernel = self.jax.jit(compose)

    def evaluate(self, rows: list[dict]) -> dict:
        if not self.jax.config.x64_enabled:
            msg = "JAX FP64 must remain enabled"
            raise RuntimeError(msg)
        n = len(rows[0]["rays"])
        if len(rows)*n*16 > MAX_NODES:
            msg = "Resident geometry quadrature cap exceeded before allocation"
            raise UnsupportedCurvedGeometryError(msg)
        parameters = np.stack([r["parameters"] for r in rows])
        rays = np.stack([r["rays"] for r in rows])
        triangles = np.stack([r["triangles"][:n] for r in rows])
        physics = np.stack([r["physics"] for r in rows])
        GravityDarkeningTemperature.validate_parameters(physics[:, 2], physics[:, 3], len(rows))
        if not np.isfinite(physics).all() or np.any(physics[:, :2] <= 0):
            msg = "Resident field mass/axis scaling must be finite and positive"
            raise UnsupportedCurvedGeometryError(msg)
        placed = [self.jax.device_put(a, self.device, may_alias=False) for a in (parameters, rays, triangles, physics)]
        result = self.kernel(*placed)
        if not np.all(np.asarray(result["proposal_valid"])):
            msg = "Unsupported resident geometry/thermal batch; abort outside priors"
            raise UnsupportedCurvedGeometryError(msg)
        return result


class _ResidentEngine:
    def __init__(self, passbands: list[str], *, device: str, radiance: str) -> None:
        self.device, self.radiance = device, radiance
        self.keep_states, self.last_diagnostics, self.prepared = False, {}, None
        self.helper = HybridBinaryLightCurve(passbands, device=device, reflection="tiled", coverage="cpu",
                                             fallback="raise", radiance=radiance)
        self.reflection = TiledDeviceReflection(settings.LIMB_DARKENING_LAW, device=device)

    def _fields(self, binary: Any, observer: Any) -> tuple:
        if self.prepared is None:
            msg = "Resident fields require this candidate's fresh batch transaction"
            raise RuntimeError(msg)
        signature, system, inputs, states = self.prepared
        self.prepared = None
        if _signature(binary) != signature:
            msg = "Resident batch candidate identity changed; no stale physical state allowed"
            raise RuntimeError(msg)
        intrinsic = {}
        for name in _NAMES:
            star, state = getattr(system, name), states[name]
            star.polar_potential_gradient_magnitude = calculate_polar_potential_gradient_magnitude(
                1., binary.mass_ratio, star.polar_radius, name, star.synchronicity,
            )
            for key in _STATE_KEYS:
                setattr(star, key, np.array(state[key], copy=True))
            intrinsic[name] = {"faces": len(star.faces), "area_order": 4, "scalar_correction": False,
                               **{k: float(state[k]) for k in ("curved_area_SMA2", "intrinsic_luminosity_ratio",
                                  "max_collocation_potential_relative_residual")},
                               "intrinsic_temperatures": star.temperatures.copy() if self.keep_states else None,
                               "geometry_equation": "device Cartesian original Roche potential"}
        if settings.REFLECTION_EFFECT and settings.REFLECTION_EFFECT_ITERATIONS > 0:
            result = self.reflection.prepare(system, settings.REFLECTION_EFFECT_ITERATIONS,
                                             device_states=states).evaluate()
            if not bool(result["proposal_valid"]):
                msg = "Resident device reflection unsupported; abort outside priors"
                raise UnsupportedCurvedGeometryError(msg)
            for name, t in zip(_NAMES, result["temperatures"], strict=True):
                states[name]["temperatures"] = t
                getattr(system, name).temperatures = np.array(t, copy=True)
        system.flat_it()
        if self.radiance == "device":
            self.helper._prepare_radiance(system, observer, {n: states[n]["temperatures"] for n in _NAMES},
                                         diagnostics=True)
        else:
            prep_surface_params(system, return_values=False, write_to_containers=True,
                                passband=observer.passband, left_bandwidth=observer.left_bandwidth,
                                right_bandwidth=observer.right_bandwidth)
        self.last_diagnostics = intrinsic
        return system, {n: inputs[n]["triangles"] for n in _NAMES}


@dataclass(frozen=True)
class ResidentCurveBatch:
    """Host comparison curves and complete batch cost; row timers are stages."""

    results: tuple
    timings: dict
    geometry_groups: tuple


class ResidentGeometryLightCurve(CurvedEdgeLightCurve):
    """Fresh changing-physics proposal batches with resident field geometry.

    The independent reference is the preceding original-CPU finite field path.
    Only field geometry/area/thermal preparation is batched across candidates.
    Reflection/radiance, observer geometry/clipping and flux execution remain
    candidate-specific. Full batch timing includes those CPU bridges and all
    host outputs. Neither fitting accuracy nor MCMC/ESS is certified here.
    """

    def __init__(self, passbands: list[str], rule: CurvedEdgeRule, *, max_resident_faces: int = 524288,
                 **kwargs: Any) -> None:
        super().__init__(passbands, rule, **kwargs)
        if (isinstance(max_resident_faces, bool) or not isinstance(max_resident_faces, int)
                or not 1 <= max_resident_faces <= MAX_NODES):
            msg = "Declare positive resident face budget<=2097152 before evaluation"
            raise ValueError(msg)
        self.max_resident_faces, self._batch_active = max_resident_faces, False
        if self.device != "reference":
            self.samplers = {n: _ResidentField(n, device=self.device) for n in _NAMES}
            self.engines = {v.key: _ResidentEngine(list(self.passbands), device=self.device, radiance=self.radiance)
                            for v in self.variants}

    def __call__(self, binary: Any, phases: Any, **kwargs: Any) -> Any:
        result = self.batch([binary], [phases], **kwargs)
        return replace(result.results[0], timings={**result.results[0].timings,
                       "resident_field_preparation": result.timings["field_preparation"],
                       "total": result.timings["total"]})

    def batch(self, binaries: list, phases: list, *, source_revision: str, observers: list | None = None,
              observer: Any = None, normalize: bool = True, diagnostics: bool = False) -> ResidentCurveBatch:
        """Process all candidates in bounded waves; never cache across calls."""
        started = perf_counter()
        if self._batch_active or not 1 <= len(binaries) <= 128 or len(phases) != len(binaries):
            msg = "Use a non-reentrant batch of1-128candidates and matching phase arrays"
            raise ValueError(msg)
        if observers is not None and (observer is not None or len(observers) != len(binaries)):
            msg = "Supply matching observers or one shared observer, not both"
            raise ValueError(msg)
        observers = [observer]*len(binaries) if observers is None else observers
        self._batch_active = True
        results, groups, prep_seconds = [], [], 0.
        try:
            if self.device == "reference":
                results = [super(ResidentGeometryLightCurve, self).__call__(
                    b, p, source_revision=source_revision, observer=o, normalize=normalize, diagnostics=diagnostics,
                ) for b, p, o in zip(binaries, phases, observers, strict=True)]
            else:
                prep = perf_counter()
                plans, sizes = [], []
                for b in binaries:
                    row = {v.key: _angular_inputs(b, v.level) for v in self.variants}
                    sizes.append(sum(len(getattr(system, n).faces) for system, _ in row.values() for n in _NAMES))
                    if sizes[-1] > self.max_resident_faces:
                        msg = "One candidate exceeds the resident face budget; request an explicit larger budget"
                        raise UnsupportedCurvedGeometryError(msg)
                    plans.append(row)
                prep_seconds += perf_counter()-prep
                start = 0
                while start < len(binaries):
                    end, total_faces = start, 0
                    while end < len(binaries) and total_faces+sizes[end] <= self.max_resident_faces:
                        total_faces += sizes[end]
                        end += 1
                    prep = perf_counter()
                    states = {i: {v.key: {} for v in self.variants} for i in range(start, end)}
                    for variant in self.variants:
                        for name in _NAMES:
                            by_shape = {}
                            for i in range(start, end):
                                inputs = plans[i][variant.key][1][name]
                                by_shape.setdefault(len(inputs["rays"]), []).append(i)
                            for nodes, indices in by_shape.items():
                                chunk = MAX_NODES//(nodes*16)
                                for k in range(0, len(indices), chunk):
                                    ids = indices[k:k+chunk]
                                    rows = [plans[i][variant.key][1][name] for i in ids]
                                    evaluated = self.samplers[name].evaluate(rows)
                                    for j, i in enumerate(ids):
                                        states[i][variant.key][name] = {key: a[j] for key, a in evaluated.items()}
                                    groups.append({"component": name, "level": variant.level,
                                                   "proposals": ids, "quarter_faces": nodes,
                                                   "quadrature_nodes": len(ids)*nodes*16})
                    prep_seconds += perf_counter()-prep
                    for i in range(start, end):
                        for variant in self.variants:
                            system, inputs = plans[i][variant.key]
                            self.engines[variant.key].prepared = (_signature(binaries[i]), system, inputs,
                                                                 states[i][variant.key])
                        result = super().__call__(binaries[i], phases[i], source_revision=source_revision,
                                                  observer=observers[i], normalize=normalize, diagnostics=diagnostics)
                        if any(e.prepared is not None for e in self.engines.values()):
                            msg = "Resident batch did not consume every fresh field level"
                            raise RuntimeError(msg)
                        results.append(result)
                    start = end
            return ResidentCurveBatch(tuple(results), {"field_preparation": prep_seconds,
                                      "total": perf_counter()-started,
                                      "row_timers_exclude_shared_preparation": self.device != "reference"},
                                      tuple(groups))
        finally:
            self._batch_active = False
            if self.device != "reference":
                for engine in self.engines.values():
                    engine.prepared = None
