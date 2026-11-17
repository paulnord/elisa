"""Complete finite field/correction comparisons on identical Roche quadrature.

Nested angular ancestry is checked geometrically, then all field variants use
the same fresh clipped packets. Original CPU/default synthesis is untouched.
This isolates computed finite field changes, not continuous-field error.
"""

# ruff: noqa: SLF001, PLR2004 - owned experimental composition and explicit domain bounds
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from elisa import Observer, const, settings, utils
from elisa.base.curves.utils import calculate_surface_element_fluxes
from elisa.binary_system import dynamic
from elisa.binary_system.curves.utils import prep_surface_params
from elisa.experimental.boundary_geometry import BoundaryRule, ConditionedRocheGeometry, boundary_packet
from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError, duffy_nodes
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_boundary import BoundaryBinaryLightCurve
from elisa.experimental.gpu_curved import CurvedCurveResult, _CaptureMesh
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_radiance import passband_revision
from elisa.experimental.gpu_reference import cpu_reflect, intrinsic
from elisa.experimental.gpu_selective_mesh import seed_system
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator

_NAMES = ("primary", "secondary")


@dataclass(frozen=True)
class FieldVariant:
    """Fixed field grid and original scalar correction angle, not a prior."""

    level: int
    correction_level: int

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 2
               for v in (self.level, self.correction_level)):
            msg = "Shared field/correction levels must be integer0-2"
            raise ValueError(msg)

    @property
    def key(self) -> str:
        return f"field{self.level}_correction{self.correction_level}"


def shared_parent_map(fine: Any, coarse: Any, levels: int) -> tuple[np.ndarray, dict[str, float]]:
    """Check nested angular cones with a conditioning-scaled FP64 solve guard.

    Repeated original edge projection changes midpoint placement but leaves it
    on its parent great-circle edge. Each fine triangle therefore lies inside
    its ancestor's positive cone. Seam roundoff is included in the solve guard;
    arbitrary nearest-parent interpolation or index reassignment is forbidden.
    """
    fine, coarse = np.asarray(fine), np.asarray(coarse)
    if (isinstance(levels, bool) or not isinstance(levels, int) or not 0 <= levels <= 3
            or fine.ndim != 3 or coarse.ndim != 3 or fine.shape[1:] != (3, 3) or coarse.shape[1:] != (3, 3)
            or len(fine) != len(coarse)*4**levels or not len(coarse)
            or not np.isfinite(fine).all() or not np.isfinite(coarse).all()):
        msg = "Shared quadrature requires compatible finite nested triangle arrays"
        raise UnsupportedCurvedGeometryError(msg)
    parent = np.arange(len(fine), dtype=np.int64)//4**levels
    matrix = np.swapaxes(coarse, 1, 2)
    condition = np.linalg.cond(matrix)
    if np.any(condition > 1e5):
        msg = "Shared angular ancestry is too ill-conditioned"
        raise UnsupportedCurvedGeometryError(msg)
    coefficients = np.swapaxes(np.linalg.solve(matrix[parent], np.swapaxes(fine, 1, 2)), 1, 2)
    eps = np.finfo(float).eps
    # A normwise backward/forward check is essential on exact seam axes. A
    # componentwise inverse-product estimate can spuriously give a zero guard
    # where the input coordinate is zero, while other coordinates carry solve
    # roundoff. The factor64 covers the small3x3 operations; condition is capped.
    cscale = np.maximum(1., np.abs(coefficients).max(axis=2))
    guard = (64*eps*condition[parent, None]*cscale)[:, :, None]
    residual = np.einsum("nij,nqj->nqi", matrix[parent], coefficients)-fine
    matrix_norm = np.linalg.norm(matrix[parent], ord=np.inf, axis=(1, 2))
    residual_guard = (64*eps*(matrix_norm[:, None]*cscale+np.abs(fine).max(axis=2)))[:, :, None]
    if (np.any(coefficients < -guard) or np.any(np.abs(residual) > residual_guard)
            or np.any(coefficients.sum(axis=2) <= 0)):
        msg = "Shared angular child escaped its declared parent cone; abort protocol"
        raise UnsupportedCurvedGeometryError(msg)
    parent.flags.writeable = False
    return parent, {"maximum_condition": float(condition.max()),
                    "minimum_coefficient": float(coefficients.min()), "maximum_roundoff_guard": float(guard.max()),
                    "maximum_reconstruction_residual": float(np.abs(residual).max())}


class _VariantMesh(_CaptureMesh):
    def __init__(self, variant: FieldVariant) -> None:
        super().__init__(variant.level)
        self.variant = variant

    def apply(self, system: Any) -> Any:
        info = super().apply(system)
        for name in _NAMES:
            # Only the original scalar correction's argument changes. Raw
            # topology/points, cached pre-correction gravity/normals and all
            # later original thermal/reflection/radiance sequencing remain.
            getattr(system, name).discretization_factor = np.radians(info[name]["seed_degrees"])/2**(
                self.variant.correction_level
            )
        return info


class _VariantEngine(BoundaryBinaryLightCurve):
    def _fields(self, binary: Any, observer: Any) -> tuple[Any, Any]:
        capture = _VariantMesh(self.variant)
        if self.device == "reference":
            system = cpu_reflect(intrinsic(binary, capture), settings.REFLECTION_EFFECT_ITERATIONS)
            system.flat_it()
        else:
            self.field.capture = capture
            system, _, temperatures, _ = self.field._build_intrinsic_system(binary)
            if self.radiance == "device":
                self.field._prepare_radiance(system, observer, temperatures, diagnostics=True)
                return system, capture.triangles
        prep_surface_params(system, return_values=False, write_to_containers=True,
                            passband=observer.passband, left_bandwidth=observer.left_bandwidth,
                            right_bandwidth=observer.right_bandwidth)
        return system, capture.triangles


@dataclass(frozen=True)
class SharedFieldResult:
    """All host curves plus common geometry provenance and complete batch cost."""

    curves: dict[str, CurvedCurveResult]
    quadrature_sha256: str
    timings: dict[str, float]
    diagnostics: Any


class SharedFieldLightCurve:
    """Fresh finite fields/correction variants on one checked angular partition.

    This returns a comparison batch, not one ordinary fitting proposal. CPU
    reference and cpu/gpu kernels share the fresh CPU geometry bridge. No
    physical prediction or quadrature packet is cached across candidates.
    """

    def __init__(self, passbands: list[str], rule: BoundaryRule, variants: tuple[FieldVariant, ...], *,
                 integration_level: int = 2, device: str = "gpu", radiance: str = "device") -> None:
        if (not isinstance(variants, tuple) or not 1 <= len(variants) <= 4
                or any(not isinstance(v, FieldVariant) for v in variants) or len(set(variants)) != len(variants)
                or isinstance(integration_level, bool) or not isinstance(integration_level, int)
                or not max(v.level for v in variants) <= integration_level <= 3):
            msg = "Supply1-4unique variants and integer integration level max(field level)..3"
            raise ValueError(msg)
        self.passbands, self.rule, self.variants = tuple(passbands), rule, variants
        self.integration_level, self.device, self.radiance = integration_level, device, radiance
        self.engines = {}
        for v in variants:
            engine = _VariantEngine(passbands, rule, field_levels=v.level, device=device, radiance=radiance)
            engine.variant = v
            self.engines[v.key] = engine
        self.law = settings.LIMB_DARKENING_LAW
        self.integrator = None if device == "reference" else SurfaceFluxIntegrator(self.law, device=device)

    def _prepare(self, model: Any, observer: Any) -> tuple:
        common = seed_system(model)
        NestedMesh(self.integration_level).apply(common)
        triangles, fields, maps, controls = {}, {}, {}, {}
        for name, x in zip(_NAMES, (0., 1.), strict=True):
            star = getattr(common, name)
            d = star.points-np.array([x, 0, 0])
            d /= np.linalg.norm(d, axis=1)[:, None]
            triangles[name] = d[star.faces]
        for variant in self.variants:
            system, cells = self.engines[variant.key]._fields(model, observer)
            fields[variant.key], maps[variant.key], controls[variant.key] = system, {}, {}
            for name in _NAMES:
                parent, control = shared_parent_map(triangles[name], cells[name],
                                                    self.integration_level-variant.level)
                maps[variant.key][name], controls[variant.key][name] = parent, control
        return common, triangles, fields, maps, controls

    def __call__(self, binary: Any, phases: Any, *, source_revision: str, observer: Any = None,  # noqa: C901, PLR0915
                 normalize: bool = True, diagnostics: bool = False) -> SharedFieldResult:
        """Synthesize complete comparison curves with original observer semantics."""
        started = perf_counter()
        phases = np.array(phases, dtype=float, copy=True)
        if (phases.ndim != 1 or not 1 <= len(phases) <= 128 or not np.isfinite(phases).all()
                or not source_revision or settings.NUMBER_OF_PROCESSES != 1 or self.law != settings.LIMB_DARKENING_LAW):
            msg = "Shared curves require1-128finite phases, fixed sources/law and serial execution"
            raise UnsupportedCurvedGeometryError(msg)
        model = deepcopy(binary)
        supplied = Observer(passband=list(self.passbands), system=model) if observer is None else deepcopy(observer)
        if tuple(supplied.passband) != self.passbands:
            msg = "Shared observer passbands/order must match"
            raise ValueError(msg)
        supplied._system = model
        geometry = {n: ConditionedRocheGeometry(model, n) for n in _NAMES}
        common, triangles, fields, maps, controls = self._prepare(model, supplied)
        fields_done = perf_counter()
        signature = hashlib.sha256(json.dumps({
            "binary": model.to_json(), "passbands": passband_revision(supplied.passband), "source": source_revision,
            "law": self.law, "reflection": (settings.REFLECTION_EFFECT, settings.REFLECTION_EFFECT_ITERATIONS),
            "single_ld": settings.USE_SINGLE_LD_COEFFICIENTS, "preserve_phases": settings.PRESERVE_INPUT_PHASES,
        }, sort_keys=True, default=str).encode()).hexdigest()
        resolved = supplied.manage_time_series(phases=phases)
        base, origin = supplied.phase_interval_reduce(resolved)
        support, reverse = dynamic.phase_crv_symmetry(common, base)
        motion = model.get_positions_method()(input_argument=support, return_nparray=False, calculate_from="phase")
        components = {v.key: {b: np.zeros((len(motion), 2)) for b in self.passbands} for v in self.variants}
        records, digest = [], hashlib.sha256()
        digest.update(json.dumps({"level": self.integration_level, "rule": asdict(self.rule)}).encode())
        geometry_seconds = integration_seconds = 0.
        for t, position in enumerate(motion):
            step = perf_counter()
            line = utils.rotate_item(np.eye(3), position, model.inclination)@const.LINE_OF_SIGHT
            if abs(np.linalg.norm(line)-1) > 64*np.finfo(float).eps:
                msg = "Original observer transform failed its unit LOS check"
                raise UnsupportedCurvedGeometryError(msg)
            overlap = np.sqrt(max(0., 1-line[0]**2)) < .5
            foreground_name = "secondary" if line[0] > 0 else "primary"
            packets, record = {}, {"phase": float(position.phase), "components": {}}
            for name in _NAMES:
                foreground = geometry[foreground_name] if overlap and name != foreground_name else None
                p = boundary_packet(triangles[name], geometry[name], line, self.rule, foreground)
                digest.update(json.dumps({"phase": float(position.phase), "component": name,
                                          "nodes": len(p["mu"])}).encode())
                for key in ("mu", "area", "parents"):
                    digest.update(np.ascontiguousarray(p[key]).tobytes())
                packets[name] = p
                record["components"][name] = {k: v for k, v in p.items() if k not in ("mu", "area", "parents")}
            geometry_seconds += perf_counter()-step
            step = perf_counter()
            for variant in self.variants:
                stars = []
                for name in _NAMES:
                    packet = packets[name]
                    index = maps[variant.key][name][packet["parents"]]
                    star = getattr(fields[variant.key], name)
                    stars.append(SimpleNamespace(
                        indices=np.arange(len(index)), los_cosines=packet["mu"],
                        coverage=packet["area"]*model.semi_major_axis**2,
                        ld_cfs={b: star.ld_cfs[b][index] for b in self.passbands},
                        normal_radiance={b: star.normal_radiance[b][index] for b in self.passbands},
                    ))
                width = max(len(s.indices) for s in stars)
                pad = 1 << max(0, width-1).bit_length()
                for band in self.passbands:
                    if self.device == "reference":
                        values = [np.sum(calculate_surface_element_fluxes(band, s)) for s in stars]
                    else:
                        _, values = self.integrator.from_containers(band, stars, pad_to=pad)
                    components[variant.key][band][t] = np.asarray(values)
            integration_seconds += perf_counter()-step
            records.append(record)
        timings = {"all_fresh_fields_and_common_partition": fields_done-started,
                   "cpu_common_geometry_clipping_hash": geometry_seconds,
                   "all_variant_integration_transfers_compile": integration_seconds,
                   "total": perf_counter()-started}
        results = {}
        for variant in self.variants:
            mapped = {b: v[reverse][origin] for b, v in components[variant.key].items()}
            results[variant.key] = CurvedCurveResult(
                resolved+model.phase_shift, HybridBinaryLightCurve._observed_fluxes(mapped, model, normalize=normalize),
                HybridBinaryLightCurve._observed_fluxes(mapped, model, normalize=False), mapped, signature,
                {**asdict(self.rule), "integration_level": self.integration_level,
                 "correction_level": variant.correction_level}, variant.level, float(model.additional_light), {},
                {"quadrature_sha256": digest.hexdigest(), "field_faces": {
                    n: len(getattr(fields[variant.key], n).faces) for n in _NAMES}},
            )
        timings["total"] = perf_counter()-started
        return SharedFieldResult(results, digest.hexdigest(), timings, {
            "phase_records": records, "ancestry_controls": controls,
            "initial_systems": fields if diagnostics else None,
            "contract": "Identical fresh clipped packets for all variants; original finite fields/correction. "
            "Common CPU geometry bridge, no continuum certificate or ordinary-proposal speedup claim.",
        })


def shared_area_moments(binary: Any, integration_level: int, variants: tuple[FieldVariant, ...]) -> dict[str, Any]:
    """CPU-only intrinsic T4 moments on shared area, without renormalizing temperatures."""
    common = seed_system(binary)
    NestedMesh(integration_level).apply(common)
    result = {}
    for variant in variants:
        mesh = _VariantMesh(variant)
        system = intrinsic(binary, mesh)
        result[variant.key] = {}
        for name, x in zip(_NAMES, (0., 1.), strict=True):
            star, grid = getattr(system, name), getattr(common, name)
            d = grid.points-np.array([x, 0, 0])
            d /= np.linalg.norm(d, axis=1)[:, None]
            parent, _ = shared_parent_map(d[grid.faces], mesh.triangles[name], integration_level-variant.level)
            v, w = duffy_nodes(d[grid.faces], 4)
            _, _, jacobian = ConditionedRocheGeometry(binary, name).sample(v)
            weights = (w*jacobian).reshape((-1, 16)).sum(axis=1)
            area = np.bincount(parent, weights=weights, minlength=len(star.faces))
            original = np.sum(star.temperatures**4*star.areas)/(star.t_eff**4*star.areas.sum())-1
            curved = np.sum(star.temperatures**4*area)/(star.t_eff**4*area.sum())-1
            result[variant.key][name] = {"original_intrinsic_closure_relative": float(original),
                                       "same_fields_shared_curved_closure_relative": float(curved),
                                       "curved_area_SMA2": float(area.sum()),
                                       "original_corrected_area_SMA2": float(star.areas.sum())}
    return result
