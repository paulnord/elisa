"""Complete curved Roche integration of fresh finite original ELISa fields.

This opt-in numerical model keeps original thermal/reflection/radiance fields
piecewise constant on angular parent faces. It replaces faceted integration and
scalar-corrected eclipse geometry by true Roche quadrature and implicit shadows.
CPU roots, adaptive partition/masks and original finite field bridges remain
explicit costs. CPU/default Observer and fitting behavior is untouched.
"""

# ruff: noqa: SLF001 - owned composition of experimental field kernels
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
from elisa.experimental.curved_geometry import CurvedRule, RocheGeometry, UnsupportedCurvedGeometryError, curved_packet
from elisa.experimental.gpu_accuracy import resolution_error
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_radiance import passband_revision
from elisa.experimental.gpu_reference import cpu_reflect, intrinsic
from elisa.experimental.gpu_surface import SurfaceFluxIntegrator

_NAMES = ("primary", "secondary")
_MAX_FIELDS, _MAX_PHASES, _MAX_FIELD_LEVEL, _MAX_BANDS = 20276, 128, 2, 8


class _CaptureMesh:
    def __init__(self, levels: int) -> None:
        self.mesh = NestedMesh(levels)
        self.triangles = {}

    def apply(self, system: Any) -> Any:
        info = self.mesh.apply(system)
        for name, x in [("primary", 0.), ("secondary", 1.)]:
            star = getattr(system, name)
            if len(star.faces) > _MAX_FIELDS:
                msg = "Curved field grid exceeds20276 parent faces per component"
                raise UnsupportedCurvedGeometryError(msg)
            rays = star.points-np.array([x, 0, 0])
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            self.triangles[name] = np.array(rays[star.faces], copy=True)
        return info


class _FieldHybrid(HybridBinaryLightCurve):
    def _initial_mesh(self, system: Any) -> None:
        super()._initial_mesh(system)
        self.capture.apply(system)


@dataclass(frozen=True)
class CurvedCurveResult:
    """Complete host curves and finite numerical provenance, with explicit costs."""

    phases: np.ndarray
    fluxes: dict[str, np.ndarray]
    raw_fluxes: dict[str, np.ndarray]
    component_fluxes: dict[str, np.ndarray]
    signature: str
    rule: dict[str, int]
    field_levels: int
    additional_light: float
    timings: dict[str, float]
    diagnostics: Any


class CurvedBinaryLightCurve:
    """Standalone finite curved quadrature; no automatic coarsening or prior cuts.

    device='reference' uses independent original CPU field/reflection/radiance
    primitives and original per-element flux arithmetic. cpu/gpu use the existing
    explicit JAX field/integration kernels. True geometry, adaptive partition and
    eclipse masks are a common fresh CPU bridge, checked independently in tests.
    Numerical boundary indicators are empirical; depth/order/field convergence
    gates remain separate from implementation parity. No inference backend is
    registered and no continuum accuracy is implied by a finite pair passing.
    """

    def __init__(self, passbands: list[str], rule: CurvedRule, *, field_levels: int = 1,
                 device: str = "gpu", radiance: str = "device") -> None:
        if not isinstance(rule, CurvedRule) or device not in ("reference", "cpu", "gpu"):
            msg = "Supply a CurvedRule and explicit reference/cpu/gpu backend"
            raise ValueError(msg)
        if (isinstance(field_levels, bool) or not isinstance(field_levels, int)
                or not 0 <= field_levels <= _MAX_FIELD_LEVEL):
            msg = "Curved field levels must be integer0-2"
            raise ValueError(msg)
        if (not passbands or len(passbands) > _MAX_BANDS or len(set(passbands)) != len(passbands)
                or radiance not in ("cpu", "device")):
            msg = "Supply unique passbands and cpu/device radiance"
            raise ValueError(msg)
        self.passbands, self.rule, self.field_levels = tuple(passbands), rule, field_levels
        self.device, self.radiance = device, radiance
        self.field = None if device == "reference" else _FieldHybrid(
            passbands, device=device, reflection="tiled", coverage="cpu", fallback="raise", radiance=radiance,
        )
        # Infrastructure/FP64 errors occur before numerical candidate handlers.
        self.integrator = None if device == "reference" else SurfaceFluxIntegrator(
            settings.LIMB_DARKENING_LAW, device=device,
        )
        self.law = settings.LIMB_DARKENING_LAW

    def _fields(self, binary: Any, observer: Any) -> tuple[Any, Any]:
        capture = _CaptureMesh(self.field_levels)
        if self.device == "reference":
            system = cpu_reflect(intrinsic(binary, capture), settings.REFLECTION_EFFECT_ITERATIONS)
            system.flat_it()
            prep_surface_params(system, return_values=False, write_to_containers=True,
                                passband=observer.passband, left_bandwidth=observer.left_bandwidth,
                                right_bandwidth=observer.right_bandwidth)
        else:
            self.field.capture = capture
            system, _, temperatures, _ = self.field._build_intrinsic_system(binary)
            if self.radiance == "device":
                self.field._prepare_radiance(system, observer, temperatures, diagnostics=True)
            else:
                prep_surface_params(system, return_values=False, write_to_containers=True,
                                    passband=observer.passband, left_bandwidth=observer.left_bandwidth,
                                    right_bandwidth=observer.right_bandwidth)
        return system, capture.triangles

    def __call__(self, binary: Any, phases: Any, *, observer: Any = None, normalize: bool = True,  # noqa: PLR0915
                 source_revision: str, diagnostics: bool = False) -> CurvedCurveResult:
        """Synthesize all phases with original mapping, third light and distance."""
        started = perf_counter()
        phases = np.array(phases, dtype=float, copy=True)
        if (phases.ndim != 1 or not 1 <= len(phases) <= _MAX_PHASES or not np.isfinite(phases).all()
                or not source_revision or settings.NUMBER_OF_PROCESSES != 1 or self.law != settings.LIMB_DARKENING_LAW):
            msg = "Curved synthesis requires1-128finite phases, fixed sources/law and serial execution"
            raise UnsupportedCurvedGeometryError(msg)
        model = deepcopy(binary)
        supplied = Observer(passband=list(self.passbands), system=model) if observer is None else deepcopy(observer)
        if tuple(supplied.passband) != self.passbands:
            msg = "Observer passbands/order must match the curved engine"
            raise ValueError(msg)
        supplied._system = model
        geometry = {n: RocheGeometry(model, n) for n in _NAMES}
        system, triangles = self._fields(model, supplied)
        fields_done = perf_counter()
        signature = hashlib.sha256(json.dumps({
            "binary": model.to_json(), "passbands": passband_revision(supplied.passband), "source": source_revision,
            "law": self.law, "reflection": (settings.REFLECTION_EFFECT, settings.REFLECTION_EFFECT_ITERATIONS),
            "single_ld": settings.USE_SINGLE_LD_COEFFICIENTS, "preserve_phases": settings.PRESERVE_INPUT_PHASES,
        }, sort_keys=True, default=str).encode()).hexdigest()
        resolved = supplied.manage_time_series(phases=phases)
        base, origin = supplied.phase_interval_reduce(resolved)
        support, reverse = dynamic.phase_crv_symmetry(system, base)
        motion = model.get_positions_method()(input_argument=support, return_nparray=False, calculate_from="phase")
        components = {b: np.zeros((len(motion), 2)) for b in self.passbands}
        records = []
        geometry_seconds = integration_seconds = 0.
        for t, position in enumerate(motion):
            step = perf_counter()
            line = utils.rotate_item(np.eye(3), position, model.inclination)@const.LINE_OF_SIGHT
            if abs(np.linalg.norm(line)-1) > 64*np.finfo(float).eps:
                msg = "Original observer transform failed its unit LOS check"
                raise UnsupportedCurvedGeometryError(msg)
            # Radius<.25 per star: a conservative projected-separation test,
            # independent of the old scalar-corrected faceted eclipse boundary.
            overlap_possible = np.sqrt(max(0., 1-line[0]**2)) < 2*.25
            foreground_name = "secondary" if line[0] > 0 else "primary"
            packets, stars, record = [], [], {"phase": float(position.phase), "components": {}}
            for name in _NAMES:
                foreground = geometry[foreground_name] if overlap_possible and name != foreground_name else None
                packet = curved_packet(triangles[name], geometry[name], line, self.rule, foreground)
                packets.append(packet)
                parents = packet["parents"]
                parent = getattr(system, name)
                stars.append(SimpleNamespace(
                    indices=np.arange(len(parents)), los_cosines=packet["mu"],
                    coverage=packet["area"]*model.semi_major_axis**2,
                    ld_cfs={b: parent.ld_cfs[b][parents] for b in self.passbands},
                    normal_radiance={b: parent.normal_radiance[b][parents] for b in self.passbands},
                ))
                record["components"][name] = {k: v for k, v in packet.items()
                                               if k not in ("mu", "area", "parents")}
            geometry_seconds += perf_counter()-step
            step = perf_counter()
            width = max(len(s.indices) for s in stars)
            pad = 1 << max(0, width-1).bit_length()
            for band in self.passbands:
                if self.device == "reference":
                    values = [np.sum(calculate_surface_element_fluxes(band, s)) for s in stars]
                else:
                    _, values = self.integrator.from_containers(band, stars, pad_to=pad)
                components[band][t] = np.asarray(values)
            integration_seconds += perf_counter()-step
            records.append(record)
        mapped = {b: v[reverse][origin] for b, v in components.items()}
        raw = HybridBinaryLightCurve._observed_fluxes(mapped, model, normalize=False)
        flux = HybridBinaryLightCurve._observed_fluxes(mapped, model, normalize=normalize)
        timings = {"fresh_finite_fields": fields_done-started,
                   "cpu_curved_geometry_adaptation_shadow": geometry_seconds,
                   "integration_with_transfers_compile": integration_seconds, "total": perf_counter()-started}
        detail = {"phase_records": records, "field_faces": {n: len(getattr(system, n).faces) for n in _NAMES},
                  "initial_system": system if diagnostics else None, "support_to_original": reverse[origin],
                  "method": "True Roche integration of piecewise-constant fresh original finite fields; "
                  "CPU geometry/masks, empirical boundary indicator, no continuum certificate"}
        return CurvedCurveResult(resolved+model.phase_shift, flux, raw, mapped, signature, asdict(self.rule),
                                 self.field_levels, float(model.additional_light), timings, detail)


def curved_difference(
    coarse: CurvedCurveResult, fine: CurvedCurveResult, sigma: Any, *, reducers: Any = None,
    mean_normalize: bool = False, observed: Any = None, ln_f: float = -np.inf,
    max_distance_sigma: float = .1, require_budget: bool = False,
) -> dict[str, Any]:
    """Enclose only a computed finite pair; nominal-noise gate is outside priors.

    Geometry/order/boundary/field protocol comparisons must be designed before
    inference. This never certifies an uncomputed surface or continuous field.
    """
    if (coarse.signature != fine.signature or not np.array_equal(coarse.phases, fine.phases)
            or coarse.component_fluxes.keys() != fine.component_fluxes.keys()):
        msg = "Curved comparison needs matching physical/source/phase snapshots"
        raise ValueError(msg)
    bands = list(coarse.component_fluxes)
    errors = sigma if isinstance(sigma, dict) else dict.fromkeys(bands, sigma)
    rows, predictions, envelopes, uncertainties, data = {}, [[], []], [], [], []
    for band in bands:
        index = np.arange(len(coarse.phases)) if reducers is None else np.asarray(reducers[band])
        if (index.ndim != 1 or not len(index) or not np.issubdtype(index.dtype, np.integer)
                or np.any((index < 0) | (index >= len(coarse.phases)))):
            msg = "Curved comparison needs valid nonempty observation reducers"
            raise ValueError(msg)
        weights, offsets, residuals = [], [], []
        for k, result in enumerate((coarse, fine)):
            total = result.component_fluxes[band].sum(axis=1)
            offset = total.mean()*result.additional_light/(1-result.additional_light)
            norm = (total[index]+offset).mean() if mean_normalize else (total+offset).max()
            if not np.isfinite(norm) or norm <= 0:
                msg = "Curved comparison normalization must be finite and positive"
                raise ValueError(msg)
            # Reuse the exact original maximum convention independently of the
            # caller's normalize flag; distance cancels from normalized curves.
            normalized = HybridBinaryLightCurve._observed_fluxes(
                {band: result.component_fluxes[band]}, SimpleNamespace(additional_light=result.additional_light),
                normalize=True,
            )[band][index]
            if mean_normalize:
                normalized /= normalized.mean()
            w, c = result.component_fluxes[band][index]/norm, offset/norm
            weights.append(w)
            offsets.append(c)
            residuals.append(normalized-w.sum(axis=1)-c)
            predictions[k].append(normalized)
        delta = weights[1]-weights[0]
        nuisance = offsets[1]-offsets[0]
        residual = residuals[1]-residuals[0]
        guard = (2*len(index)+32)*np.finfo(float).eps*(
            np.abs(weights[0]).sum(axis=1)+np.abs(weights[1]).sum(axis=1)+abs(offsets[0])+abs(offsets[1])+2
        )
        envelope = np.abs(delta).sum(axis=1)+abs(nuisance)+np.abs(residual)+guard
        actual = predictions[1][-1]-predictions[0][-1]
        rows[band] = {"actual_change": actual.tolist(), "component_envelope": envelope.tolist(),
                      "reconstruction_max_abs": float(np.max(np.abs(delta.sum(axis=1)+nuisance+residual-actual))),
                      "roundoff_guard_max": float(guard.max())}
        envelopes.append(envelope)
        uncertainties.append(np.broadcast_to(errors[band], actual.shape))
        if observed is not None:
            data.append(np.asarray(observed[band]))
    a, b = (np.concatenate(v) for v in predictions)
    error, envelope = np.concatenate(uncertainties), np.concatenate(envelopes)
    measured = resolution_error(b, a, error, observed=None if observed is None else np.concatenate(data),
                                ln_f=ln_f, max_distance_sigma=max_distance_sigma)
    nominal_bound = resolution_error(envelope, np.zeros_like(envelope), error, max_distance_sigma=max_distance_sigma)
    likelihood_bound = None
    if observed is not None:
        scale = error*np.exp(.5*np.logaddexp(0., 2*ln_f))
        r, e = (np.concatenate(data)-a)/scale, envelope/scale
        likelihood_bound = float(np.sum(np.abs(r)*e+.5*e**2))
    report = {"bands": rows, "actual": measured, "nominal_component_bound": nominal_bound,
              "finite_log_likelihood_absolute_bound": likelihood_bound,
              "within_finite_budget": nominal_bound["within_diagnostic_budget"], "continuum_certified": False,
              "coarse": {"rule": coarse.rule, "field_levels": coarse.field_levels},
              "fine": {"rule": fine.rule, "field_levels": fine.field_levels},
              "contract": "Finite prediction-change enclosure only; nominal sigma independent of ln_f. "
              "No interval boundary certificate, continuous field or continuum error bound."}
    if require_budget and not report["within_finite_budget"]:
        msg = "Curved finite quadrature/field change exceeds numerical budget; abort evaluation"
        raise ResolutionNotQualifiedError(msg, report)
    return report
