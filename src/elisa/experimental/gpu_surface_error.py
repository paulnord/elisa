"""Nested complete curves and finite, observation-normalized surface indicators.

The envelope encloses the *measured finite model change*, including third light,
normalization and summation residuals. It is not a continuum error bound or an
accuracy certificate. Unsupported refined meshes raise; no coarse fallback is
substituted for the requested numerical model. Legacy CPU defaults are untouched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np

from elisa import Observer, settings
from elisa.base.curves.utils import calculate_surface_element_fluxes
from elisa.experimental.gpu_accuracy import resolution_error
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_nested_mesh import NestedMesh, UnsupportedNestedMeshError
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_radiance import passband_revision

_MAX_PHASES = 128
_NAMES = ("primary", "secondary")


def _frozen(value: Any) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class SurfaceSnapshot:
    """Immutable grouped finite-model output; no physical containers are cached."""

    levels: int
    signature: str
    additional_light: float
    phases: np.ndarray
    fluxes: Any
    totals: Any
    groups: Any
    boundary: tuple[np.ndarray, np.ndarray]
    mu_min: tuple[np.ndarray, np.ndarray]
    mu_max: tuple[np.ndarray, np.ndarray]
    face_counts: tuple[int, int]
    elapsed_seconds: float
    curve_seconds: float


class NestedBinaryLightCurve(HybridBinaryLightCurve):
    """Explicit finite nested-grid experiment with CPU roots and coverage.

    Each complete curve rebuilds original seed mesh, projected midpoint vertices,
    gravity, corrected state, irradiation, radiance and coverage. There is no
    automatic coarse fallback. This is a standalone numerical diagnostic, not a
    registered fitting backend or a new physical prior.
    """

    def __init__(
        self, passbands: list[str], mesh: NestedMesh, *, device: str = "gpu", radiance: str = "device",
    ) -> None:
        if not isinstance(mesh, NestedMesh):
            msg = "Supply an explicit NestedMesh"
            raise TypeError(msg)
        self.mesh = mesh
        super().__init__(passbands, device=device, reflection="tiled", coverage="cpu", radiance=radiance,
                         fallback="raise")

    def _initial_mesh(self, system: Any) -> None:
        super()._initial_mesh(system)
        seed = hashlib.sha256()
        for name in _NAMES:
            star = getattr(system, name)
            seed.update(np.asarray(star.points, dtype=np.float64).tobytes())
            seed.update(np.asarray(star.faces, dtype=np.int64).tobytes())
        self._seed_signature = seed.hexdigest()  # diagnostic identity only; no reusable physical arrays
        self.mesh.apply(system)

    def _unsupported_curve(self, *_args: Any, reason: str, **_kwargs: Any) -> Any:
        raise UnsupportedNestedMeshError(reason)

    def snapshot(
        self, binary: Any, phases: Any, *, observer: Any = None, source_revision: str,
    ) -> SurfaceSnapshot:
        """Return complete host curves and root-face groups, charging all costs.

        source_revision identifies the caller's fixed scientific table/assets
        manifest. The worker verifies it independently before each allocation.
        Call this outside candidate-penalty handlers; unsupported numerics abort.
        """
        started = perf_counter()
        if settings.NUMBER_OF_PROCESSES != 1 or not source_revision:
            msg = "Surface-error snapshots require serial synthesis and a fixed source revision"
            raise UnsupportedNestedMeshError(msg)
        if not 1 <= len(phases) <= _MAX_PHASES:
            msg = "Surface-error snapshots support1-128 phases"
            raise UnsupportedNestedMeshError(msg)
        supplied = Observer(passband=list(self.passbands), system=binary) if observer is None else observer
        payload = {"binary": binary.to_json(), "passbands": passband_revision(supplied.passband),
                   "sources": source_revision, "law": settings.LIMB_DARKENING_LAW,
                   "reflection": (settings.REFLECTION_EFFECT, settings.REFLECTION_EFFECT_ITERATIONS),
                   "single_ld": settings.USE_SINGLE_LD_COEFFICIENTS,
                   "preserve_phases": settings.PRESERVE_INPUT_PHASES}
        result = self(binary, phases, normalize=True, diagnostics=True, observer=supplied)
        payload["seed_mesh"] = self._seed_signature
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        detail = result.diagnostics
        positions, mapping = detail["positions"], detail["support_to_original"]
        counts = [len(getattr(positions[0], n).faces) for n in _NAMES]
        roots = [n//4**self.mesh.levels for n in counts]
        groups = {b: [np.zeros((len(positions), n)) for n in roots] for b in self.passbands}
        boundary = [np.zeros((len(positions), n), dtype=bool) for n in roots]
        mu_min = [np.full((len(positions), n), np.inf) for n in roots]
        mu_max = [np.full((len(positions), n), -np.inf) for n in roots]
        for t, on_pos in enumerate(positions):
            for component, name in enumerate(_NAMES):
                star = getattr(on_pos, name)
                base = len(star.faces)//4//4**self.mesh.levels
                parents = np.concatenate([np.repeat(np.arange(base), 4**self.mesh.levels)+q*base for q in range(4)])
                for band in self.passbands:
                    values = calculate_surface_element_fluxes(band, star)
                    groups[band][component][t] = np.bincount(parents[star.indices], values, minlength=roots[component])
                fraction = star.coverage/(star.areas*on_pos.semi_major_axis**2)
                partial = (fraction > 0) & (fraction < 1-16*np.finfo(float).eps)
                boundary[component][t] = np.bincount(parents, partial.astype(int), minlength=roots[component]) > 0
                np.minimum.at(mu_min[component][t], parents, star.los_cosines)
                np.maximum.at(mu_max[component][t], parents, star.los_cosines)
        frozen_groups = {b: tuple(_frozen(v[mapping]) for v in values) for b, values in groups.items()}
        return SurfaceSnapshot(
            self.mesh.levels, signature, float(binary.additional_light), _frozen(result.phases),
            MappingProxyType({b: _frozen(v) for b, v in result.fluxes.items()}),
            MappingProxyType({b: _frozen(v.sum(axis=1)) for b, v in result.component_fluxes.items()}),
            MappingProxyType(frozen_groups), tuple(_frozen(v[mapping]) for v in boundary),
            tuple(_frozen(v[mapping]) for v in mu_min), tuple(_frozen(v[mapping]) for v in mu_max),
            tuple(counts), perf_counter()-started, result.timings["total"],
        )


def compare_surfaces(  # noqa: PLR0915 - explicit finite decomposition and observation mapping
    coarse: SurfaceSnapshot, fine: SurfaceSnapshot, sigma: Any, *, reducers: Any = None,
    mean_normalize: bool = False, ln_f: float = -np.inf, max_distance_sigma: float = .1,
    require_budget: bool = False, observed: Any = None,
) -> dict[str, Any]:
    """Finite root-face change, cancellation envelope and contact/limb attribution.

    Optional reducers preserve actual per-band observation order/multiplicity;
    mean_normalize applies the original fitter's final observation mean. The
    finite budget uses nominal sigma independently of nuisance noise inflation;
    likelihood diagnostics retain the original ln_f law. Source,
    physical candidate and phase snapshots must match. No continuum extrapolation
    or change of likelihood/noise/prior semantics is performed.
    """
    if (coarse.signature != fine.signature or fine.levels <= coarse.levels
            or not np.array_equal(coarse.phases, fine.phases) or coarse.fluxes.keys() != fine.fluxes.keys()):
        msg = "Compare ordered nested levels of the same candidate/phases/source snapshot"
        raise ValueError(msg)
    bands = list(coarse.fluxes)
    errors = sigma if isinstance(sigma, dict) else dict.fromkeys(bands, sigma)
    rows, envelopes, component_envelopes, sigmas = {}, [], [], []
    coarse_predictions, fine_predictions, observations = [], [], []
    for band in bands:
        index = np.arange(len(coarse.phases)) if reducers is None else np.asarray(reducers[band])
        if index.ndim != 1 or not len(index) or not np.issubdtype(index.dtype, np.integer) or np.any(
            (index < 0) | (index >= len(coarse.phases)),
        ):
            msg = "Supply nonempty valid observation reducer indices"
            raise ValueError(msg)
        components, constants, residuals, predictions = [], [], [], []
        for snapshot in (coarse, fine):
            raw = snapshot.totals[band]
            offset = raw.mean()*snapshot.additional_light/(1-snapshot.additional_light)
            norm = (raw[index]+offset).mean() if mean_normalize else (raw+offset).max()
            if not np.isfinite(norm) or norm <= 0:
                msg = "Surface normalization must be finite and positive"
                raise ValueError(msg)
            weights = np.concatenate(snapshot.groups[band], axis=1)[index]/norm
            prediction = snapshot.fluxes[band][index].copy()
            if mean_normalize:
                prediction /= prediction.mean()
            constant = offset/norm
            residual = prediction-weights.sum(axis=1)-constant
            components.append(weights)
            constants.append(constant)
            residuals.append(residual)
            predictions.append(prediction)
        delta = components[1]-components[0]
        nuisance = constants[1]-constants[0]
        residual = residuals[1]-residuals[0]
        reconstructed = delta.sum(axis=1)+nuisance+residual
        observed_change = predictions[1]-predictions[0]
        # Conservative host arithmetic guard for the finite decomposition.
        # k*eps/(1-k*eps) bounds sequential accumulation; it is not a physical
        # truncation/error model and cannot bound an uncomputed finer surface.
        k = 2*delta.shape[1]+2*len(index)+32
        eps = np.finfo(float).eps
        gamma = k*eps/(1-k*eps)
        guard = gamma*(np.abs(components[0]).sum(axis=1)+np.abs(components[1]).sum(axis=1)
                       +abs(constants[0])+abs(constants[1])+np.abs(predictions[0])+np.abs(predictions[1]))
        envelope = np.abs(delta).sum(axis=1)+abs(nuisance)+np.abs(residual)+guard
        split = coarse.groups[band][0].shape[1]
        component_envelope = (np.abs(delta[:, :split].sum(axis=1))+np.abs(delta[:, split:].sum(axis=1))
                              +abs(nuisance)+np.abs(residual)+guard)
        boundary = np.concatenate([a | b for a, b in zip(coarse.boundary, fine.boundary, strict=True)], axis=1)[index]
        limb = np.concatenate([(np.minimum(a, b) <= 0) & (np.maximum(c, d) > 0) for a, b, c, d in zip(
            coarse.mu_min, fine.mu_min, coarse.mu_max, fine.mu_max, strict=True,
        )], axis=1)[index]
        metric = resolution_error(predictions[1], predictions[0], errors[band], ln_f=ln_f,
                                  max_distance_sigma=max_distance_sigma,
                                  observed=None if observed is None else observed[band])
        effective_error = np.broadcast_to(errors[band], observed_change.shape)*np.exp(.5*np.logaddexp(0., 2*ln_f))
        weighted_delta = delta/effective_error[:, None]
        parent_score = np.linalg.norm(weighted_delta, axis=0)
        top = np.argsort(parent_score)[-min(10, len(parent_score)):][::-1]
        rows[band] = {"actual": metric,
                      "finite_envelope": envelope.tolist(), "actual_change": observed_change.tolist(),
                      "reconstruction_max_abs": float(np.max(np.abs(reconstructed-observed_change))),
                      "roundoff_guard_max": float(guard.max()), "third_light_normalization_change": float(nuisance),
                      "summation_normalization_residual_max_abs": float(np.max(np.abs(residual))),
                      "component_envelope": component_envelope.tolist(),
                      "partial_eclipse_L1_fraction": float(np.abs(weighted_delta)[boundary].sum()
                                                           /max(np.abs(weighted_delta).sum(), np.finfo(float).tiny)),
                      "limb_L1_fraction": float(np.abs(weighted_delta)[limb].sum()
                                                /max(np.abs(weighted_delta).sum(), np.finfo(float).tiny)),
                      "boundary_limb_L1_fraction": float(np.abs(weighted_delta)[boundary | limb].sum()
                                                         /max(np.abs(weighted_delta).sum(), np.finfo(float).tiny)),
                      "largest_root_groups": top.tolist(), "root_group_scores": parent_score[top].tolist()}
        envelopes.append(envelope)
        component_envelopes.append(component_envelope)
        sigmas.append(np.broadcast_to(errors[band], observed_change.shape))
        coarse_predictions.append(predictions[0])
        fine_predictions.append(predictions[1])
        if observed is not None:
            observations.append(np.asarray(observed[band]))
    joint = resolution_error(
        np.concatenate(fine_predictions), np.concatenate(coarse_predictions), np.concatenate(sigmas),
        ln_f=ln_f, max_distance_sigma=max_distance_sigma,
        observed=None if observed is None else np.concatenate(observations),
    )
    bound = resolution_error(np.concatenate(envelopes), np.zeros(sum(map(len, envelopes))), np.concatenate(sigmas),
                             ln_f=ln_f, max_distance_sigma=max_distance_sigma)
    component_bound = resolution_error(
        np.concatenate(component_envelopes), np.zeros(sum(map(len, envelopes))), np.concatenate(sigmas),
        ln_f=ln_f, max_distance_sigma=max_distance_sigma,
    )
    nominal_bound = resolution_error(
        np.concatenate(component_envelopes), np.zeros(sum(map(len, envelopes))), np.concatenate(sigmas),
        max_distance_sigma=max_distance_sigma,
    )
    likelihood_bound = None
    if observed is not None:
        effective_error = np.concatenate(sigmas)*np.exp(.5*np.logaddexp(0., 2*ln_f))
        r = (np.concatenate(observations)-np.concatenate(coarse_predictions))/effective_error
        e = np.concatenate(component_envelopes)/effective_error
        likelihood_bound = float(np.sum(np.abs(r)*e+.5*e**2))
    report = {"levels": (coarse.levels, fine.levels), "bands": rows, "joint": joint,
              "finite_envelope_joint": bound, "component_envelope_joint": component_bound,
              "nominal_component_envelope_joint": nominal_bound,
              "finite_log_likelihood_absolute_bound": likelihood_bound,
              "within_finite_budget": nominal_bound["within_diagnostic_budget"],
              "budget_rule": "Nominal-sigma component envelope, independent of ln_f; "
              "root-face L1 is a separate localization diagnostic",
              "continuum_certified": False,
              "contract": "Finite normalized model-change envelope only; no uncomputed/continuum error bound, "
              "no assumed order/monotonicity, prior cut or CPU fallback"}
    if require_budget and not report["within_finite_budget"]:
        msg = "Finite nested surface change exceeds numerical budget; abort evaluation"
        raise ResolutionNotQualifiedError(msg, report)
    return report
