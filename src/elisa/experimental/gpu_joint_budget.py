"""Joint finite field/boundary checks anchored to one complete prediction.

This bounds the sum of the supplied finite perturbations, including their
Gaussian quadratic cross terms. It does not enclose an uncomputed combination,
interaction, continuum model or posterior. Axes and resolutions must be chosen
before inference; failures stay outside physical-prior penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa.experimental.gpu_curved import curved_difference
from elisa.experimental.gpu_curved_edges import InferenceBudget
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError


@dataclass(frozen=True)
class JointInferenceBudget:
    """Same nominal D<=.1, |delta logL|<=.01 gates applied to the sum envelope."""

    max_distance_sigma: float = 0.1
    max_log_likelihood: float = 0.01

    def __post_init__(self) -> None:
        InferenceBudget(self.max_distance_sigma, self.max_log_likelihood)

    def compare(
        self,
        anchor: Any,
        alternatives: dict[str, Any],
        sigma: Any,
        *,
        observed: Any,
        reducers: Any = None,
        mean_normalize: bool = False,
        require_budget: bool = True,
    ) -> dict:
        if observed is None:
            msg = "Joint finite likelihood controls require observed data"
            raise ValueError(msg)
        if not alternatives or any(not isinstance(k, str) or not k.strip() for k in alternatives):
            msg = "Joint finite controls need named, predeclared nonempty axes"
            raise ValueError(msg)
        reports = {
            name: curved_difference(
                anchor,
                variant,
                sigma,
                observed=observed,
                reducers=reducers,
                mean_normalize=mean_normalize,
                ln_f=-np.inf,
                max_distance_sigma=self.max_distance_sigma,
            )
            for name, variant in alternatives.items()
        }
        bands = list(anchor.component_fluxes)
        envelopes, scales, per_band = [], [], {}
        cross_terms = 0.0
        # All finite pairs have the SAME anchor residual and original observation
        # mapping/normalization. Sum |r|E_i+.5 E_i^2, then add sum_{i<j} E_i E_j.
        likelihood = sum(r["finite_log_likelihood_absolute_bound"] for r in reports.values())
        for band in bands:
            axis = np.array([r["bands"][band]["component_envelope"] for r in reports.values()])
            error = np.broadcast_to(sigma[band] if isinstance(sigma, dict) else sigma, axis.shape[1:])
            total = np.sum(axis, axis=0)
            scaled = axis / error
            for i in range(len(axis)):
                for j in range(i):
                    cross_terms += float(np.sum(scaled[i] * scaled[j]))
            envelopes.append(total)
            scales.append(error)
            per_band[band] = {"sum_component_envelope": total.tolist()}
        distance = float(np.linalg.norm(np.concatenate(envelopes) / np.concatenate(scales)))
        likelihood += cross_terms
        # The computed-pair envelopes already include their reconstruction guard.
        # Account for this additional positive summation and quadratic arithmetic.
        rounding = (8 * len(reports) + 8 * sum(len(v) for v in envelopes) + 32) * np.finfo(float).eps
        distance = float(np.nextafter(distance * (1 + rounding), np.inf))
        likelihood = float(np.nextafter(likelihood * (1 + rounding), np.inf))
        qualified = distance <= self.max_distance_sigma and likelihood <= self.max_log_likelihood
        result = {
            "axes": reports,
            "bands": per_band,
            "distance_sigma_bound": distance,
            "finite_log_likelihood_absolute_bound": likelihood,
            "quadratic_cross_terms": cross_terms,
            "within_finite_budget": qualified,
            "max_distance_sigma": self.max_distance_sigma,
            "max_log_likelihood": self.max_log_likelihood,
            "nuisance_noise_used_for_gate": False,
            "continuum_certified": False,
            "uncomputed_interactions_certified": False,
            "contract": "Sum of specified finite perturbation envelopes about the common anchor. "
            "No uncomputed combined-resolution, continuum, posterior or model-adequacy certificate.",
        }
        if require_budget and not qualified:
            msg = "Joint finite field/boundary change exceeds nominal distance or likelihood budget; abort evaluation"
            raise ResolutionNotQualifiedError(msg, result)
        return result
