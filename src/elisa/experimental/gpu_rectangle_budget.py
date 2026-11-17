"""Finite two-axis controls with an explicitly computed interaction corner.

The four curves are one field/depth rectangle, not samples of a continuum bound.
The mixed finite difference closes the algebra for this COMPUTED combination.
Uncomputed combinations/interactions, posterior bias and model adequacy remain
uncertified. Nominal accuracy failures always propagate outside physical priors.
"""

# ruff: noqa: SLF001 - original observation normalization helper
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_curved_edges import InferenceBudget
from elisa.experimental.gpu_joint_budget import JointInferenceBudget
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError


@dataclass(frozen=True)
class RectangleInferenceBudget:
    max_distance_sigma: float = 0.1
    max_log_likelihood: float = 0.01

    def __post_init__(self) -> None:
        InferenceBudget(self.max_distance_sigma, self.max_log_likelihood)

    def compare(
        self,
        fine_deep: Any,
        coarse_deep: Any,
        fine_shallow: Any,
        coarse_shallow: Any,
        sigma: Any,
        *,
        observed: Any,
        reducers: Any = None,
        mean_normalize: bool = False,
        require_budget: bool = True,
    ) -> dict:
        a, f, d, c = fine_deep, coarse_deep, fine_shallow, coarse_shallow
        if not (
            a.field_levels == d.field_levels > f.field_levels == c.field_levels
            and a.rule == f.rule
            and d.rule == c.rule
            and a.rule["depth"] > d.rule["depth"]
            and {k: v for k, v in a.rule.items() if k != "depth"} == {k: v for k, v in d.rule.items() if k != "depth"}
        ):
            msg = "Finite rectangle requires matched field levels and only a boundary-depth change"
            raise ValueError(msg)
        joint = JointInferenceBudget(self.max_distance_sigma, self.max_log_likelihood).compare(
            a,
            {"field": f, "depth": d},
            sigma,
            observed=observed,
            reducers=reducers,
            mean_normalize=mean_normalize,
            require_budget=False,
        )
        pair = InferenceBudget(self.max_distance_sigma, self.max_log_likelihood).compare(
            a,
            c,
            sigma,
            observed=observed,
            reducers=reducers,
            mean_normalize=mean_normalize,
            require_budget=False,
        )
        envelopes = []
        errors = []
        residuals = []
        mixeds = []
        bands = {}
        for band in a.component_fluxes:
            field = joint["axes"]["field"]["bands"][band]
            depth = joint["axes"]["depth"]["bands"][band]
            combined = pair["nominal_report"]["bands"][band]
            mixed = np.array(combined["actual_change"]) - field["actual_change"] - np.array(depth["actual_change"])
            guard = (2 * len(mixed) + 64) * np.finfo(float).eps * np.maximum(1.0, np.abs(mixed))
            interaction = np.abs(mixed) + guard
            envelope = np.array(field["component_envelope"]) + depth["component_envelope"] + interaction
            pred = HybridBinaryLightCurve._observed_fluxes(a.component_fluxes, a, normalize=True)[band]
            if reducers is not None:
                pred = pred[np.asarray(reducers[band])]
            if mean_normalize:
                pred = pred / pred.mean()
            error = np.broadcast_to(sigma[band] if isinstance(sigma, dict) else sigma, pred.shape)
            errors.append(error)
            envelopes.append(envelope)
            residuals.append(np.asarray(observed[band]) - pred)
            mixeds.append(mixed)
            bands[band] = {
                "measured_interaction": mixed.tolist(),
                "interaction_envelope": interaction.tolist(),
                "computed_combination_envelope": envelope.tolist(),
            }
            if not np.all(np.abs(combined["actual_change"]) <= envelope):
                msg = "Computed rectangle failed its finite algebraic enclosure"
                raise ResolutionNotQualifiedError(msg, {"computed_combination_enclosed": False})
        error = np.concatenate(errors)
        envelope = np.concatenate(envelopes)
        residual = np.concatenate(residuals)
        rounding = (8 * len(error) + 64) * np.finfo(float).eps
        distance = float(np.nextafter(np.linalg.norm(envelope / error) * (1 + rounding), np.inf))
        bound = float(
            np.nextafter(
                np.sum(abs(residual) * envelope / error**2 + 0.5 * (envelope / error) ** 2) * (1 + rounding),
                np.inf,
            ),
        )
        qualified = (
            pair["within_finite_budget"] and distance <= self.max_distance_sigma and bound <= self.max_log_likelihood
        )
        result = {
            "linear_axes_control": joint,
            "direct_combined_pair": pair,
            "bands": bands,
            "measured_interaction_distance_sigma": float(np.linalg.norm(np.concatenate(mixeds) / error)),
            "computed_combination_distance_sigma_bound": distance,
            "computed_combination_log_likelihood_bound": bound,
            "within_finite_budget": qualified,
            "max_distance_sigma": self.max_distance_sigma,
            "max_log_likelihood": self.max_log_likelihood,
            "nuisance_noise_used_for_gate": False,
            "computed_combination_enclosed": True,
            "uncomputed_interactions_certified": False,
            "continuum_certified": False,
            "contract": "Four computed finite field/depth corners; measured interaction closes this combination only. "
            "Direct component-envelope gate retained. No continuum, uncomputed interaction, "
            "fit or posterior certificate.",
        }
        if require_budget and not qualified:
            msg = "Computed field/depth rectangle exceeds nominal inference budget"
            raise ResolutionNotQualifiedError(msg, result)
        return result
