"""Extend the eight-curve field/donor/eclipse check to an independent interior cubature rule."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_curved_edges import InferenceBudget
from scripts.compare_gpu_joint_budget import curve
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_quadratic_source import assess


def grid(parts, degree, primary_cubic=False):
    engine = ("symmetric-" if degree is not None else "") + "shadow-sampled-receiver-heat"
    if primary_cubic:
        engine += "-primary-cubic"
    result = assess(parts, engine=engine, interior_degree=degree)
    profiles = [profile for part in parts for profile in part["profiles"]]
    finest = [p for p in profiles if (p["depth"], p["donor_level"]) == (4, 2)]
    if len(finest) != 1:
        raise ValueError("Expected a unique finest field/donor/eclipse corner")
    p = finest[0]
    row = p["samples"][0]["rows"][0]
    return result, p, row


def mapped(c, reducers):
    flux = HybridBinaryLightCurve._observed_fluxes(c.component_fluxes, c, normalize=True)
    samples = {b: f[np.asarray(reducers[b])] for b, f in flux.items()}
    return {b: v / np.mean(v) for b, v in samples.items()}


def stellar_changes(coarse, fine, reducers, sigma, observed):
    """Average both orders of replacing one emitting star, with third light and normalization reapplied."""

    def mixed(first, second):
        return mapped(
            replace(
                fine,
                component_fluxes={
                    b: np.column_stack((first.component_fluxes[b][:, 0], second.component_fluxes[b][:, 1]))
                    for b in sigma
                },
            ),
            reducers,
        )

    low, high = mapped(coarse, reducers), mapped(fine, reducers)
    primary, secondary = mixed(fine, coarse), mixed(coarse, fine)
    components = {name: [] for name in ("primary", "secondary", "total")}
    likelihood = {name: [] for name in components}
    observation_ids = []
    scales = {}
    for b, err in sigma.items():
        err = np.asarray(err)
        # Fit the scale from the summed component light at the sampled phases. The residual uses peak
        # component flux as its denominator so total eclipses remain well defined.
        old = coarse.component_fluxes[b]
        new = fine.component_fluxes[b]
        ratio = new.sum(axis=0) / old.sum(axis=0)
        residual = np.max(np.abs(new - old * ratio), axis=0) / new.max(axis=0)
        scales[b] = {
            name: dict(mean_fractional_change=float(ratio[j] - 1), peak_scaled_shape_residual=float(residual[j]))
            for j, name in enumerate(("primary", "secondary"))
        }
        a = ((primary[b] - low[b]) + (high[b] - secondary[b])) / 2
        d = ((secondary[b] - low[b]) + (high[b] - primary[b])) / 2
        total = high[b] - low[b]
        np.testing.assert_allclose(a + d, total, rtol=0, atol=4 * np.finfo(float).eps)
        for name, v in zip(components, (a, d, total), strict=True):
            components[name].extend(v / err)

        residual_sigma = (np.asarray(observed[b]) - low[b]) / err
        dp, ds, dt = [(flux[b] - low[b]) / err for flux in (primary, secondary, high)]
        lp, ls, lt = [x * (residual_sigma - x / 2) for x in (dp, ds, dt)]
        # Average the two replacement orders in log likelihood as well as flux.
        likelihood["primary"].extend((lp + (lt - ls)) / 2)
        likelihood["secondary"].extend((ls + (lt - lp)) / 2)
        likelihood["total"].extend(lt)
        observation_ids.extend(
            dict(band=b, band_index=i, phase=float(fine.phases[j])) for i, j in enumerate(reducers[b])
        )
    np.testing.assert_allclose(np.add(likelihood["primary"], likelihood["secondary"]), likelihood["total"],
                               rtol=8e-15, atol=4e-17)
    attribution = {}
    for name, values in likelihood.items():
        largest = int(np.argmax(np.abs(values)))
        attribution[name] = dict(
            signed_change=float(np.sum(values)),
            largest_observation=dict(**observation_ids[largest], signed_change=float(values[largest])),
        )
    return dict(
        mapped_distance_sigma={name: float(np.linalg.norm(values)) for name, values in components.items()},
        component_flux_changes=scales,
        nominal_log_likelihood_changes=attribution,
    )


def compare(target, alternative, alternative_degree, *, primary_cubic=False):
    """Enclose all 16 computed corners relative to the six-point finest curve."""
    if alternative_degree not in (None, 5):
        raise ValueError("Choose a distinct seven-point or Duffy interior comparison")
    a, pa, ra = grid(target, 4, primary_cubic)
    b, pb, rb = grid(alternative, alternative_degree, primary_cubic)
    for key in ("backend", "source_revision", "physical_constants", "observations", "uncertainties", "reducers"):
        if pa[key] != pb[key]:
            raise ValueError(f"Interior comparison changed {key}")
    if a["physical"] != b["physical"]:
        raise ValueError("Interior comparison requires identical physical parameters")
    ca, cb = (curve(row, "curvature_field3") for row in (ra, rb))

    # Signatures bind the physical system and tables. All non-interior rule settings must also match.
    def common(c):
        return {k: v for k, v in c.rule.items() if k != "interior_degree" and not k.endswith("_receiver3_donor_cap2")}

    if common(ca) != common(cb):
        raise ValueError("A non-interior numerical setting changed")
    budget = InferenceBudget()
    observed, sigma, reducers = (pa[k] for k in ("observations", "uncertainties", "reducers"))
    pair = budget.compare(
        ca, cb, sigma, observed=observed, reducers=reducers, mean_normalize=True, require_budget=False
    )
    bridge = pair["nominal_report"]
    pred = mapped(ca, reducers)
    residuals, errors = [], []
    for band, y in observed.items():
        np.testing.assert_allclose(pred[band], ra["prediction"][band], atol=2e-12, rtol=3e-13)
        # Walk from the target finest corner to the other finest corner, then to any of its seven alternatives.
        other = np.array(b["finite_control"]["bands"][band]["envelope"]) + bridge["bands"][band]["component_envelope"]
        envelope = np.maximum(a["finite_control"]["bands"][band]["envelope"], other)
        errors.extend(envelope / sigma[band])
        residuals.extend((y - pred[band]) / sigma[band])
    e, r = np.array(errors), np.array(residuals)
    rounding = 1 + (len(e) * 8 + 96) * np.finfo(float).eps
    distance = float(np.nextafter(np.linalg.norm(e) * rounding, np.inf))
    bound = float(np.nextafter(np.sum(np.abs(r) * e + 0.5 * e**2) * rounding, np.inf))
    measured = []
    for part in [*target, *alternative]:
        for profile in part["profiles"]:
            row = profile["samples"][0]["rows"][0]
            for variant in row["curves"]:
                other = mapped(curve(row, variant), reducers)
                delta = np.concatenate([(other[k] - pred[k]) / sigma[k] for k in sigma])
                actual = float(delta @ (r - 0.5 * delta))
                if not np.all(np.abs(delta) <= e) or not abs(actual) <= bound:
                    raise ArithmeticError("Interior envelope failed to contain a measured corner")
                measured.append(abs(actual))
    if len(measured) != 16:
        raise ValueError("Interior comparison requires sixteen finite models")
    passed = (
        a["usable_log_posterior"] is not None
        and b["usable_log_posterior"] is not None
        and pair["within_finite_budget"]
        and distance <= budget.max_distance_sigma
        and bound <= budget.max_log_likelihood
    )
    return dict(
        alternative="Duffy 3x3" if alternative_degree is None else "symmetric seven-point",
        distance_sigma=distance,
        log_likelihood_bound=bound,
        limit_distance=budget.max_distance_sigma,
        limit_log_likelihood=budget.max_log_likelihood,
        pass_finite_check=passed,
        continuum_certified=False,
        max_measured_nominal_log_likelihood_change=max(measured),
        within_rule_bounds=[x["finite_control"]["combined_log_likelihood_bound"] for x in (a, b)],
        interior_pair_bound=bridge["finite_log_likelihood_absolute_bound"],
        interior_pair_distance_sigma=bridge["actual"]["distance_sigma"],
        observed_field_changes=stellar_changes(curve(ra, "curvature_field2"), ca, reducers, sigma, observed),
        usable_log_posterior=a["usable_log_posterior"] if passed else None,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--six", type=Path, nargs=2, required=True)
    p.add_argument("--alternative", type=Path, nargs=2, required=True)
    p.add_argument("--alternative-rule", choices=("duffy", "seven"), required=True)
    p.add_argument("--primary-cubic", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        target = [json.loads(path.read_bytes()) for path in args.six]
        alternative = [json.loads(path.read_bytes()) for path in args.alternative]
        result = compare(target, alternative, 5 if args.alternative_rule == "seven" else None, primary_cubic=args.primary_cubic)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
