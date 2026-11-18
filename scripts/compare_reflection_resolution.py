"""Bound the measured field, donor and eclipse-depth changes in eight complete curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_curved_edges import InferenceBudget
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_rectangle_budget import RectangleInferenceBudget
from scripts.compare_gpu_joint_budget import curve

# A set bit changes this axis from its high to low resolution.
FIELD, DONOR, DEPTH = 1, 2, 4


def envelope(changes: dict, component_bounds: dict) -> tuple:
    """Möbius differences for the observed corners, with explicit FP64 accumulation guard."""
    terms = {}
    total = sum(component_bounds.values())
    for mask in range(1, 8):
        lower = [v for k, v in terms.items() if (k & mask) == k]
        terms[mask] = changes[mask] - sum(lower)
        if mask not in component_bounds:
            total = total + np.abs(terms[mask])
    scale = np.maximum(1.0, sum(np.abs(v) for v in changes.values()))
    total = total + 128 * np.finfo(float).eps * scale
    if any(np.any(np.abs(v) > total) for v in changes.values()):
        raise ArithmeticError("Finite reflection error envelope failed to enclose a measured corner")
    return total, terms


def check(curves, sigma, observed, reducers, *, require_budget=True):
    """Independent finite corner gates, both field/depth rectangles and the three-axis enclosure."""
    opts = dict(observed=observed, reducers=reducers, mean_normalize=True, require_budget=False)
    budget = InferenceBudget()
    rows = {m: budget.compare(curves[0], curves[m], sigma, **opts) for m in range(1, 8)}
    rectangles = {
        m: RectangleInferenceBudget().compare(
            curves[m], curves[m | FIELD], curves[m | DEPTH], curves[m | FIELD | DEPTH], sigma, **opts
        )
        for m in (0, DONOR)
    }
    bands, errors, residuals = {}, [], []
    for band in observed:
        reports = {m: r["nominal_report"]["bands"][band] for m, r in rows.items()}
        e, terms = envelope(
            {m: np.array(r["actual_change"]) for m, r in reports.items()},
            {m: np.array(reports[m]["component_envelope"]) for m in (FIELD, DONOR, DEPTH)},
        )
        pred = HybridBinaryLightCurve._observed_fluxes(curves[0].component_fluxes, curves[0], normalize=True)[band]
        pred = pred[np.asarray(reducers[band])]
        pred = pred / pred.mean()
        err = np.broadcast_to(sigma[band], pred.shape)
        errors.extend(e / err)
        residuals.extend((np.asarray(observed[band]) - pred) / err)
        bands[band] = {"envelope": e.tolist(), "signed_terms": {m: v.tolist() for m, v in terms.items()}}
    e = np.array(errors)
    guard = 1 + (8 * len(e) + 96) * np.finfo(float).eps
    distance = float(np.nextafter(np.linalg.norm(e) * guard, np.inf))
    bound = float(np.nextafter(np.sum(np.abs(residuals) * e + 0.5 * e**2) * guard, np.inf))
    qualified = (
        all(r["within_finite_budget"] for r in [*rows.values(), *rectangles.values()])
        and distance <= budget.max_distance_sigma
        and bound <= budget.max_log_likelihood
    )
    result = {
        "combined_distance_sigma": distance,
        "combined_log_likelihood_bound": bound,
        "within_finite_budget": qualified,
        "bands": bands,
        "pair_bounds": {m: r["nominal_report"]["finite_log_likelihood_absolute_bound"] for m, r in rows.items()},
        "signed_delta_log_likelihood": {
            m: r["nominal_report"]["actual"]["delta_log_likelihood"] for m, r in rows.items()
        },
        "rectangle_bounds": {m: r["computed_combination_log_likelihood_bound"] for m, r in rectangles.items()},
        "axis_bits": {"field": FIELD, "donor": DONOR, "depth": DEPTH},
        "limit_D": budget.max_distance_sigma,
        "limit_log_likelihood": budget.max_log_likelihood,
        "continuum_certified": False,
    }
    if require_budget and not qualified:
        raise ResolutionNotQualifiedError("Computed reflection/field/depth combination exceeds nominal budget", result)
    return result


def compare(profiles: list[dict]) -> list[dict]:
    if len(profiles) != 4:
        raise ValueError("Supply exactly two donor levels crossed with two eclipse depths")
    ref = profiles[0]
    keys = (
        "engine",
        "backend",
        "levels",
        "integration_level",
        "receiver_points",
        "order",
        "slice_order",
        "irradiation",
        "source_revision",
        "physical_constants",
        "batch",
        "jd",
        "iteration_start",
        "reducers",
        "observations",
        "uncertainties",
    )
    if (
        ref["engine"] not in ("area-reflection", "angular-reflection", "spatial-reflection", "emitting-area")
        or not ref["irradiation"]
        or len(ref["levels"]) != 2
        or any(any(p[k] != ref[k] for k in keys) for p in profiles)
    ):
        raise ValueError("Expected matching irradiated profiles differing only in donor and depth")
    caps = sorted({p["donor_level"] for p in profiles}, reverse=True)
    depths = sorted({p["depth"] for p in profiles}, reverse=True)
    if len(caps) != 2 or len(depths) != 2 or len({(p["donor_level"], p["depth"]) for p in profiles}) != 4:
        raise ValueError("Incomplete receiver/field/depth grid")
    ordered = sorted(profiles, key=lambda p: (-p["depth"], -p["donor_level"]))
    out = []
    for samples in zip(*(p["samples"] for p in ordered), strict=True):
        for rows in zip(*(s["rows"] for s in samples), strict=True):
            grid = {}
            base_rule = None
            for j, (p, row) in enumerate(zip(ordered, rows, strict=True)):
                if row["physical"] != rows[0]["physical"]:
                    raise ValueError("Changed physical proposal in refinement comparison")
                if j % 2 and any(row[k] != rows[j - 1][k] for k in ("phase_records", "quadrature_sha256")):
                    raise ValueError("Changing donors must preserve all observer rays at fixed depth")
                for key in row["curves"]:
                    c = curve(row, key)
                    rule = dict(c.rule)
                    prefix = p["engine"].replace("-", "_")
                    marker = f"{prefix}_receiver{p['receiver_points']}_donor_cap{p['donor_level']}"
                    if (
                        rule.pop(marker, None) != 1
                        or rule.pop("depth") != p["depth"]
                        or c.field_levels not in p["levels"]
                    ):
                        raise ValueError("Curve rule disagrees with input resolution metadata")
                    if base_rule is not None and base_rule != rule:
                        raise ValueError("Unmatched observer or source protocol")
                    base_rule = rule
                    grid[2 * j + int(c.field_levels == min(p["levels"]))] = c
            if grid.keys() != set(range(8)):
                raise ValueError("Missing a finite field corner")
            args = grid, ref["uncertainties"], ref["observations"], ref["reducers"]
            try:
                result = check(*args)
            except ResolutionNotQualifiedError as e:
                result = e.report
            out.append({"physical": rows[0]["physical"], "control": result})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs=4, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [p.read_bytes() for p in args.profiles]
    result = {
        "rows": compare([json.loads(b) for b in inputs]),
        "input_sha256": [hashlib.sha256(b).hexdigest() for b in inputs],
        "contract": "Nominal finite changes including all measured mixed terms; original observation mapping and strict limits.",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
