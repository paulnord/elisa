"""Compare the six-node optical field and enhanced junction/slice rule on complete reflected curves."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.analytics.params import parameters
from elisa.experimental.quadratic_source import QuadraticSourceLightCurve
from scripts.compare_gpu_joint_budget import curve
from scripts.compare_reflection_resolution import check, DEPTH, DONOR, FIELD
from scripts.compare_reflected_posteriors import mapped_density
from scripts.reflected_trial import evaluate_depth, fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import physical
from scripts.validate_gpu_surface_error import asset_revision


def evaluate(args, baseline, *, source_type=QuadraticSourceLightCurve, source_options=None, rule_type=None):
    if args.backend == "gpu" and not args.verified_bounded_gpu_window:
        raise ValueError("GPU synthesis requires a verified bounded unit")
    if baseline["source_revision"] != asset_revision() or baseline["engine"] != "emitting-area":
        raise ValueError("Expected the original reflected fixture with matching table revision")
    props = baseline["samples"][0]["rows"][0]["physical"]
    fit, _ = fixture()
    vector = np.array(parameters.vector_normalizer([props[k] for k in fit.fitable], fit.fitable, fit.normalization))
    for band, values in fit.y_data.items():
        np.testing.assert_array_equal(values, baseline["observations"][band])
        np.testing.assert_array_equal(fit.y_err[band], baseline["uncertainties"][band])
        np.testing.assert_array_equal(fit.x_data_reducer[band], baseline["reducers"][band])
    measured = {}

    class ProfiledSource(source_type):
        def batch(self, *args, **kwargs):
            batch = super().batch(*args, **kwargs)
            result = batch.results[0]
            measured["batch"] = batch.timings
            measured["curve"] = result.timings
            measured["fields"] = {
                key: {name: component["reconstructed_field"] for name, component in components.items()}
                for key, components in result.diagnostics["ancestry_controls"].items()
            }
            return batch

    def observer(bands, rule, *, device):
        if args.boundary == "refined":
            rule = replace(rule, slice_order=12, junction_depth=4)
        if rule_type is not None:
            rule = rule_type(**asdict(rule))
        return ProfiledSource(bands, rule, device=device, **(source_options or {}))

    result = evaluate_depth(fit, vector, args.depth, args.backend, observer_type=observer)
    for p in result["profiles"]:
        row = p["samples"][0]["rows"][0]
        if row["physical"] != props or p["physical_constants"] != baseline["physical_constants"]:
            raise ValueError("Unmatched physical inputs")
        rule = row["finite_metadata"]["curvature_field3"]["rule"]
        p.update(slice_order=rule["slice_order"], junction_depth=rule["junction_depth"])
    result["measured"] = measured
    return result


def assess(parts, *, engine="quadratic-spectral-source", interior_degree=None):
    if engine.startswith("symmetric-") != (interior_degree in (4, 5)):
        raise ValueError("Select the matching finite interior rule")
    fit, _ = fixture()
    profiles = [p for part in parts for p in part["profiles"]]
    first = profiles[0]
    row0 = first["samples"][0]["rows"][0]
    props = row0["physical"]
    vector = np.asarray(parameters.vector_normalizer([props[k] for k in fit.fitable], fit.fitable, fit.normalization))
    if physical(fit, vector) != props:
        raise ValueError("Physical proposal differs from original constraint reconstruction")
    prior = float(fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas))
    if not np.isfinite(prior):
        raise ValueError("Only in-prior physical comparisons are supported")
    rays, grid, densities = {}, {}, {}
    common_rule = None
    for p in profiles:
        for k in (
            "engine",
            "physical_constants",
            "source_revision",
            "backend",
            "order",
            "slice_order",
            "junction_depth",
            "levels",
            "receiver_points",
            "integration_level",
            "irradiation",
            "observations_irradiated",
            "observations",
            "reducers",
            "uncertainties",
        ):
            if p[k] != first[k]:
                raise ValueError(f"Unmatched {k} in refinement grid")
        if (
            p["engine"] != engine
            or p["source_revision"] != asset_revision()
            or not p["irradiation"]
            or not p["observations_irradiated"]
            or p["batch"] != 1
            or p["jd"]
            or len(p["samples"]) != 1
            or len(p["samples"][0]["rows"]) != 1
            or p["levels"] != [2, 3]
        ):
            raise ValueError("Expected one complete reflected six-node proposal")
        row = p["samples"][0]["rows"][0]
        d, cap = p["depth"], p["donor_level"]
        if d not in (3, 4) or cap not in (1, 2) or row["physical"] != props:
            raise ValueError("Unmatched resolution or trial")
        trace = (row["quadrature_sha256"], row["phase_records"])
        if rays.setdefault(d, trace) != trace:
            raise ValueError("Emitter contrast needs identical observer packets")
        for band, obs in fit.y_data.items():
            for a, b in (
                (p["observations"][band], obs),
                (p["uncertainties"][band], fit.y_err[band]),
                (p["reducers"][band], fit.x_data_reducer[band]),
            ):
                np.testing.assert_array_equal(a, b)
        for key in row["curves"]:
            c = curve(row, key)
            rule = dict(c.rule)
            if (
                rule.get("interior_degree") != interior_degree
                or c.field_levels not in (2, 3)
                or rule.pop(f"{engine.replace('-', '_')}_receiver3_donor_cap{cap}", 0) != 1
                or rule.pop("depth") != d
                or rule["slice_order"] != p["slice_order"]
                or rule["junction_depth"] != p["junction_depth"]
                or rule["order"] != 3
                or rule["integration_level"] != 3
                or p["receiver_points"] != 3
                or (rule["slice_order"], rule["junction_depth"]) not in ((8, 3), (12, 4))
            ):
                raise ValueError("Invalid six-node curve rule")
            if common_rule is None:
                common_rule = rule
            if common_rule != rule:
                raise ValueError("Changed a fixed integration setting")
            idx = (FIELD if c.field_levels == 2 else 0) | (DONOR if cap == 1 else 0) | (DEPTH if d == 3 else 0)
            if idx in grid:
                raise ValueError("Repeated finite corner")
            grid[idx] = c
            pred, densities[idx] = mapped_density(fit, props, prior, c)
            if c.field_levels == 3:
                for b, values in pred.items():
                    np.testing.assert_allclose(values, row["prediction"][b], rtol=3e-13, atol=2e-12)
                np.testing.assert_allclose(densities[idx], row["posterior"], rtol=0, atol=1e-6)
    if set(grid) != set(range(8)):
        raise ValueError("Incomplete finite grid")
    # This diagnostic reports failed finite grids without returning a usable inference density.
    control = check(grid, first["uncertainties"], first["observations"], first["reducers"], require_budget=False)
    return dict(
        physical=props,
        finite_control=control,
        corner_log_posteriors=densities,
        usable_log_posterior=densities[0] if control["within_finite_budget"] else None,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("evaluate", "assess"))
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--boundary", choices=("original", "refined"), default="refined")
    p.add_argument("--depth", type=int, choices=(3, 4))
    p.add_argument("--backend", choices=("reference", "cpu", "gpu"), default="reference")
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    docs = [json.loads(v.read_bytes()) for v in args.inputs]
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        if args.command == "evaluate":
            if len(docs) != 1 or args.depth is None:
                p.error("evaluate requires one original profile and a depth")
            result = evaluate(args, docs[0])
        else:
            result = assess(docs)
    args.output.write_text(json.dumps(result, allow_nan=False, default=lambda a: a.tolist()) + "\n")
