"""Compare quadratic emitter positions at fixed physical parameters over the eight finite models."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.analytics.params import parameters
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.quadratic_reflection import PairedQuadraticLightCurve
from scripts.compare_gpu_joint_budget import curve
from scripts.compare_reflection_resolution import check, DEPTH, DONOR, FIELD
from scripts.reflected_trial import evaluate_depth, fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_surface_error import asset_revision


def evaluate(args, baseline):
    if args.backend == "gpu" and not args.verified_bounded_gpu_window:
        raise ValueError("Use a verified bounded GPU unit")
    if baseline["engine"] != "emitting-area" or baseline["source_revision"] != asset_revision():
        raise ValueError("Select a reference emitting-area profile from the same fixture")
    fit, _ = fixture()
    original = baseline["samples"][0]["rows"][0]
    props = original["physical"]
    vector = np.array(parameters.vector_normalizer([props[k] for k in fit.fitable], fit.fitable, fit.normalization))
    result = evaluate_depth(fit, vector, args.depth, args.backend, observer_type=PairedQuadraticLightCurve)
    for p in result["profiles"]:
        row = p["samples"][0]["rows"][0]
        if row["physical"] != props or p["physical_constants"] != baseline["physical_constants"]:
            raise ValueError("Changed physical inputs")
        for key in ("observations", "reducers", "uncertainties"):
            for band, v in p[key].items():
                np.testing.assert_array_equal(v, baseline[key][band])
        if p["depth"] == baseline["depth"]:
            # Match every adaptive decision across the old and new irradiation models.
            if row["phase_records"] != original["phase_records"]:
                raise ValueError("Observer refinement changed with the irradiation model")
    return result


def assess(parts):
    grid = {}
    profiles = [p for part in parts for p in part["profiles"]]
    first = profiles[0]
    source = first["samples"][0]["rows"][0]
    per_depth = {}
    for p in profiles:
        row = p["samples"][0]["rows"][0]
        for key in (
            "engine",
            "backend",
            "source_revision",
            "physical_constants",
            "observations",
            "reducers",
            "uncertainties",
            "levels",
            "order",
            "integration_level",
            "receiver_points",
            "slice_order",
            "irradiation",
            "observations_irradiated",
        ):
            if first[key] != p[key]:
                raise ValueError(f"Mismatched {key}")
        if p["engine"] != "quadratic-emitting-area" or row["physical"] != source["physical"]:
            raise ValueError("Mismatched finite quadratic proposal")
        meta = (row["phase_records"], row["quadrature_sha256"])
        if per_depth.setdefault(p["depth"], meta) != meta:
            raise ValueError("The two donor resolutions require identical eclipse packets")
        for key in row["curves"]:
            c = curve(row, key)
            marker = f"quadratic_emitting_area_receiver3_donor_cap{p['donor_level']}"
            if (
                c.rule.get(marker) != 1
                or c.rule["depth"] != p["depth"]
                or c.field_levels not in (2, 3)
                or p["depth"] not in (3, 4)
                or p["donor_level"] not in (1, 2)
            ):
                raise ValueError("Wrong corner in the quadratic resolution grid")
            index = (
                (DEPTH if p["depth"] == 3 else 0)
                | (DONOR if p["donor_level"] == 1 else 0)
                | (FIELD if c.field_levels == 2 else 0)
            )
            if index in grid:
                raise ValueError("Duplicate corner in the quadratic resolution grid")
            grid[index] = c
    if set(grid) != set(range(8)):
        raise ValueError("Incomplete quadratic resolution grid")
    try:
        budget = check(grid, first["uncertainties"], first["observations"], first["reducers"])
    except ResolutionNotQualifiedError as exc:
        budget = exc.report
    return dict(physical=source["physical"], finite_control=budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "assess"))
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(3, 4))
    parser.add_argument("--backend", choices=("reference", "cpu", "gpu"), default="reference")
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    args = parser.parse_args()
    docs = [json.loads(p.read_bytes()) for p in args.inputs]
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        if args.command == "evaluate":
            if len(docs) != 1 or args.depth is None:
                parser.error("evaluate requires one baseline and a depth")
            result = evaluate(args, docs[0])
        else:
            result = assess(docs)
    args.output.write_text(json.dumps(result, allow_nan=False, default=lambda a: a.tolist()) + "\n")
