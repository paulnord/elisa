"""Replay compact-target transitions and compare all physical curves across worker pools."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from elisa import settings
from scripts import sample_reflected_binary as sampler
from scripts.assess_walker_chains import OriginalFacet, coordinates
from scripts.compact_cohort import replay
from scripts.compact_interior_fit import compact_inputs
from scripts.compare_stationary_eclipse import compare, read_grid
from scripts.validate_gpu_atmosphere import table_settings


def paired(a, out_a, b, out_b, *, facets=()):
    x, y = replay(a, out_a), replay(b, out_b)
    for key in ("seed", "target", "scales", "position", "rng", "step"):
        if x["start"][key] != y["start"][key]:
            raise ValueError(f"Compared streams have different starting {key}")
    for key in ("position", "rng", "step"):
        if x["selected"][key] != y["selected"][key]:
            raise ArithmeticError(f"Selected {key} differs across implementations")
    for section in ("start", "selected"):
        np.testing.assert_allclose(x[section]["density"], y[section]["density"], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(x["proposed"], y["proposed"])
    if (x["status"], x["log_u"], x["prior"]) != (y["status"], y["log_u"], y["prior"]):
        raise ArithmeticError("Different resolution or sampler decision for the same proposal")
    controls = {}
    if np.isfinite(x["prior"]):
        for rule, depth in product(("six", "duffy"), (3, 4)):
            controls[rule + str(depth)] = compare(
                read_grid(out_a, a, False, rule, depth, compact=True),
                read_grid(out_b, b, False, rule, depth, compact=True),
            )
    row = dict(
        seed=x["start"]["seed"],
        step=x["start"]["step"] + 1,
        status=x["status"],
        log_uniform=x["log_u"],
        proposal=x["proposed"],
        comparisons=controls,
        selected=coordinates(x["fit"], x["selected"]["position"]),
        start=coordinates(x["fit"], x["start"]["position"]),
        density=x["selected"]["density"],
        margins=x["margins"],
    )
    if np.isfinite(x["prior"]):
        row["proposed_coordinates"] = coordinates(x["fit"], x["proposed"])
        row["original_facets"] = {
            str(d): f.decision(x["start"]["position"], x["proposed"], x["log_u"]) for d, f in facets
        }
    e, f = x["evidence"], y["evidence"]
    if (e is None) != (f is None):
        raise ArithmeticError("Implementations disagree on likelihood availability")
    if e is not None:
        for rule in ("six", "duffy"):
            for i in map(str, range(8)):
                np.testing.assert_allclose(
                    e["corner_log_posteriors"][rule][i], f["corner_log_posteriors"][rule][i], atol=1e-6, rtol=0
                )
        for k in ("distance_sigma", "log_likelihood_bound", "max_measured_nominal_log_likelihood_change"):
            np.testing.assert_allclose(e["report"][k], f["report"][k], atol=1e-6, rtol=0)
            row[k] = e["report"][k]
        if x["margins"] is not None:
            np.testing.assert_allclose(x["margins"], y["margins"], atol=2e-6, rtol=0)
        row["proposed_density"] = e["report"]["usable_log_posterior"]
    return row


def checked_paths(root):
    fresh = (root / "pool.json").exists()
    pool = sampler.read_record(root / ("pool.json" if fresh else "resumed-pool.json"))
    starts = sorted(root.glob("walker-*/start.json"))
    meta = [sampler.read_record(p.parent / "draw/result.json") for p in starts]
    if not starts or meta != pool["rows"] or len({row["seed"] for row in meta}) != len(starts):
        raise ValueError("Incomplete worker pool")
    if fresh and (
        pool.get("evaluated", len(starts)) != len(starts) or len({r["pid"] for r in meta}) != pool["workers"]
    ):
        raise ValueError("The requested worker count did not evaluate this complete cohort")
    return starts


def compare_pools(left, right, *, native_facets=False):
    a, b = checked_paths(left), checked_paths(right)
    fit = compact_inputs(a[0], False)[1]
    facets = [(d, OriginalFacet(d, fit)) for d in (12, 6)] if native_facets else []
    return [paired(p, p.parent / "draw", q, q.parent / "draw", facets=facets) for p, q in zip(a, b, strict=True)]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--original-facets", action="store_true")
    p.add_argument("--export-selected", type=Path, help="Save matching reference/GPU states for the next generation")
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        rows = compare_pools(args.left, args.right, native_facets=args.original_facets)
        sampler.write_record(args.output, rows)
        if args.export_selected:
            selected = {}
            for root in (args.left, args.right):
                states = [sampler.read_record(f.parent / "draw/state.json") for f in checked_paths(root)]
                selected[states[0]["backend"]] = states
            if set(selected) != {"reference", "gpu"}:
                raise ValueError("Export expects matched reference and GPU states")
            sampler.write_record(args.export_selected, selected)
