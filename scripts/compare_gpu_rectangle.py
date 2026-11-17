"""Assess a computed field/depth rectangle from matching complete driver outputs."""

# ruff: noqa: INP001, S101, T201, PLR2004 - scientific postprocessing, no device work
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_rectangle_budget import RectangleInferenceBudget
from scripts.compare_gpu_joint_budget import curve


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deep", type=Path, required=True)
    p.add_argument("--shallow", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = [args.deep, args.shallow]
    deep, shallow = [json.loads(v.read_text()) for v in paths]
    for key in [
        "backend",
        "batch",
        "jd",
        "levels",
        "integration_level",
        "source_revision",
        "order",
        "slice_order",
        "observations",
        "uncertainties",
        "reducers",
        "engine",
        "iteration_start",
        "physical_constants",
    ]:
        assert deep[key] == shallow[key], key
    assert deep["depth"] > shallow["depth"]
    assert len(deep["samples"]) == len(shallow["samples"])
    rows = []
    for iteration, (x, y) in enumerate(zip(deep["samples"], shallow["samples"], strict=True)):
        for proposal, (a, b) in enumerate(zip(x["rows"], y["rows"], strict=True)):
            assert a["physical"] == b["physical"]
            keys = sorted(a["finite_metadata"], key=lambda k: a["finite_metadata"][k]["field_levels"])
            assert len(keys) == 2
            coarse, fine = keys
            corners = [curve(a, fine), curve(a, coarse), curve(b, fine), curve(b, coarse)]
            budget = RectangleInferenceBudget()
            kwargs = {"observed": deep["observations"], "reducers": deep["reducers"], "mean_normalize": True}
            control = budget.compare(*corners, deep["uncertainties"], **kwargs, require_budget=False)
            stopped = False
            try:
                budget.compare(*corners, deep["uncertainties"], **kwargs)
            except ResolutionNotQualifiedError:
                stopped = True
            assert stopped != control["within_finite_budget"]
            rows.append(
                {
                    "iteration": iteration + deep["iteration_start"],
                    "proposal": proposal,
                    "physical": a["physical"],
                    "control": control,
                    "accuracy_abort": stopped,
                },
            )
    report = {
        "backend": deep["backend"],
        "engine": deep["engine"],
        "rows": rows,
        "input_sha256": [hashlib.sha256(v.read_bytes()).hexdigest() for v in paths],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract": (
            "Four computed finite field/depth corners. No continuum, uncomputed interaction, posterior or ESS claim."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "backend": deep["backend"],
                "combined_likelihood_bounds": [r["control"]["computed_combination_log_likelihood_bound"] for r in rows],
            },
        ),
    )


if __name__ == "__main__":
    main()
