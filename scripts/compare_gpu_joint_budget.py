"""Joint finite controls from three matched complete validation-driver outputs.

Anchor: fine field pair, depth2/slice8. Alternatives: depth2/slice4 and
depth1/slice8, with the same common integration grid and physical proposals.
No device initialization or synthesis occurs here. Full synthesis costs reside
in each input; this postprocessing is not a likelihood-throughput benchmark.
"""

# ruff: noqa: INP001, S101, T201, PLR2004 - scientific standalone driver
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa.experimental.gpu_curved import CurvedCurveResult
from elisa.experimental.gpu_joint_budget import JointInferenceBudget
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError


def curve(row: dict, key: str) -> CurvedCurveResult:
    meta = dict(row["finite_metadata"][key])
    meta["phases"] = np.asarray(meta["phases"])
    values = {attr: {band: np.asarray(v) for band, v in bands.items()} for attr, bands in row["curves"][key].items()}
    return CurvedCurveResult(**meta, **values, timings={}, diagnostics={})


def compare(anchor: dict, slice_alternative: dict, depth_alternative: dict) -> list[dict]:
    for alternative in [slice_alternative, depth_alternative]:
        for key in [
            "backend",
            "batch",
            "jd",
            "levels",
            "integration_level",
            "source_revision",
            "order",
            "observations",
            "uncertainties",
            "reducers",
        ]:
            assert anchor[key] == alternative[key], key
        assert len(anchor["samples"]) == len(alternative["samples"])
    assert (anchor["depth"], anchor["slice_order"]) == (2, 8)
    assert (slice_alternative["depth"], slice_alternative["slice_order"]) == (2, 4)
    assert (depth_alternative["depth"], depth_alternative["slice_order"]) == (1, 8)
    rows = []
    for iteration, samples in enumerate(
        zip(anchor["samples"], slice_alternative["samples"], depth_alternative["samples"], strict=True),
    ):
        for proposal, variants in enumerate(zip(*(s["rows"] for s in samples), strict=True)):
            a, s, d = variants
            assert a["physical"] == s["physical"] == d["physical"]
            keys = sorted(a["finite_metadata"], key=lambda k: a["finite_metadata"][k]["field_levels"])
            assert len(keys) == 2
            coarse, fine = keys
            for r in variants:
                assert r["finite_metadata"][fine]["field_levels"] == a["finite_metadata"][fine]["field_levels"]
            reference = curve(a, fine)
            alternatives = {
                "thermal_radiance_field": curve(a, coarse),
                "slice_order": curve(s, fine),
                "boundary_depth": curve(d, fine),
            }
            kwargs = {"observed": anchor["observations"], "reducers": anchor["reducers"], "mean_normalize": True}
            budget = JointInferenceBudget()
            report = budget.compare(reference, alternatives, anchor["uncertainties"], **kwargs, require_budget=False)
            stopped = False
            try:
                budget.compare(reference, alternatives, anchor["uncertainties"], **kwargs)
            except ResolutionNotQualifiedError:
                stopped = True
            assert stopped != report["within_finite_budget"]
            rows.append(
                {
                    "iteration": iteration,
                    "proposal": proposal,
                    "physical": a["physical"],
                    "accuracy_abort": stopped,
                    "joint": report,
                },
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--slice-alternative", type=Path, required=True)
    parser.add_argument("--depth-alternative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.anchor, args.slice_alternative, args.depth_alternative]
    data = [json.loads(p.read_text()) for p in paths]
    rows = compare(*data)
    report = {
        "rows": rows,
        "backend": data[0]["backend"],
        "input_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract": "Joint nominal Gaussian control on three computed finite changes about one anchor. "
        "No uncomputed interaction/combination, continuum, fit/MCMC or observational adequacy claim.",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "backend": report["backend"],
                "rows": len(rows),
                "joint_likelihood_bounds": [r["joint"]["finite_log_likelihood_absolute_bound"] for r in rows],
            },
        ),
    )


if __name__ == "__main__":
    main()
