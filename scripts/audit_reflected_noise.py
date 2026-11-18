"""Measure how assumed time-correlated residuals change densities on the guarded physical path."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from scripts.reflected_trial import fixture
from scripts.sample_reflected_binary import part_path, read_record
from scripts.validate_gpu_atmosphere import table_settings


def evaluate(fit, profile, reported, rho, phase_length):
    row = profile["samples"][0]["rows"][0]
    if profile["jd"] or profile["depth"] != 4 or profile["donor_level"] != 2 or row["physical"] != reported["physical"]:
        raise ValueError("Use the same finest model and phase-coordinate proposal")
    density = reported["prior"]
    band_stats = {}
    for band, pred in row["prediction"].items():
        np.testing.assert_array_equal(profile["observations"][band], fit.y_data[band])
        np.testing.assert_array_equal(profile["uncertainties"][band], fit.y_err[band])
        phase = np.asarray(fit.x_data[band])
        scale = fit.y_err[band] * np.hypot(1.0, np.exp(row["physical"]["nuisance@ln_f"]))
        r = (fit.y_data[band] - np.asarray(pred)) / scale
        corr = rho * np.exp(-np.abs(phase[:, None] - phase) / phase_length) + (1 - rho) * np.eye(len(r))
        lower = np.linalg.cholesky(corr)
        whitened = np.linalg.solve(lower, r)
        density -= 0.5 * (whitened @ whitened + len(r) * np.log(2 * np.pi))
        density -= np.log(np.diag(lower)).sum() + np.log(scale).sum()
        band_stats[band] = dict(
            count=len(r), rms_sigma=float(np.sqrt(np.mean(r * r))), whitened_rms=float(np.sqrt(np.mean(whitened**2)))
        )
    return dict(log_posterior=float(density), bands=band_stats)


def assess(path, initial, rho, length):
    if not 0 <= rho < 1 or not np.isfinite(length) or length <= 0:
        raise ValueError("Select a correlation fraction in [0,1) and a positive finite phase length")
    state = read_record(path)
    fit, _ = fixture()

    def result(part, report):
        profiles = part["profiles"] if "profiles" in part else [part]
        fine = [p for p in profiles if p["donor_level"] == 2 and p["depth"] == 4]
        if len(fine) != 1:
            raise ValueError("Supply one depth-4 profile with donor level 2")
        profile = fine[0]
        np.testing.assert_allclose(
            evaluate(fit, profile, report, 0, length)["log_posterior"],
            report["usable_log_posterior"],
            rtol=0,
            atol=1e-6,
        )
        return evaluate(fit, profile, report, rho, length)

    start = result(initial, state["initial"])
    current = start["log_posterior"]
    rows = []
    for event in state["events"]:
        if event["report"] is None:
            continue
        d = result(read_record(part_path(path.parent, event["step"], 4)), event["report"])
        delta = d["log_posterior"] - current
        rows.append(
            dict(
                step=event["step"],
                delta_on_guarded_path=delta,
                conditional_accept=bool(event["log_u"] < min(0.0, delta)),
                **d,
            )
        )
        if event["accepted"]:
            current = d["log_posterior"]
    return dict(rho=rho, unwrapped_phase_length=length, initial=start, conditional_trials=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("initial_depth4", type=Path)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--phase-length", type=float, default=0.01)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        d = assess(args.checkpoint, json.loads(args.initial_depth4.read_bytes()), args.rho, args.phase_length)
    args.output.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
