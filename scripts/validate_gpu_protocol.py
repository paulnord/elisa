"""Direct-observation and empirical mesh-qualified physical fitting assessment.

GPU commands require an independently restored bounded systemd allocation.
Qualification costs are included; finite references do not certify continuum
accuracy. The high-precision failed check is an expected protocol stop.
"""

# ruff: noqa: INP001, S101, T201 - standalone scientific driver/bootstrap
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.experimental.gpu_protocol import MeshResolutionCheck, ResolutionNotQualifiedError
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import compare, fit_assessment, make_problem, normalized_vector, physical, prepare_fit


def protocol_problem(*, sigma: float = 0.05, sampling: str = "uniform") -> Any:
    """Generate a separate, explicitly low-precision positive-control dataset."""
    problem = make_problem(sampling=sampling)
    # Recover the native direct CPU truth before adding the declared noise.
    truth_fit = prepare_fit(problem)
    truth_fit.fit_xs, truth_fit.interp_treshold = None, 1000
    truth = truth_fit.predict_light_curves(problem.truth)
    rng = np.random.default_rng(909)
    for band, data in problem.data.items():
        data.y_data = truth[band]+rng.normal(0, sigma, len(truth[band]))
        data.y_err = np.full(len(truth[band]), sigma)
    return problem


def direct_mapping(device: str) -> list[dict[str, Any]]:
    rows = []
    for jd, preserve in [(False, False), (True, False), (False, True)]:
        with settings.override(PRESERVE_INPUT_PHASES=preserve):
            problem = make_problem(jd=jd, sampling="uniform", free_nuisance=True, shape_parameters=True)
            original = prepare_fit(problem)
            direct_cpu = prepare_fit(problem, phase_mode="observations")
            hybrid = prepare_fit(problem, device=device, phase_mode="observations")
            original.fit_xs, original.interp_treshold = None, len(original.x_data_reduced)+1
            base = normalized_vector(original)
            row = {"jd": jd, "preserve_phases": preserve, "candidates": []}
            for vector in [base, np.full(base.shape, 0.65), np.linspace(0.25, 0.7, len(base))]:
                props = physical(original, vector)
                expected = original.predict_light_curves(props)
                a, b = direct_cpu.predict_light_curves(props), hybrid.predict_light_curves(props)
                row["candidates"].append({
                    "parameters": props,
                    "cpu_protocol_vs_direct_original": {k: compare(a[k], expected[k]) for k in expected},
                    "hybrid_vs_direct_original": {k: compare(b[k], expected[k]) for k in expected},
                    "posterior": compare(hybrid.ln_probability(vector), direct_cpu.ln_probability(vector),
                                         atol=1e-6, rtol=0),
                    "execution": hybrid.hybrid_synthesis.last_evaluation,
                })
            native = prepare_fit(problem)
            direct = direct_cpu.predict_light_curves(physical(direct_cpu, base))
            legacy = native.predict_light_curves(physical(native, base))
            row["native_support_difference_max_abs"] = {k: float(np.max(np.abs(direct[k]-legacy[k]))) for k in direct}
            rows.append(row)
    return rows


def qualification(device: str) -> dict[str, Any]:
    check = MeshResolutionCheck()
    strict = make_problem(sampling="uniform")
    failures = []
    with settings.override(PENALIZE_ALL_EXCEPTIONS=True):
        for requested in [None, device]:
            for kind in ["lsq", "mcmc"]:
                fit = prepare_fit(strict, kind=kind, device=requested,
                                  phase_mode="observations", resolution_check=check)
                vector = normalized_vector(fit)
                try:
                    (fit.model_to_fit(vector) if kind == "lsq" else fit.ln_probability(vector))
                except ResolutionNotQualifiedError as exc:
                    failures.append({"backend": "original_cpu" if requested is None else "hybrid",
                                     "callback": kind, "report": exc.report})
                else:
                    msg = "The known high-precision mesh failure was hidden by candidate penalties"
                    raise AssertionError(msg)
                assert not fit._protocol_active  # noqa: SLF001
    problem = protocol_problem()
    original = prepare_fit(problem, phase_mode="observations", resolution_check=check)
    hybrid = prepare_fit(problem, device=device, phase_mode="observations", resolution_check=check)
    base = normalized_vector(original)
    predictions = []
    for vector in [base, np.full(base.shape, 0.65), np.linspace(0.35, 0.65, len(base))]:
        props = physical(original, vector)
        a, b = original.predict_light_curves(props), hybrid.predict_light_curves(props)
        report = deepcopy(hybrid.last_resolution_check)
        predictions.append({"parameters": props, "qualification": report,
                            "parity": {k: compare(b[k], a[k]) for k in a},
                            "posterior": compare(hybrid.ln_probability(vector), original.ln_probability(vector),
                                                 atol=1e-6, rtol=0)})
        assert report["status"] == "qualified"
    costs = {}
    for label, requested, enabled in [("direct_original_cpu", None, False), ("direct_hybrid", device, False),
                                       ("qualified_original_cpu", None, True), ("qualified_hybrid", device, True)]:
        started = perf_counter()
        fit = prepare_fit(problem, device=requested, phase_mode="observations",
                          resolution_check=check if enabled else None)
        setup = perf_counter()-started
        started = perf_counter()
        fit.ln_probability(base)
        first = perf_counter()-started
        samples, reports = [], []
        for fraction in [0.0, 0.02, 0.05]:
            vector = (1-fraction)*base+fraction*np.linspace(0.3, 0.7, len(base))
            started = perf_counter()
            fit.ln_probability(vector)
            samples.append(perf_counter()-started)
            reports.append(deepcopy(fit.last_resolution_check))
        costs[label] = {"setup_seconds": setup, "first_callback_seconds": first,
                        "samples_seconds": samples, "median_seconds": float(np.median(samples)),
                        "qualification_reports": reports}
    assessment = fit_assessment(device, problem=problem, phase_mode="observations", resolution_check=check)
    return {"strict_sigma1e4_failures": failures, "low_precision_sigma": 0.05,
            "predictions": predictions, "costs_including_all_cpu_checks": costs,
            "public_fit_assessment": assessment,
            "scientific_limit": "5% assigned noise and matching synthetic noise is a separate positive control, "
            "not a relaxed 1e-4 criterion. Reference meshes are empirical; no continuum certificate."}


def run(device: str) -> dict[str, Any]:
    started = perf_counter()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        mapping = direct_mapping(device)
        qualified = qualification(device)
    return {"device": device, "all_correctness_gates_passed": True, "direct_mapping": mapping,
            "qualification": qualified, "wall_seconds": perf_counter()-started,
            "contract": "Explicit original observation support; fresh original CPU target/two refinements per "
            "qualified callback outside penalty boundary, then original CPU/hybrid target evaluation. "
            "All complete costs include qualification and transfers; no physical cache or MCMC convergence claim."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.device)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("device", "all_correctness_gates_passed", "wall_seconds")}))


if __name__ == "__main__":
    main()
