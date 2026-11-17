"""Noise-scaled mesh/phase/spectral audit of complete reflected curves and fits.

Finite reference grids are not continuum truth. Run GPU only in an independent
bounded restored allocation. Original CPU and device engines remain unchanged.
"""

# ruff: noqa: INP001, T201 - standalone scientific validation/bootstrap
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

from elisa import Observer, const, settings
from elisa.analytics.models import lc
from elisa.experimental.gpu_accuracy import BlackbodySpectralProbe, resolution_error
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from scripts.validate_gpu_atmosphere import BANDS, table_settings
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_binary import contact_phases, fixture, measurement
from scripts.validate_gpu_fit import fit_assessment, make_problem, prepare_fit

MESH_LEVELS = (12, 8, 6, 4, 3)


def audit(a: dict, b: dict, sigma: Any = 1e-4, *, observed: dict | None = None) -> dict[str, Any]:
    """Per-band and joint errors with equal, explicit noise and original ln_f."""
    bands = list(a)
    errors = sigma if isinstance(sigma, dict) else dict.fromkeys(bands, sigma)
    return {
        "bands": {
            k: resolution_error(a[k], b[k], errors[k], observed=None if observed is None else observed[k], ln_f=-2)
            for k in bands
        },
        "joint": resolution_error(
            np.concatenate([a[k] for k in bands]),
            np.concatenate([b[k] for k in bands]),
            np.concatenate([np.broadcast_to(errors[k], np.shape(a[k])) for k in bands]),
            observed=None if observed is None else np.concatenate([observed[k] for k in bands]),
            ln_f=-2,
        ),
    }


def mesh_curves(device: str) -> list[dict[str, Any]]:
    engine = HybridBinaryLightCurve(list(BANDS), device=device)
    out = []
    for name, inclination, atlas in [("partial_bb", 85, "bb"), ("total_bb", 90, "bb"), ("partial_ck04", 88, "ck04")]:
        original_curves, rows = [], []
        for disc in MESH_LEVELS:
            binary = (
                fixture({"inclination": inclination, "discretization": disc})
                if atlas == "bb"
                else table_fixture({"inclination": inclination})
            )
            if atlas == "ck04":
                binary.primary.discretization_factor = binary.secondary.discretization_factor = np.radians(disc)
            phases = contact_phases(binary)
            started = perf_counter()
            expected = Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)[1]
            cpu_seconds = perf_counter() - started
            started = perf_counter()
            result = engine(binary, phases, normalize=True)
            first = perf_counter() - started
            parity = {b: measurement(result.fluxes[b], expected[b], atol=2e-12) for b in BANDS}
            costs = {"original_cpu": [], "hybrid": []}
            repeats = 3 if disc in (12, 6) else 1
            for _ in range(repeats):
                started = perf_counter()
                Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)
                costs["original_cpu"].append(perf_counter() - started)
                started = perf_counter()
                warm = engine(binary, phases, normalize=True)
                costs["hybrid"].append(perf_counter() - started)
            original_curves.append(expected)
            rows.append(
                {
                    "discretization_deg": disc,
                    "observations_per_band": len(phases),
                    "parity": parity,
                    "backend": result.backend,
                    "fallback_reason": result.fallback_reason,
                    "first_seconds": {"original_cpu": cpu_seconds, "hybrid": first},
                    "samples_seconds": costs,
                    "median_seconds": {k: float(np.median(v)) for k, v in costs.items()},
                    "warm_stages": warm.timings, "timing_repeats": repeats,
                },
            )
        for i, row in enumerate(rows):
            row["vs_finest_3deg"] = audit(original_curves[i], original_curves[-1])
            if i:
                row["change_from_previous"] = audit(original_curves[i - 1], original_curves[i])
        out.append(
            {
                "name": name,
                "levels": rows,
                "reference": "3 degree original CPU finite mesh, native quadrature",
                "limitation": "Pair changes are empirical; no monotone convergence or continuum error bound assumed",
            },
        )
    return out


def prediction(fit: Any, props: dict, disc: float, support: Any = None, *, synthesis: Any = None) -> dict:
    """Use original per-band mapping/normalization on an owned diagnostic fitter."""
    fit.discretization, fit.fit_xs = disc, support
    return fit.predict_light_curves(props, **({} if synthesis is None else {"synthesis": synthesis}))


def phase_curves(device: str) -> dict[str, Any]:
    problem = make_problem()
    original, hybrid = prepare_fit(problem), prepare_fit(problem, device=device)
    output = {}
    for l3 in [0.0, 0.2]:
        props = dict(problem.truth, **{"system@additional_light": l3})
        direct = prediction(original, props, 8)
        rows = []
        for count in [32, 64, 128, 256, 1024, 8192]:
            support = np.linspace(-0.1, 1.1, count)
            started = perf_counter()
            expected = prediction(original, props, 8, support)
            cpu_seconds = perf_counter() - started
            started = perf_counter()
            actual = prediction(hybrid, props, 8, support)
            gpu_seconds = perf_counter() - started
            rows.append(
                {
                    "support_points": count,
                    "native_vs_direct": audit(expected, direct),
                    "parity": {b: measurement(actual[b], expected[b], atol=2e-12) for b in expected},
                    "seconds": {"original_cpu": cpu_seconds, "hybrid_first": gpu_seconds},
                    "execution": hybrid.hybrid_synthesis.last_evaluation,
                },
            )
        for i in range(1, len(rows)):
            a = prediction(original, props, 8, np.linspace(-0.1, 1.1, rows[i - 1]["support_points"]))
            b = prediction(original, props, 8, np.linspace(-0.1, 1.1, rows[i]["support_points"]))
            rows[i]["change_from_previous"] = audit(a, b)
        output[str(l3)] = {
            "levels": rows,
            "interpretation": "Pure interpolation control"
            if l3 == 0
            else "Original support-dependent third-light mean AND interpolation; not pure interpolation error",
        }
    return output


def spectral_curves() -> dict[str, Any]:
    binary = fixture({"inclination": 85, "discretization": 8})
    observer = Observer(passband=list(BANDS), system=binary)
    phases = contact_phases(binary)
    original = observer.lc(phases=phases, normalize=True)[1]
    curves, spectra = [], []
    rows = []
    temperatures = np.array([3000.0, 4800.0, 6500.0, 10000.0, 30000.0])
    for order in [8, 16, 32]:
        probe = BlackbodySpectralProbe(observer.passband, order=order)
        started = perf_counter()
        curves.append(probe.observe(observer, phases)["fluxes"])
        seconds = perf_counter() - started
        spectra.append(probe.surface_flux(temperatures))
        rows.append(
            {
                "gauss_order": order,
                "curve_seconds_cpu_diagnostic": seconds,
                "native_vs_probe": audit(original, curves[-1]),
                "quadrature_nodes": {b: len(v[0]) for b, v in probe.quadrature.items()},
            },
        )
    control = probe.observe(observer, phases, native_grid_control=True)["fluxes"]
    for i, row in enumerate(rows):
        row["probe_vs_order32"] = audit(curves[i], curves[-1])
    # Full original physical serializer/mapping/mean normalization, spectral ablation only.
    problem = make_problem(free_nuisance=True)
    fit = prepare_fit(problem)
    complete_probe = BlackbodySpectralProbe(fit.observer.passband, order=32)

    def synthesis(phases: Any, discretization: float, observer: Any, **props: Any) -> dict:
        observer._system = lc.prepare_binary(discretization=discretization, **props)  # noqa: SLF001
        return complete_probe.observe(observer, phases)["fluxes"]

    native = prediction(fit, problem.truth, 8)
    spectral = prediction(fit, problem.truth, 8, synthesis=synthesis)
    exact_sigma = 2 * const.PI**5 * const.BOLTZMAN_CONST**4 / (15 * const.PLANCK_CONST**3 * const.C**2)
    return {
        "levels": rows,
        "same_native_grid_control": {b: measurement(control[b], original[b], atol=2e-12) for b in BANDS},
        "fitter_spectral_effect": audit(native, spectral),
        "temperatures_K": temperatures.tolist(),
        "order32_surface_flux": {b: v.tolist() for b, v in spectra[-1].items()},
        "order16_vs32_surface_relative": {b: float(np.max(np.abs(spectra[1][b] / spectra[2][b] - 1))) for b in BANDS},
        "finite_bolometric_vs_infinite_planck_relative": (
            spectra[-1]["bolometric"] / (exact_sigma * temperatures**4) - 1
        ).tolist(),
        "independent_planck_sigma": exact_sigma,
        "legacy_temperature_law_sigma": const.STEFAN_BOLTZMAN_CONST,
        "limitations": "Known-response blackbody quadrature/truncation only. No missing atlas spectral information, "
        "atmosphere model accuracy or LD-table physical accuracy can be inferred by oversampling these fixtures.",
    }


def fit_resolution(device: str) -> dict[str, Any]:
    problem = make_problem()
    truth_fit = prepare_fit(problem)
    reference = prediction(truth_fit, problem.truth, 3)
    rng = np.random.default_rng(908)
    for band, data in problem.data.items():
        data.y_data = reference[band] + rng.normal(0, 1e-4, len(reference[band]))
        data.y_err = np.full(len(reference[band]), 1e-4)
    results = []
    for disc in [12, 6]:
        current = deepcopy(problem)
        current.setup["discretization"] = disc
        assessment = fit_assessment(device, problem=current)
        for value in assessment["least_squares"].values():
            value["difference_from_injected"] = {
                key: value["fitted"][key] - problem.truth[key] for key in value["fitted"]
            }
        # Evaluate coarse truth against known finite-reference data, using original likelihood/noise.
        model = prepare_fit(current)
        expected = prediction(model, problem.truth, disc)
        results.append(
            {
                "discretization_deg": disc,
                "public_fit_assessment": assessment,
                "truth_model_change": audit(
                    expected, reference, model.y_err, observed=model.y_data,
                ),
            },
        )
    return {
        "reference": "Native original CPU 3deg truth + fixed seed908 sigma1e-4 noise; finite mesh, not continuum truth",
        "injected": problem.truth,
        "levels": results,
        "limits": "Two-parameter fits measure mesh/noise effects together; "
        "no converged posterior/ESS or exact bias estimator.",
    }


def run(device: str, part: str) -> dict[str, Any]:
    started = perf_counter()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = (
            {"mesh": mesh_curves(device), "phase": phase_curves(device), "spectral": spectral_curves()}
            if part == "curves"
            else {"fitting": fit_resolution(device)}
        )
    return dict(
        result,
        device=device,
        part=part,
        all_implementation_parity_gates_passed=True,
        wall_seconds=perf_counter() - started,
        accuracy_contract="Noise-scaled diagnostic budget may fail for BOTH CPU and GPU; "
        "not a physical prior or GPU implementation discrepancy. Finite reference is not a continuum certificate.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--part", choices=["curves", "fitting"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.device, args.part)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    keys = ("device", "part", "all_implementation_parity_gates_passed", "wall_seconds")
    print(json.dumps({k: result[k] for k in keys}))


if __name__ == "__main__":
    main()
