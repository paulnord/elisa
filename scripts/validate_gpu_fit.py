"""Bounded physical fitting/likelihood assessment against original ELISa.

Requires an independently restored authorized allocation for --device gpu.
Short emcee runs below test callback/move equivalence, not convergence or ESS.
"""

# ruff: noqa: INP001, T201, S101 - standalone scientific validation driver
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.testing import assert_allclose

from elisa import BinarySystem, Observer, settings
from elisa import units as u
from elisa.analytics import LCData
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.lc_prediction import predict_light_curves
from elisa.analytics.models import cost_fns
from elisa.analytics.params import parameters
from elisa.experimental.gpu_fit import HybridDetachedLeastSquaresFit, HybridDetachedMCMCFit

BANDS = ("Generic.Bessell.B", "Generic.Bessell.V", "bolometric")


@dataclass
class FitProblem:
    config: dict[str, Any]
    data: dict[str, Any]
    setup: dict[str, Any]
    atmosphere: dict[str, str]
    ld: dict[str, Any]
    truth: dict[str, Any]


def make_problem(
    *, jd: bool = False, sampling: str = "direct", free_nuisance: bool = False,
    shape_parameters: bool = False, standard: bool = False, mass_ratio: float = 0.7,
) -> FitProblem:
    """Generate deterministic real CPU data with uneven grids and repetitions."""
    base = {
        "system": {"inclination": 86.4, "period": 3.0, "eccentricity": 0.0,
                   "argument_of_periastron": 90.0, "mass_ratio": mass_ratio, "semi_major_axis": 10.0,
                   "phase_shift": 0.017, "additional_light": 0.2},
        "primary": {"t_eff": 6500.0, "surface_potential": 8.0, "synchronicity": 1.0,
                    "gravity_darkening": 0.32, "albedo": 0.5, "metallicity": 0.0},
        "secondary": {"t_eff": 5000.0, "surface_potential": 12.0, "synchronicity": 1.0,
                      "gravity_darkening": 0.32, "albedo": 0.5, "metallicity": 0.0},
    }
    if standard:
        del base["system"]["mass_ratio"], base["system"]["semi_major_axis"]
        base["primary"]["mass"], base["secondary"]["mass"] = 1.2, 1.2 * mass_ratio
    cfg = {g: {k: {"value": v, "fixed": True} for k, v in values.items()} for g, values in base.items()}
    cfg["system"]["inclination"] = {"value": 85.0, "fixed": False, "min": 83.0, "max": 89.0}
    cfg["primary"]["t_eff"] = {"value": 6300.0, "fixed": False, "min": 6000.0, "max": 7000.0}
    cfg["secondary"]["t_eff"] = {"constraint": "primary@t_eff - 1500"}
    cfg["nuisance"] = {"ln_f": {"value": -2.0, "fixed": not free_nuisance, "min": -5.0, "max": 1.0}}
    if free_nuisance:
        cfg["system"]["additional_light"] = {"value": 0.2, "fixed": False, "min": 0.0, "max": 0.8}
    if shape_parameters:
        cfg["primary"]["surface_potential"] = {"value": 8.0, "fixed": False, "min": 7.8, "max": 8.2}
        if standard:
            cfg["primary"]["mass"] = {"value": 1.2, "fixed": False, "min": 1.1, "max": 1.3}
        else:
            cfg["system"]["mass_ratio"] = {"value": mass_ratio, "fixed": False, "min": 0.6, "max": 0.8}
    if jd:
        cfg["system"]["primary_minimum_time"] = {"value": 2450000.0, "fixed": True}
    truth = {f"{g}@{k}": v for g, values in base.items() for k, v in values.items()}
    atmosphere = dict.fromkeys(("primary", "secondary"), "bb")
    ld = {name: {band: [0.4] for band in BANDS} for name in ("primary", "secondary")}
    extended = parameters.extend_json_with_atm_params(truth, atmosphere_model=atmosphere,
                                                     limb_darkening_coefficients=ld)
    grids = {
        BANDS[0]: np.array([0, .007, .025, .25, .48, .5, .516, .75, 1, .007]) + .017,
        BANDS[1]: np.array([.5, 0, .005, .01, .02, .04, .25, .475, .49, .5, 1.005, -.5, .75]) + .017,
        BANDS[2]: np.array([0, .01, .25, .5, .76]) + .017,
    }
    coordinates = {band: 2450000.0 + 3 * phases if jd else phases for band, phases in grids.items()}
    xs, reducers = parameters.xs_reducer(coordinates)
    observer = Observer(passband=list(BANDS), system=None)
    observer.system_cls = BinarySystem
    if jd:
        extended["system@primary_minimum_time"] = 2450000.0
    # Generate truth through the ORIGINAL prediction layer, so l3 uses the
    # synthesis-union mean before per-band mapping. Per-band independent
    # Observer calls would define a different cadence-dependent l3 convention.
    truth_curves = predict_light_curves(xs, {b: np.ones(len(x)) for b, x in coordinates.items()}, reducers,
                                       12.0, observer, 1000, extended, x_data=coordinates)
    data = {}
    rng = np.random.default_rng(717)
    for band, phases in grids.items():
        values = truth_curves[band] + rng.normal(0, 1e-4, len(phases))
        data[band] = LCData(x_data=coordinates[band],
                            y_data=values, y_err=np.full(len(phases), 1e-4),
                            x_unit=u.d if jd else u.dimensionless_unscaled, y_unit=u.dimensionless_unscaled)
    threshold = 1000 if sampling == "direct" else 12
    samples: Any = "uniform"
    if sampling == "manual_equal_length":
        size = len(np.unique(np.concatenate([d.x_data for d in data.values()])))
        samples = np.linspace(-0.1, 1.1, size)
    return FitProblem(cfg, data, {"discretization": 12.0, "interp_treshold": threshold, "samples": samples},
                      atmosphere, ld, truth)


def prepare_fit(
    problem: FitProblem, *, kind: str = "mcmc", device: str | None = None, radiance: str = "device",
) -> Any:
    """Use the same real set_up for original and experimental fitting classes."""
    if kind == "mcmc":
        fit = (mcmc.DetachedLightCurveFit() if device is None
               else HybridDetachedMCMCFit(device=device, radiance=radiance))
    else:
        fit = (least_squares.DetachedLightCurveFit() if device is None
               else HybridDetachedLeastSquaresFit(device=device, radiance=radiance))
    fit.atmosphere_model, fit.limb_darkening_coefficients = deepcopy(problem.atmosphere), deepcopy(problem.ld)
    fit.set_up(parameters.BinaryInitialParameters(**deepcopy(problem.config)), deepcopy(problem.data),
               passband=BANDS, observer_system_cls=BinarySystem, **deepcopy(problem.setup))
    if kind == "mcmc":
        fit.norm_init_vector = normalized_vector(fit)
        fit.sigmas = fit.normalized_sigma(fit.norm_init_vector)
    return fit


def normalized_vector(fit: Any) -> np.ndarray:
    return np.asarray(parameters.vector_normalizer(fit.initial_vector, fit.fitable, fit.normalization))


def physical(fit: Any, normalized: Any) -> dict[str, Any]:
    values = parameters.vector_renormalizer(normalized, fit.fitable, fit.normalization)
    return parameters.prepare_properties_set(np.asarray(values), fit.fitable, fit.constrained, fit.fixed)


def compare(actual: Any, expected: Any, *, atol: float = 2e-12, rtol: float = 3e-13) -> dict[str, Any]:
    a, b = np.broadcast_arrays(np.asarray(actual), np.asarray(expected))
    assert_allclose(a, b, atol=atol, rtol=rtol)
    return {"max_abs": float(np.max(np.abs(a - b), initial=0)), "elements": a.size, "atol": atol, "rtol": rtol}


def prediction_cases(device: str) -> list[dict[str, Any]]:
    rows = []
    for mode, jd, standard in [("direct", False, False), ("direct", True, False),
                                ("manual_equal_length", False, False), ("uniform", False, False),
                                ("direct", False, True)]:
        problem = make_problem(jd=jd, sampling=mode, free_nuisance=True, shape_parameters=True, standard=standard)
        original, hybrid = prepare_fit(problem), prepare_fit(problem, device=device)
        base = normalized_vector(original)
        if mode == "manual_equal_length":
            assert original.fit_xs is not None
            assert len(original.fit_xs) == len(original.x_data_reduced)
        vectors = [base, np.full(base.shape, 0.65), np.linspace(0.25, 0.7, len(base))]
        row: dict[str, Any] = {"mode": mode, "jd": jd, "standard_masses": standard,
                               "parameters": list(original.fitable),
                               "observations": {b: len(y) for b, y in original.y_data.items()},
                               "support": None if original.fit_xs is None else original.fit_xs.tolist(),
                               "candidates": []}
        for vector in vectors:
            props = physical(original, vector)
            expected = original.predict_light_curves(props)
            predicted = hybrid.predict_light_curves(props)
            direct_cpu = predict_light_curves(
                original.x_data_reduced, original.y_data, original.x_data_reducer, original.discretization,
                original.observer, 1000, props, x_data=original.x_data, y_err=original.y_err,
                atmosphere_model=problem.atmosphere, limb_darkening_coefficients=problem.ld,
            )
            gaussian_cpu, gaussian_hybrid = original.likelihood(vector), hybrid.likelihood(vector)
            post_cpu, post_hybrid = original.ln_probability(vector), hybrid.ln_probability(vector)
            row["candidates"].append({
                "normalized_parameters": vector.tolist(),
                "predictions": {b: compare(predicted[b], expected[b]) for b in BANDS},
                "likelihood": compare(gaussian_hybrid, gaussian_cpu, atol=1e-6, rtol=0),
                "posterior": compare(post_hybrid, post_cpu, atol=1e-6, rtol=0),
                "means": {b: float(predicted[b].mean()) for b in BANDS},
                "original_sampling_error_vs_direct": {
                    b: float(np.max(np.abs(expected[b] - direct_cpu[b]))) for b in BANDS
                },
                "execution": hybrid.hybrid_synthesis.last_evaluation,
            })
            assert hybrid.hybrid_synthesis.last_evaluation["backend"] == "hybrid"
        before = hybrid.hybrid_synthesis.calls
        assert np.isneginf(hybrid.ln_probability(np.full(base.shape, -0.1)))
        assert hybrid.hybrid_synthesis.calls == before
        # Priors are inherited, including the original normal-prior convention.
        original.sigmas[0] = hybrid.sigmas[0] = 0.2
        row["normal_prior_posterior"] = compare(hybrid.ln_probability(vectors[-1]),
                                                original.ln_probability(vectors[-1]), atol=1e-6, rtol=0)
        rows.append(row)
    return rows


def fit_assessment(device: str, *, steps: int = 3) -> dict[str, Any]:
    """Exercise actual original fit() lifecycle and serialized result contracts."""
    problem = make_problem()
    results = {}
    for label, cls, options in [("original_cpu", least_squares.DetachedLightCurveFit, {}),
                                ("hybrid", HybridDetachedLeastSquaresFit, {"device": device})]:
        constructed_at = perf_counter()
        fit = cls(**options)
        constructor_seconds = perf_counter() - constructed_at
        fit.atmosphere_model, fit.limb_darkening_coefficients = deepcopy(problem.atmosphere), deepcopy(problem.ld)
        started = perf_counter()
        serialized = fit.fit(deepcopy(problem.data), parameters.BinaryInitialParameters(**deepcopy(problem.config)),
                             **deepcopy(problem.setup), diff_step=1e-4, max_nfev=16,
                             xtol=1e-9, ftol=1e-9, gtol=1e-8)
        elapsed = perf_counter() - started
        props = {k: v["value"] for k, v in fit.flat_result.items() if k != "r_squared"}
        predicted = fit.predict_light_curves(props)
        results[label] = {"constructor_seconds": constructor_seconds,
                          "seconds_including_setup_compile": elapsed,
                          "total_seconds_including_constructor": constructor_seconds + elapsed,
                          "fitted": {k: props[k] for k in fit.fitable},
                          "r_squared": fit.flat_result["r_squared"]["value"],
                          "chi_squared": float(cost_fns.wssr(fit.y_data, fit.y_err, predicted)),
                          "serialized_result": serialized}
    differences = {key: compare(results["hybrid"]["fitted"][key], results["original_cpu"]["fitted"][key],
                                atol=0.01 if key.endswith("t_eff") else 1e-4, rtol=0)
                   for key in results["original_cpu"]["fitted"]}
    differences["chi_squared"] = compare(results["hybrid"]["chi_squared"], results["original_cpu"]["chi_squared"],
                                         atol=1e-5, rtol=0)
    samplers = {}
    starting = prepare_fit(problem)
    initial = normalized_vector(starting) + np.random.default_rng(813).normal(0, 0.01, (8, len(starting.fitable)))
    random_state = np.random.get_state()  # noqa: NPY002 - replay original emcee contract
    try:
        for label, cls, options in [("original_cpu", mcmc.DetachedLightCurveFit, {}),
                                    ("hybrid", HybridDetachedMCMCFit, {"device": device})]:
            constructed_at = perf_counter()
            fit = cls(**options)
            constructor_seconds = perf_counter() - constructed_at
            fit.atmosphere_model, fit.limb_darkening_coefficients = deepcopy(problem.atmosphere), deepcopy(problem.ld)
            np.random.seed(814)  # noqa: NPY002 - replay the original emcee random-state contract
            started = perf_counter()
            fit.fit(deepcopy(problem.data), parameters.BinaryInitialParameters(**deepcopy(problem.config)),
                    **deepcopy(problem.setup), nwalkers=8, nsteps=steps, burn_in=0,
                    initial_state=initial.copy(), save=False, progress=False)
            elapsed = perf_counter() - started
            sampler = fit.last_sampler
            samplers[label] = {"constructor_seconds": constructor_seconds,
                               "seconds_including_setup_compile": elapsed,
                               "total_seconds_including_constructor": constructor_seconds + elapsed,
                               "chain": sampler.get_chain().tolist(),
                               "log_probability": sampler.get_log_prob().tolist(),
                               "acceptance_fraction": sampler.acceptance_fraction.tolist()}
    finally:
        np.random.set_state(random_state)  # noqa: NPY002 - restore caller state
    chain_parity = compare(samplers["hybrid"]["chain"], samplers["original_cpu"]["chain"], atol=0, rtol=0)
    density_parity = compare(samplers["hybrid"]["log_probability"], samplers["original_cpu"]["log_probability"],
                             atol=1e-6, rtol=0)
    return {"least_squares": results, "least_squares_parity": differences,
            "sampler_smoke": samplers, "chain_parity": chain_parity, "chain_density_parity": density_parity,
            "sampler_limitations": (f"{steps} steps,8 walkers: callback/move replay only; "
                                    "no convergence, posterior precision or ESS/sec")}


def timings(device: str, *, repeats: int = 3) -> dict[str, Any]:
    problem = make_problem(free_nuisance=True)
    paths = {"original_cpu": (None, "cpu"), "preceding_cpu_radiance_hybrid": (device, "cpu"),
             "hybrid": (device, "device")}
    if device == "gpu":
        paths["hybrid_jax_cpu"] = ("cpu", "device")
    result = {}
    for label, (requested, radiance) in paths.items():
        before_setup = perf_counter()
        fit = prepare_fit(problem, device=requested, radiance=radiance)
        setup_seconds = perf_counter() - before_setup
        vector = normalized_vector(fit)
        start = perf_counter()
        fit.ln_probability(vector)
        cold = perf_counter() - start
        samples = []
        for fraction in np.linspace(0, 0.05, repeats):
            changed = (1 - fraction) * vector + fraction * np.linspace(0.2, 0.8, len(vector))
            start = perf_counter()
            fit.ln_probability(changed)
            samples.append(perf_counter() - start)
        result[label] = {"object_and_observation_setup_seconds": setup_seconds,
                         "first_callback_seconds": cold, "changed_parameter_callback_seconds": samples,
                         "median_seconds": float(np.median(samples)),
                         "last_synthesis": getattr(fit, "hybrid_synthesis", None).last_evaluation
                         if hasattr(fit, "hybrid_synthesis") else None}
    return result


def run(device: str) -> dict[str, Any]:
    started = perf_counter()
    with settings.override(REFLECTION_EFFECT=False, LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1,
                           NUMBER_OF_MCMC_PROCESSES=1, SUPPRESS_WARNINGS=True):
        parity = prediction_cases(device)
        assessment = fit_assessment(device)
        costs = timings(device)
    return {"device": device, "all_correctness_gates_passed": True, "prediction_cases": parity,
            "physical_fit_assessment": assessment, "transfer_inclusive_likelihood_timings": costs,
            "wall_seconds": perf_counter() - started,
            "contract": ("Original serializer/constraints, support/JD mapping/interpolation, mean-normalization, "
                         "residuals/noise/priors and fit lifecycle; fresh hybrid physical synthesis per callback"),
            "limitations": ["Experimental hybrid with CPU geometry/reflection/custom LD and hull/visibility",
                            "Serial physical callbacks and CPU emcee moves; no batched/resident physical sampler",
                            "No convergence/ESS or physical-posterior precision claim from short chains",
                            "Full callbacks include candidate construction, transfers, costs and shape compilation"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.device)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("device", "all_correctness_gates_passed", "wall_seconds")}))


if __name__ == "__main__":
    main()
