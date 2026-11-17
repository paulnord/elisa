"""Finer reflected complete curves, explicit paired grids and physical fitting.

GPU use requires a bounded independently restored allocation. Finite references
are not continuum truth. Original CPU and independent primitive-tiled CPU
references are labelled separately; all complete timings include transfers.
"""

# ruff: noqa: INP001, S101, T201, SLF001 - standalone scientific driver/owned observer
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

from elisa import Observer, settings
from elisa.analytics.models import cost_fns, lc
from elisa.analytics.params import parameters
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from scripts.fine_mesh_reference import intrinsic, observe
from scripts.validate_gpu_accuracy import audit
from scripts.validate_gpu_atmosphere import BANDS, table_settings
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_binary import contact_phases, fixture, measurement
from scripts.validate_gpu_fit import compare, make_problem, normalized_vector, physical, prepare_fit

LEVELS = (3., 2., 1.5)
_ORIGINAL_MIN_REFERENCE = 2.


def curves(device: str) -> list[dict[str, Any]]:  # noqa: C901 - explicit matched controls and resource modes
    """Contact-focused curves with explicit original/independent CPU controls."""
    out = []
    engines = {}
    for name, inc, atlas in [("partial_bb", 85, "bb"), ("total_bb", 90, "bb"), ("partial_ck04", 88, "ck04")]:
        rows, references = [], []
        for disc in LEVELS:
            binary = (fixture({"inclination": inc, "discretization": disc})
                      if atlas == "bb" else table_fixture({"inclination": inc}))
            if atlas == "ck04":
                binary.primary.discretization_factor = binary.secondary.discretization_factor = np.radians(disc)
            phases = contact_phases(binary)
            observer = Observer(passband=list(BANDS), system=binary)
            t = perf_counter()
            reference = observe(observer, phases)[1]
            independent_seconds = perf_counter()-t
            dense, dense_seconds, original_parity = None, None, None
            if disc >= _ORIGINAL_MIN_REFERENCE:
                t = perf_counter()
                dense = observer.lc(phases=phases, normalize=True)[1]
                dense_seconds = perf_counter()-t
                original_parity = {b: measurement(reference[b], dense[b], atol=2e-12) for b in BANDS}
            execution, parity, costs = {}, {}, {}
            requested = ["cpu", "gpu"] if device == "gpu" else ["cpu"]
            for selected in requested:
                key = selected, atlas
                if key not in engines:
                    engines[key] = HybridBinaryLightCurve(list(BANDS), device=selected, reflection="tiled",
                                                          coverage="cpu", radiance="device" if atlas == "bb" else "cpu",
                                                          fallback="raise")
                engine = engines[key]
                t = perf_counter()
                result = engine(binary, phases, normalize=True)
                costs[selected] = {"first_complete_seconds": perf_counter()-t}
                assert result.backend == "hybrid"
                parity[selected] = {b: measurement(result.fluxes[b], reference[b], atol=2e-12) for b in BANDS}
                execution[selected] = {"backend": result.backend, "stages": result.timings,
                                       "radiance": engine.radiance, "coverage": engine.coverage,
                                       "reflection": engine.reflection}
                # One repeated complete call only in the partial BB case. Other
                # first-call costs remain labelled as such, not steady throughput.
                if name == "partial_bb":
                    t = perf_counter()
                    warm = engine(binary, phases, normalize=True)
                    costs[selected]["repeat_complete_seconds"] = perf_counter()-t
                    for b in BANDS:
                        measurement(warm.fluxes[b], reference[b], atol=2e-12)
            state = intrinsic(binary)
            counts = {n: len(getattr(state, n).faces) for n in ("primary", "secondary")}
            workspace = {}
            for selected in requested:
                kernel = engines[selected, atlas]._reflection_engines[settings.LIMB_DARKENING_LAW]
                packet = kernel.prepare(state, settings.REFLECTION_EFFECT_ITERATIONS)
                stats = kernel._evaluate.lower(*packet._arrays).compile().memory_analysis()
                visible = [len(v) for v in packet.visible_indices]
                symmetric = [len(v) for v in packet.symmetry_indices]
                workspace[selected] = {
                    "visible_faces": visible, "symmetric_faces": symmetric,
                    "full_donor_pairs": symmetric[0]*visible[1]+visible[0]*symmetric[1],
                    "max_pair_tile": kernel.block_size*max(visible),
                    "original_full_visible_matrix_elements": visible[0]*visible[1],
                    "compiled_temporary_bytes": stats.temp_size_in_bytes,
                    "compiled_argument_bytes": stats.argument_size_in_bytes,
                    "compiled_output_bytes": stats.output_size_in_bytes,
                    "limitation": "Compiled program memory accounting excludes allocator/compiler/runtime/host bridges",
                }
            references.append(reference)
            rows.append({"degrees_both_components": disc, "face_counts": counts, "phases": phases.tolist(),
                         "reference_fluxes": {b: v.tolist() for b, v in reference.items()},
                         "independent_cpu_seconds": independent_seconds, "unmodified_cpu_seconds": dense_seconds,
                         "independent_vs_unmodified_cpu": original_parity,
                         "hybrid_parity": parity, "execution": execution, "costs": costs,
                         "reflection_workspace": workspace})
        for i, row in enumerate(rows):
            row["vs_finite_1p5deg"] = audit(references[i], references[-1])
            if i:
                row["change_from_previous"] = audit(references[i-1], references[i])
        out.append({"case": name, "levels": rows,
                    "reference_contract": "Independent CPU original-primitives tiles, validated against unmodified "
                    "dense CPU through2deg;1.5deg is finite reference, not continuum truth"})
    return out


def reference_synthesis(phases: Any, discretization: float, observer: Any, **props: Any) -> Any:
    observer._system = lc.prepare_binary(discretization=discretization, **props)
    return observe(observer, phases)[1]


def fit_problem() -> Any:
    """Generate same finite1.5deg CPU reference truth and fixed1e-4 noise."""
    problem = make_problem()
    problem.config["system"]["inclination"] = {"value": 86.4, "fixed": True}
    problem.setup["discretization"] = 3.
    fit = prepare_fit(problem, phase_mode="observations", mesh_mode="paired")
    truth = fit._direct_prediction(problem.truth, 1.5, reference_synthesis, observer=fit.observer)
    rng = np.random.default_rng(1010)
    for band, data in problem.data.items():
        data.y_data = truth[band]+rng.normal(0, 1e-4, len(truth[band]))
        data.y_err = np.full(len(truth[band]), 1e-4)
    return problem


def fit_level(device: str, problem: Any, disc: float) -> dict[str, Any]:
    """Run original public LSQ at a declared paired mesh and one free temperature."""
    problem.setup["discretization"] = disc
    rows = {}
    selected = [("unmodified_cpu", None), ("tiled_jax_cpu", "cpu")]
    if device == "gpu":
        selected.append(("tiled_gpu", "gpu"))
    for label, requested in selected:
        # prepare_fit is not charged to public fit twice: construct a fresh
        # instance, then run the real public lifecycle below from its start.
        from scripts.validate_gpu_fit import fit_classes  # noqa: PLC0415

        cpu_cls, hybrid_cls = fit_classes("lsq", "detached", protocol=True)
        t = perf_counter()
        options = {"phase_mode": "observations", "mesh_mode": "paired"}
        fit = (cpu_cls(**options) if requested is None else hybrid_cls(
            device=requested, reflection="tiled", coverage="cpu", **options,
        ))
        fit.atmosphere_model, fit.limb_darkening_coefficients = deepcopy(problem.atmosphere), deepcopy(problem.ld)
        result = fit.fit(deepcopy(problem.data), parameters.BinaryInitialParameters(**deepcopy(problem.config)),
                         **deepcopy(problem.setup), max_nfev=16, jac="2-point", diff_step=1e-4,
                         xtol=1e-9, ftol=1e-9, gtol=1e-8)
        seconds = perf_counter()-t
        props = {k: v["value"] for k, v in fit.flat_result.items() if k != "r_squared"}
        predicted = fit.predict_light_curves(props)
        rows[label] = {"seconds_constructor_setup_compile_callbacks_serialization": seconds,
                       "primary_temperature_K": props["primary@t_eff"],
                       "chi_squared": float(cost_fns.wssr(fit.y_data, fit.y_err, predicted)),
                       "serialized_result": result}
        if requested is None:
            refined = fit._direct_prediction(props, 1.5, reference_synthesis, observer=deepcopy(fit.observer))
            rows[label]["fitted_resolution_effect"] = audit(predicted, refined, fit.y_err, observed=fit.y_data)
    expected = rows["unmodified_cpu"]
    parity = {label: {"temperature_K": compare(row["primary_temperature_K"], expected["primary_temperature_K"],
                                               atol=.01, rtol=0),
                      "chi_squared": compare(row["chi_squared"], expected["chi_squared"], atol=1e-5, rtol=0)}
              for label, row in rows.items() if label != "unmodified_cpu"}
    posterior = []
    for requested in [d for _, d in selected]:
        t = perf_counter()
        fit = prepare_fit(problem, device=requested, phase_mode="observations", mesh_mode="paired",
                          reflection="tiled", coverage="cpu")
        setup = perf_counter()-t
        base = normalized_vector(fit)
        samples = []
        for value in [float(base[0]), .5, .7]:
            vector = np.array([value])
            t = perf_counter()
            density = fit.ln_probability(vector)
            samples.append({"normalized_temperature": value, "log_posterior": float(density),
                            "complete_seconds": perf_counter()-t,
                            "physical": physical(fit, vector),
                            "execution": fit.hybrid_synthesis.last_evaluation if requested else None})
        posterior.append({"backend": "unmodified_cpu" if requested is None else requested,
                          "setup_seconds": setup, "candidates": samples})
    for row in posterior[1:]:
        for a, b in zip(row["candidates"], posterior[0]["candidates"], strict=True):
            a["parity"] = compare(a["log_posterior"], b["log_posterior"], atol=1e-6, rtol=0)
    return {"degrees_both_components": disc, "fits": rows, "fit_parity": parity, "posterior": posterior,
            "numerical_protocol": f"Observed phases, both{disc}deg, finite1.5deg independent CPU truth, "
            "seed1010 Gaussian "
            "sigma1e-4; original normalization, constraints, priors and noise. One fitted T_primary, constrained "
            "T_secondary=T_primary-1500; inclination fixed86.4deg.",
            "limitations": "Unqualified finite-mesh fitting assessment, no continuum posterior or inference recovery "
            "certificate; MCMC density callbacks only in this finer experiment, prior milestone sampler replay "
            "retained, no convergence/ESS claim"}


def fitting(device: str, levels: tuple[float, ...] = (3., 2.)) -> dict[str, Any]:
    """Assess two finer meshes against exactly the same finite CPU truth/data."""
    problem = fit_problem()
    return {"levels": [fit_level(device, problem, disc) for disc in levels],
            "truth": "Independent original-CPU-primitives1.5deg finite reference, T6500/5000K, incl86.4deg; "
            "seed1010 and sigma1e-4; no continuum certificate"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--part", choices=["curves", "fitting"], required=True)
    parser.add_argument("--fit-level", type=float, choices=[3., 2.], help="One fit mesh per bounded window")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        levels = (args.fit_level,) if args.fit_level is not None else (3., 2.)
        result = (curves(args.device) if args.part == "curves" else fitting(args.device, levels))
    report = {"part": args.part, "device": args.device, "all_implementation_parity_gates_passed": True,
              "results": result, "wall_seconds": perf_counter()-started,
              "cost_contract": "All timed calls start/end with host curves/densities; CPU bridges and all transfers "
              "included. Explicit JAX CPU/device selection. First per-instance calls include compilation incurred, "
              "reused code shapes may be warm. Full driver wall includes synthetic references and auxiliary "
              "post-fit diagnostics."}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    keys = ["part", "device", "all_implementation_parity_gates_passed", "wall_seconds"]
    print(json.dumps({k: report[k] for k in keys}))


if __name__ == "__main__":
    main()
