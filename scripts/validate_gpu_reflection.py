"""Device irradiation in complete curves and original physical fitting.

GPU use requires the independently restored bounded systemd allocation. All
complete costs include CPU bridges, transfers and matching final host outputs.
"""

# ruff: noqa: INP001, S101, T201 - standalone scientific validation/bootstrap
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import Observer, ld, settings
from elisa.binary_system.curves.c_router import prep_initial_system
from elisa.binary_system.surface import temperature as cpu_reflection
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_reflection import DeviceReflection
from scripts.validate_gpu_atmosphere import (
    BANDS,
    intermediates,
    sources,
    table_settings,
)
from scripts.validate_gpu_atmosphere import (
    fit_problem as table_problem,
)
from scripts.validate_gpu_atmosphere import (
    fixture as table_fixture,
)
from scripts.validate_gpu_binary import contact_phases, measurement
from scripts.validate_gpu_binary import fixture as bb_fixture
from scripts.validate_gpu_fit import compare, fit_assessment, make_problem, normalized_vector, physical, prepare_fit

CASES = [
    {"name": "primary_cooler_non_eclipsing", "q": 0.5, "spin": [0.5, 1.5],
     "temperatures": [4800, 7500], "inclination": 60, "iterations": 1},
    {"name": "secondary_cooler_total", "q": 2.0, "spin": [1.5, 0.5], "potentials": [12, 16],
     "temperatures": [7500, 4800], "inclination": 90, "iterations": 3},
    {"name": "equal_temperatures_cosine", "q": 1.0, "potentials": [12, 12],
     "temperatures": [6500, 6500], "inclination": 85, "iterations": 2, "law": "cosine"},
    {"name": "logarithmic_partial", "inclination": 85, "iterations": 2, "law": "logarithmic"},
    {"name": "square_root_grazing", "inclination": 79, "iterations": 5, "law": "square_root"},
    {"name": "ck04_actual_table_ld", "inclination": 88, "iterations": 2, "atlas": "ck04"},
]


def fixture(case: dict[str, Any]) -> Any:
    if case.get("atlas") == "ck04":
        return table_fixture(case)
    # The older binary fixture labels only "linear" as its one-coefficient law.
    return bb_fixture({"discretization": 12, **case,
                       "law": "linear" if case.get("law") == "cosine" else case.get("law", "linear")})


def original_blocks(system: Any, packet: Any) -> dict[str, Any]:
    """Use the unchanged CPU symmetry helpers on exactly the same input state."""
    names = ("primary", "secondary")
    stars = [getattr(system, name) for name in names]
    visible, symmetry = {}, {}
    for i, name in enumerate(names):
        visible[name] = np.zeros(len(stars[i].faces), dtype=bool)
        symmetry[name] = np.zeros(len(stars[i].faces), dtype=bool)
        visible[name][packet.visible_indices[i]] = True
        symmetry[name][packet.symmetry_indices[i]] = True
    shape, reduced = cpu_reflection.get_distance_matrix_shape(system, visible)
    distance, join = cpu_reflection.get_symmetrical_distance_matrix(
        shape, reduced, {n: s.face_centres for n, s in zip(names, stars, strict=True)}, visible, symmetry,
    )
    gamma = cpu_reflection.get_symmetrical_gammma(
        shape[:2], reduced, {n: s.normals for n, s in zip(names, stars, strict=True)}, join, visible, symmetry,
    )
    cpu_reflection.check_symmetric_gamma_for_negative_num(gamma, reduced)
    q = cpu_reflection.get_symmetrical_q_ab(shape[:2], reduced, gamma, distance)
    ldc = {n: ld.get_bolometric_ld_coefficients(s.temperatures, s.log_g, s.metallicity,
                                               s.limb_darkening_coefficients)
           for n, s in zip(names, stars, strict=True)}
    dg = cpu_reflection.get_symmetrical_d_gamma(shape[:2], reduced, ldc, gamma, reflection_only=True)
    a, b = reduced
    return {"coupling": (q[:a]*dg["secondary"][:a], q[:, :b]*dg["primary"][:, :b]),
            "distance": (distance[:a], distance[:, :b]),
            "gamma": ((gamma["primary"][:a], gamma["secondary"][:a]),
                      (gamma["primary"][:, :b], gamma["secondary"][:, :b]))}


def reflection_case(case: dict[str, Any], device: str) -> dict[str, Any]:
    """Compare actual device matrices/iterations with unchanged complete CPU."""
    binary = fixture(case)
    with settings.override(REFLECTION_EFFECT=False):
        initial = prep_initial_system(binary)
    kernel = DeviceReflection(case.get("law", "linear"), device=device)
    started = perf_counter()
    packet = kernel.prepare(initial, case["iterations"])
    prepared_seconds = perf_counter()-started
    started = perf_counter()
    actual = kernel._jax.tree.map(np.asarray, packet.evaluate())  # noqa: SLF001
    first_seconds = perf_counter()-started
    expected = initial.copy()
    cpu_reflection.apply_reflection_effect(expected, 1.0, case["iterations"])
    blocks = original_blocks(initial, packet)
    comparisons = {"coupling": [], "distance": [], "gamma": [], "temperatures": []}
    for i, name in enumerate(("primary", "secondary")):
        for key in ("coupling", "distance"):
            comparisons[key].append(measurement(actual[key][i], blocks[key][i], atol=64*np.finfo(float).eps))
        comparisons["gamma"].append([
            measurement(a, b, atol=64*np.finfo(float).eps)
            for a, b in zip(actual["gamma"][i], blocks["gamma"][i], strict=True)
        ])
        before, after = (getattr(s, name) for s in (initial, expected))
        comparisons["temperatures"].append(measurement(actual["temperatures"][i], after.temperatures))
        dark = np.ones(len(before.faces), dtype=bool)
        dark[packet.visible_indices[i]] = False
        assert np.all(actual["temperatures"][i] >= before.temperatures*(1-3e-13))
        np.testing.assert_allclose(actual["temperatures"][i][dark], before.temperatures[dark], rtol=0, atol=0)
        comparisons["temperatures"][-1]["maximum_heating_K"] = float(np.max(after.temperatures-before.temperatures))
    costs = {}

    def original_temperatures() -> Any:
        reflected = cpu_reflection.apply_reflection_effect(initial.copy(), 1.0, case["iterations"])
        return reflected.primary.temperatures, reflected.secondary.temperatures

    def device_temperatures() -> Any:
        reflected = kernel.prepare(initial.copy(), case["iterations"]).evaluate()
        return tuple(np.asarray(t) for t in reflected["temperatures"])

    for label, function in [
        ("original_cpu", original_temperatures), ("device_full", device_temperatures),
    ]:
        samples = []
        for _ in range(3):
            started = perf_counter()
            function()
            samples.append(perf_counter()-started)
        costs[label] = {"samples_seconds": samples, "median_seconds": float(np.median(samples))}
    return {"comparisons": comparisons, "costs_matching_host_temperatures": costs,
            "preparation_seconds": prepared_seconds, "first_evaluation_seconds": first_seconds,
            "faces": [len(initial.primary.faces), len(initial.secondary.faces)],
            "visible": [len(x) for x in packet.visible_indices],
            "quarter_visible": [len(x) for x in packet.symmetry_indices]}


def curves(device: str) -> dict[str, Any]:
    started = perf_counter()
    engine = HybridBinaryLightCurve(list(BANDS), device=device, fallback="raise")
    previous = HybridBinaryLightCurve(list(BANDS), device=device, fallback="raise", reflection="cpu")
    result = {"constructors_seconds": perf_counter()-started, "cases": []}
    for case in CASES:
        with settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=case["iterations"],
                               LIMB_DARKENING_LAW=case.get("law", "linear")):
            binary = fixture(case)
            phases = contact_phases(binary)
            actual = engine(binary, phases, normalize=False, diagnostics=True)
            expected = Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=False)[1]
            normalized = engine(binary, phases, normalize=True)
            expected_norm = Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)[1]
            assert actual.backend == "hybrid"
            assert actual.diagnostics["reflection"] is not None
            row = {"fixture": case, "observations_per_band": len(phases),
                   "raw": {b: measurement(actual.fluxes[b], expected[b]) for b in BANDS},
                   "normalized": {b: measurement(normalized.fluxes[b], expected_norm[b], atol=2e-12) for b in BANDS},
                   "reflection": reflection_case(case, device),
                   "intermediates": intermediates(actual, binary), "first_full_call_seconds": actual.timings,
                   "seconds": {k: [] for k in ("original_cpu", "cpu_reflection_hybrid", "device_reflection_hybrid")},
                   "stages": []}
            for _ in range(3):
                for label, function in [
                    ("original_cpu", lambda binary=binary, phases=phases:
                     Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)),
                    ("cpu_reflection_hybrid", lambda binary=binary, phases=phases:
                     previous(binary, phases, normalize=True)),
                    ("device_reflection_hybrid", lambda binary=binary, phases=phases:
                     engine(binary, phases, normalize=True)),
                ]:
                    started = perf_counter()
                    r = function()
                    row["seconds"][label].append(perf_counter()-started)
                    if label == "device_reflection_hybrid":
                        row["stages"].append(r.timings)
            row["median_seconds"] = {k: float(np.median(v)) for k, v in row["seconds"].items()}
            result["cases"].append(row)
    return result


def fitting(device: str) -> dict[str, Any]:
    out = {}
    for family, builder in [("blackbody", make_problem), ("ck04_table_ld", table_problem)]:
        problem = (builder(free_nuisance=True, shape_parameters=True) if family == "blackbody"
                   else builder(shape_parameters=True))
        problem.config["primary"]["albedo"] = {
            "value": 0.5 if family == "blackbody" else 0.1,
            "fixed": False, "min": 0.1 if family == "blackbody" else 0.05,
            "max": 0.9 if family == "blackbody" else 0.15,
        }
        original = prepare_fit(problem)
        hybrid = prepare_fit(problem, device=device)
        base = normalized_vector(original)
        comparisons = []
        for vector in [base, np.full(base.shape, 0.65), np.linspace(0.25, 0.7, len(base))]:
            props = physical(original, vector)
            a, b = hybrid.predict_light_curves(props), original.predict_light_curves(props)
            comparisons.append({"parameters": props,
                                "predictions": {band: compare(a[band], b[band]) for band in problem.data},
                                "posterior": compare(hybrid.ln_probability(vector), original.ln_probability(vector),
                                                     rtol=0, atol=1e-6),
                                "execution": hybrid.hybrid_synthesis.last_evaluation})
            assert hybrid.hybrid_synthesis.last_evaluation["backend"] == "hybrid"
        assessment = fit_assessment(device, problem=builder())
        costs = {}
        for label, requested, mode in [("original_cpu", None, "cpu"), ("cpu_reflection_hybrid", device, "cpu"),
                                        ("device_reflection_hybrid", device, "device")]:
            started = perf_counter()
            fit = prepare_fit(problem, device=requested, reflection=mode)
            setup = perf_counter()-started
            started = perf_counter()
            fit.ln_probability(base)
            first = perf_counter()-started
            samples = []
            for f in [0.0, 0.02, 0.05]:
                vector = (1-f)*base+f*np.linspace(0.2, 0.8, len(base))
                started = perf_counter()
                fit.ln_probability(vector)
                samples.append(perf_counter()-started)
            costs[label] = {"setup_seconds": setup, "first_callback_seconds": first,
                            "samples_seconds": samples, "median_seconds": float(np.median(samples))}
        out[family] = {"predictions": comparisons, "public_fit_assessment": assessment, "likelihood_costs": costs}
    return out


def run(device: str) -> dict[str, Any]:
    started = perf_counter()
    with table_settings():
        curve_results = curves(device)
        with settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
            fit_results = fitting(device)
    return {"device": device, "all_correctness_gates_passed": True, "curves": curve_results,
            "fitting": fit_results, "source_hashes": sources(), "wall_seconds": perf_counter()-started,
            "contract": "Fresh device irradiation matrices and sequential iterations feed emission/coverage; "
            "CPU mesh, visibility/source/LD preparation, host T for tables and hull bridges remain. "
            "All full costs include transfers and matching final host curves; chains are smoke checks only."}


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
