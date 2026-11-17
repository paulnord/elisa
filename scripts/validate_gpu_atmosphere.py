"""Real repository CK04/Van Hamme fixtures through complete hybrid curves/fits.

GPU use requires the independently restored bounded allocation. This script
does not manage production. Fixture tables are unmodified existing CSV sources;
their sparse coverage is declared, not extended with synthetic atmospheres.
"""

# ruff: noqa: INP001, T201, S101 - standalone scientific validation/import bootstrap
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import BinarySystem, Observer, Star, settings
from elisa import units as u
from elisa.analytics import LCData
from elisa.analytics.binary_fit.lc_prediction import predict_light_curves
from elisa.analytics.params import parameters
from elisa.base.curves.utils import calculate_surface_element_fluxes
from elisa.binary_system.curves.c_router import prep_initial_system
from elisa.binary_system.curves.utils import prep_surface_params
from elisa.binary_system.surface.coverage import compute_surface_coverage
from elisa.binary_system.utils import move_sys_onpos
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from scripts.validate_gpu_binary import contact_phases, measurement
from scripts.validate_gpu_fit import FitProblem, compare, fit_assessment, normalized_vector, physical, prepare_fit

BANDS = ("Generic.Bessell.V", "bolometric")
DATA = Path(__file__).resolve().parents[1] / "tests/data"
CASES = [
    {"name": "ck04_partial", "inclination": 85, "reflection": False},
    {"name": "ck04_total_reflection", "inclination": 90, "reflection": True},
    {"name": "ck04_non_eclipsing_changed_q", "inclination": 60, "q": 0.8, "reflection": False},
    {"name": "ck04_cool_giant", "inclination": 88, "period": 24, "temperatures": [5340, 5410], "reflection": True},
    {"name": "ck04_hot_compact", "inclination": 88, "period": 1.8, "temperatures": [25400, 25600], "reflection": False},
    {"name": "blackbody_table_ld", "inclination": 88, "atlas": "bb", "reflection": True},
]


@contextmanager
def table_settings() -> Any:
    """Process-only paths to real versioned test assets; no external downloads."""
    with settings.override(
        CK04_ATM_TABLES=str(DATA / "ck04"),
        LD_TABLES=str(DATA / "light_curves/limbdarkening"),
        LIMB_DARKENING_LAW="linear",
        NUMBER_OF_PROCESSES=1,
        NUMBER_OF_MCMC_PROCESSES=1,
        SUPPRESS_WARNINGS=True,
    ):
        yield


def fixture(case: dict[str, Any]) -> BinarySystem:
    temps = case.get("temperatures", [10080, 10160])
    common = {
        "synchronicity": 1,
        "atmosphere": case.get("atlas", "ck04"),
        "metallicity": 0,
        "gravity_darkening": 0.02,
        "albedo": 0.1,
        "discretization_factor": 12,
    }
    return BinarySystem(
        Star(mass=1.2, t_eff=temps[0], surface_potential=10, **common),
        Star(mass=1.2 * case.get("q", 0.7), t_eff=temps[1], surface_potential=8, **common),
        period=case.get("period", 4),
        eccentricity=0,
        argument_of_periastron=90,
        inclination=case["inclination"],
        phase_shift=0.017,
        additional_light=0.2,
        distance=25,
    )


def sources() -> dict[str, str]:
    """Hashes of actual inherited data, including response and wavelength grid."""
    root = DATA.parents[1]
    paths = list((DATA / "ck04").rglob("*.csv"))
    paths += list((DATA / "light_curves/limbdarkening").glob("*.csv"))
    paths += [root / "src/elisa/passband/Generic.Bessell.V.csv", root / "src/elisa/data/wavelength.json"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def intermediates(result: Any, binary: Any) -> dict[str, Any]:
    """Actual device face/component flux versus unmodified CPU containers."""
    cpu = prep_initial_system(binary)
    observer = Observer(passband=list(BANDS), system=binary)
    prep_surface_params(
        cpu,
        return_values=False,
        write_to_containers=True,
        passband=observer.passband,
        left_bandwidth=observer.left_bandwidth,
        right_bandwidth=observer.right_bandwidth,
    )
    out: dict[str, Any] = {"state": {}, "phases": []}
    for name in ("primary", "secondary"):
        actual, expected = getattr(result.diagnostics["initial_system"], name), getattr(cpu, name)
        row = {
            key: measurement(getattr(actual, key), getattr(expected, key), atol=64 * np.finfo(float).eps)
            for key in ("temperatures", "log_g", "areas")
        }
        row["temperature_range_K"] = [float(actual.temperatures.min()), float(actual.temperatures.max())]
        row["log_g_cgs_range"] = [float(actual.log_g.min() + 2), float(actual.log_g.max() + 2)]
        row["bands"] = {}
        for band in BANDS:
            row["bands"][band] = {
                "normal_radiance": measurement(actual.normal_radiance[band], expected.normal_radiance[band]),
                "ld": measurement(actual.ld_cfs[band], expected.ld_cfs[band]),
                "ld_range": [float(actual.ld_cfs[band].min()), float(actual.ld_cfs[band].max())],
            }
        out["state"][name] = row
    for i, position in enumerate(result.diagnostics["positions"]):
        cpu_pos = move_sys_onpos(cpu, position.position, on_copy=True)
        compute_surface_coverage(
            cpu_pos,
            binary.semi_major_axis,
            in_eclipse=bool(result.diagnostics["in_eclipse"][i]),
            return_values=False,
            write_to_containers=True,
        )
        packet = result.diagnostics["occultation"]["records"][i // 8]["result"]
        record: dict[str, Any] = {"phase": float(position.position.phase), "components": {}}
        for comp, name in enumerate(("primary", "secondary")):
            actual, expected = getattr(position, name), getattr(cpu_pos, name)
            row = {
                "coverage": measurement(
                    actual.coverage / binary.semi_major_axis**2,
                    expected.coverage / binary.semi_major_axis**2,
                    atol=64 * np.finfo(float).eps,
                ),
                "bands": {},
            }
            for b, band in enumerate(BANDS):
                selected = calculate_surface_element_fluxes(band, expected)
                full = np.zeros(len(expected.faces))
                full[expected.indices] = selected
                reference_total = selected.sum()
                face = packet["face_flux"][2 * (i % 8) + comp, b, : len(expected.faces)]
                total = packet["component_flux"][i % 8, comp, b]
                row["bands"][band] = {
                    "face_flux_over_component": measurement(
                        face / reference_total if reference_total else face,
                        full / reference_total if reference_total else full,
                        atol=2e-12,
                    ),
                    "component_flux": measurement(total, reference_total),
                }
            record["components"][name] = row
        out["phases"].append(record)
    return out


def curve_cases(device: str, *, repeats: int = 3) -> dict[str, Any]:
    started = perf_counter()
    engine = HybridBinaryLightCurve(list(BANDS), device=device, fallback="raise")
    previous = HybridBinaryLightCurve(list(BANDS), device=device, fallback="raise", radiance="cpu")
    out: dict[str, Any] = {"constructor_seconds": perf_counter() - started, "cases": []}
    for case in CASES:
        with settings.override(REFLECTION_EFFECT=case["reflection"]):
            binary = fixture(case)
            phases = contact_phases(binary)
            actual = engine(binary, phases, diagnostics=True)
            expected = Observer(passband=list(BANDS), system=binary).lc(phases=phases)[1]
            normalized = engine(binary, phases, normalize=True)
            cpu_norm = Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)[1]
            row = {
                "fixture": case,
                "phases": phases.tolist(),
                "raw": {b: measurement(actual.fluxes[b], expected[b]) for b in BANDS},
                "normalized": {b: measurement(normalized.fluxes[b], cpu_norm[b], atol=2e-12) for b in BANDS},
                "first_call_seconds": actual.timings,
                "radiance_metadata": {
                    k: actual.diagnostics["radiance"][k]
                    for k in ("emission", "temperature_source", "source_preparation")
                },
                "intermediates": intermediates(actual, binary),
                "seconds": {"original_cpu": [], "cpu_radiance_hybrid": [], "hybrid": []},
                "stages": [],
            }
            for _ in range(repeats):
                for label, function in [
                    (
                        "original_cpu",
                        lambda binary=binary, phases=phases: Observer(passband=list(BANDS), system=binary).lc(
                            phases=phases, normalize=True),
                    ),
                    ("cpu_radiance_hybrid",
                     lambda binary=binary, phases=phases: previous(binary, phases, normalize=True)),
                    ("hybrid", lambda binary=binary, phases=phases: engine(binary, phases, normalize=True)),
                ]:
                    before = perf_counter()
                    r = function()
                    row["seconds"][label].append(perf_counter() - before)
                    if label == "hybrid":
                        row["stages"].append(r.timings)
            row["median_seconds"] = {k: float(np.median(v)) for k, v in row["seconds"].items()}
            out["cases"].append(row)
    return out


def fit_problem(*, shape_parameters: bool = False) -> FitProblem:
    """Two real bands, different repeated/unsorted grids; original union mapping."""
    values = {
        "system": {
            "inclination": 86.4,
            "period": 4.0,
            "eccentricity": 0.0,
            "argument_of_periastron": 90.0,
            "mass_ratio": 0.7,
            "semi_major_axis": 13.0,
            "phase_shift": 0.017,
            "additional_light": 0.2,
        },
        "primary": {
            "t_eff": 10080.0,
            "surface_potential": 10.0,
            "synchronicity": 1.0,
            "gravity_darkening": 0.02,
            "albedo": 0.1,
            "metallicity": 0.0,
        },
        "secondary": {
            "t_eff": 10160.0,
            "surface_potential": 8.0,
            "synchronicity": 1.0,
            "gravity_darkening": 0.02,
            "albedo": 0.1,
            "metallicity": 0.0,
        },
    }
    config = {g: {k: {"value": v, "fixed": True} for k, v in group.items()} for g, group in values.items()}
    config["primary"]["t_eff"] = {"value": 10050.0, "fixed": False, "min": 10020.0, "max": 10180.0}
    config["system"]["inclination"] = {"value": 85.0, "fixed": False, "min": 83.0, "max": 89.0}
    config["nuisance"] = {"ln_f": {"value": -2.0, "fixed": not shape_parameters, "min": -5.0, "max": 1.0}}
    if shape_parameters:
        config["primary"]["surface_potential"] = {"value": 10.0, "fixed": False, "min": 9.9, "max": 10.1}
        config["system"]["mass_ratio"] = {"value": 0.7, "fixed": False, "min": 0.65, "max": 0.75}
        config["system"]["additional_light"] = {"value": 0.2, "fixed": False, "min": 0.0, "max": 0.6}
    truth = {f"{g}@{k}": v for g, group in values.items() for k, v in group.items()}
    atmosphere = dict.fromkeys(("primary", "secondary"), "ck04")
    extended = parameters.extend_json_with_atm_params(truth, atmosphere_model=atmosphere)
    grids = {
        BANDS[0]: np.array([0.5, 0, 0.005, 0.01, 0.025, 0.25, 0.49, 0.5, 1.005, -0.5, 0.75]) + 0.017,
        BANDS[1]: np.array([0, 0.01, 0.25, 0.5, 0.75, 1.0]) + 0.017,
    }
    xs, reducer = parameters.xs_reducer(grids)
    observer = Observer(passband=list(BANDS))
    observer.system_cls = BinarySystem
    flux = predict_light_curves(
        xs, {b: np.ones(len(x)) for b, x in grids.items()}, reducer, 12.0, observer, 1000, extended,
    )
    rng = np.random.default_rng(905)
    data = {
        b: LCData(
            x_data=x,
            y_data=flux[b] + rng.normal(0, 1e-4, len(x)),
            y_err=np.full(len(x), 1e-4),
            x_unit=u.dimensionless_unscaled,
            y_unit=u.dimensionless_unscaled,
        )
        for b, x in grids.items()
    }
    return FitProblem(config, data, {"discretization": 12.0, "interp_treshold": 1000, "samples": "uniform"},
                      atmosphere, None, truth)


def fitting(device: str) -> dict[str, Any]:
    problem = fit_problem(shape_parameters=True)
    original, hybrid = prepare_fit(problem), prepare_fit(problem, device=device)
    vector = normalized_vector(original)
    comparisons = []
    for current in [vector, np.full(vector.shape, 0.65), np.linspace(0.25, 0.7, len(vector))]:
        props = physical(original, current)
        a, b = hybrid.predict_light_curves(props), original.predict_light_curves(props)
        comparisons.append(
            {
                "parameters": props,
                "predictions": {band: compare(a[band], b[band]) for band in BANDS},
                "posterior": compare(
                    hybrid.ln_probability(current), original.ln_probability(current), atol=1e-6, rtol=0,
                ),
                "execution": hybrid.hybrid_synthesis.last_evaluation,
            },
        )
        assert hybrid.hybrid_synthesis.last_evaluation["backend"] == "hybrid"
    assessment = fit_assessment(device, problem=fit_problem())
    timings = {}
    paths = [("original_cpu", None, "cpu"), ("cpu_radiance_hybrid", device, "cpu"), ("hybrid", device, "device")]
    if device == "gpu":
        paths.append(("hybrid_jax_cpu", "cpu", "device"))
    for label, requested, radiance in paths:
        start = perf_counter()
        fit = prepare_fit(problem, device=requested, radiance=radiance)
        setup = perf_counter() - start
        start = perf_counter()
        fit.ln_probability(vector)
        cold = perf_counter() - start
        times = []
        for fraction in [0.0, 0.02, 0.05]:
            changed = (1 - fraction) * vector + fraction * np.linspace(0.2, 0.8, len(vector))
            start = perf_counter()
            fit.ln_probability(changed)
            times.append(perf_counter() - start)
        timings[label] = {
            "setup_seconds": setup,
            "first_callback_seconds": cold,
            "samples_seconds": times,
            "median_seconds": float(np.median(times)),
        }
    return {"predictions": comparisons, "public_fit_assessment": assessment, "likelihood_costs": timings}


def run(device: str) -> dict[str, Any]:
    start = perf_counter()
    with table_settings():
        curves = curve_cases(device)
        with settings.override(REFLECTION_EFFECT=True):
            fit = fitting(device)
    return {
        "device": device,
        "all_correctness_gates_passed": True,
        "curves": curves,
        "fitting": fit,
        "source_hashes": sources(),
        "wall_seconds": perf_counter() - start,
        "contract": ("Fresh actual CK04/Van Hamme V+bolometric curves, CPU source/LD boundaries and device "
                     "unique-spectrum quadrature/T4 interpolation/optical depth/occultation. "
                     "All transfers/host bridges included."),
        "limitations": [
            "Sparse solar-metallicity fixtures at three T/g regions; no general model accuracy/convergence claim",
            ("Original nearest gravity/metallicity, T4 interpolation and piecewise-linear LD are preserved; "
             "no continuous g/metal interpolation"),
            "Three warm samples; short chains do not establish convergence, posterior precision or ESS/sec",
        ],
    }


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
