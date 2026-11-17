"""Fresh spots/eccentric/contact CPU physics with opt-in device flux callbacks.

Run GPU only inside the independently restored bounded allocation. Reference
physics/coverage/approximations are unmodified ELISa. Full costs are reported;
the broader path ports integration only and is not all-GPU synthesis.
"""

# ruff: noqa: INP001, S101, T201 - standalone scientific driver/import bootstrap
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

from elisa import BinarySystem, Observer, Star, settings
from elisa import units as u
from elisa.analytics import LCData
from elisa.analytics.binary_fit.lc_prediction import predict_light_curves
from elisa.analytics.params import parameters
from elisa.base.curves.utils import calculate_surface_element_fluxes
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from scripts.validate_gpu_atmosphere import sources, table_settings
from scripts.validate_gpu_binary import measurement
from scripts.validate_gpu_fit import FitProblem, compare, fit_assessment, normalized_vector, physical, prepare_fit

BANDS = ("Generic.Bessell.V", "bolometric")
CASES = [
    {"name": "synchronous_spot", "spots": True, "reflection": True},
    {"name": "asynchronous_spot", "spots": True, "spin": 1.1, "reflection": False},
    {"name": "eccentric_exact", "eccentricity": 0.2, "reflection": True},
    {"name": "eccentric_configured_neighbours", "eccentricity": 0.2, "approximation": True, "reflection": True},
    {"name": "eccentric_spot", "eccentricity": 0.15, "spots": True, "spin": 1.1, "reflection": False},
    {"name": "near_contact", "q": 1.0, "potentials": [3.9, 3.9], "reflection": True},
    {"name": "overcontact", "q": 1.0, "potentials": [3.5, 3.5], "reflection": True},
    {
        "name": "spotted_ck04",
        "spots": True,
        "atlas": "ck04",
        "potentials": [10.0, 8.0],
        "temperatures": [10080.0, 10160.0],
        "period": 4.0,
        "reflection": True,
    },
]


def case_settings(case: dict[str, Any]) -> Any:
    return settings.override(
        REFLECTION_EFFECT=case["reflection"],
        LIMB_DARKENING_LAW=case.get("law", "linear"),
        USE_INTERPOLATION_APPROXIMATION=False,
        USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION=False,
        USE_SIMILAR_NEIGHBOURS_APPROXIMATION=case.get("approximation", False),
        MAX_D_FLUX=0.01 if case.get("approximation", False) else 2e-4,
    )


def fixture(case: dict[str, Any]) -> BinarySystem:
    q, potentials = case.get("q", 0.7), case.get("potentials", [8.0, 12.0])
    atlas, temps = case.get("atlas", "bb"), case.get("temperatures", [6500.0, 5000.0])
    coefficient = [0.4] if case.get("law", "linear") == "linear" else [0.3, 0.2]
    ld = dict.fromkeys(BANDS, coefficient) if atlas == "bb" else None
    common = {
        "atmosphere": atlas,
        "albedo": 0.1 if atlas == "ck04" else 0.5,
        "gravity_darkening": 0.02 if atlas == "ck04" else 0.32,
        "discretization_factor": 12,
        "limb_darkening_coefficients": ld,
    }
    spots = [
        {
            "longitude": case.get("longitude", 45),
            "latitude": 70,
            "angular_radius": 20,
            "temperature_factor": 1.005 if atlas == "ck04" else case.get("temperature_factor", 0.9),
        },
    ]
    return BinarySystem(
        Star(
            mass=1.2,
            t_eff=temps[0],
            surface_potential=potentials[0],
            synchronicity=case.get("spin", 1),
            spots=spots if case.get("spots", False) else None,
            **common,
        ),
        Star(mass=1.2 * q, t_eff=temps[1], surface_potential=potentials[1], synchronicity=1, **common),
        period=case.get("period", 3),
        inclination=case.get("inclination", 88),
        eccentricity=case.get("eccentricity", 0),
        argument_of_periastron=60,
        phase_shift=0.017,
        additional_light=0.2,
        distance=25,
    )


def phases_for(binary: Any) -> np.ndarray:
    """Actual CPU conjunction phases plus samples on either side and quadratures."""
    conjunctions = binary.orbit.conjunctions
    secondary = (conjunctions["secondary_eclipse"]["true_phase"] - conjunctions["primary_eclipse"]["true_phase"]) % 1
    offsets = np.array([-0.035, -0.01, -0.002, 0, 0.002, 0.01, 0.035])
    phases = np.concatenate((offsets, secondary + offsets, [0.25, 0.75, 1.002, -0.5]))
    return phases[::-1] + binary.phase_shift


def face_checks(result: Any) -> dict[str, Any]:
    records = result.diagnostics["records"]
    max_face = max_component = 0.0
    distances, nfaces, spot_counts = [], [], []
    for record in records:
        system = record.system
        distances.append(float(system.position.distance))
        nfaces.append([len(system.primary.faces), len(system.secondary.faces)])
        spot_counts.append([len(system.primary.spots), len(system.secondary.spots)])
        for b, band in enumerate(BANDS):
            for comp, star in enumerate((system.primary, system.secondary)):
                expected = calculate_surface_element_fluxes(band, star)
                count, row = record.selected_counts[comp], 2 * b + comp
                total = expected.sum()
                actual = record.face_flux[row, :count]
                error = measurement(
                    actual / total if total else actual, expected / total if total else expected, atol=2e-12,
                )
                max_face = max(max_face, error["max_abs"])
                error = measurement(record.component_flux[b, comp], total)
                max_component = max(max_component, error["max_relative_nonzero"])
    return {
        "callbacks": len(records),
        "max_face_flux_over_component_abs": max_face,
        "max_component_flux_relative": max_component,
        "distances": distances,
        "face_counts": nfaces,
        "spot_counts": spot_counts,
        "max_padded_faces": result.diagnostics["max_padded_faces"],
        "route": result.diagnostics["route"],
    }


def curve_cases(device: str) -> dict[str, Any]:
    before = perf_counter()
    engine = HybridBinaryLightCurve(list(BANDS), device=device, general="device_integration", fallback="raise")
    fallback = HybridBinaryLightCurve(list(BANDS), device=device)
    out: dict[str, Any] = {"constructor_seconds": perf_counter() - before, "cases": []}
    for case in CASES:
        with case_settings(case):
            binary = fixture(case)
            phases = phases_for(binary)
            actual = engine(binary, phases, diagnostics=True)
            expected = Observer(passband=list(BANDS), system=binary).lc(phases=phases)[1]
            normalized = engine(binary, phases, normalize=True)
            cpu_norm = Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)[1]
            cpu_fallback = fallback(binary, phases, normalize=True)
            assert actual.backend == "hybrid_native"
            assert cpu_fallback.backend == "cpu_fallback"
            row = {
                "fixture": case,
                "morphology": binary.morphology,
                "phases": phases.tolist(),
                "raw": {b: measurement(actual.fluxes[b], expected[b]) for b in BANDS},
                "normalized": {b: measurement(normalized.fluxes[b], cpu_norm[b], atol=2e-12) for b in BANDS},
                "default_fallback": {
                    b: measurement(cpu_fallback.fluxes[b], cpu_norm[b], rtol=0, atol=0) for b in BANDS
                },
                "first_call_seconds": actual.timings,
                "intermediates": face_checks(actual),
                "seconds": {"original_cpu": [], "hybrid_native": [], "default_cpu_fallback": []},
                "stages": [],
            }
            if case.get("approximation", False):
                with case_settings({**case, "approximation": False}):
                    rigorous = Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)[1]
                row["original_configured_approximation_error"] = {
                    b: float(np.max(np.abs(cpu_norm[b] - rigorous[b]))) for b in BANDS
                }
            for _ in range(3):
                start = perf_counter()
                Observer(passband=list(BANDS), system=binary).lc(phases=phases, normalize=True)
                row["seconds"]["original_cpu"].append(perf_counter() - start)
                start = perf_counter()
                warm = engine(binary, phases, normalize=True)
                row["seconds"]["hybrid_native"].append(perf_counter() - start)
                row["stages"].append(warm.timings)
                start = perf_counter()
                fallback(binary, phases, normalize=True)
                row["seconds"]["default_cpu_fallback"].append(perf_counter() - start)
            row["median_seconds"] = {k: float(np.median(v)) for k, v in row["seconds"].items()}
            out["cases"].append(row)
    return out


def fit_problem(family: str, *, shape: bool = False) -> FitProblem:
    over = family == "overcontact"
    spot = family == "spot"
    values = {
        "system": {
            "inclination": 86.4,
            "period": 3.0,
            "eccentricity": 0.15 if family == "eccentric" else 0.0,
            "argument_of_periastron": 60.0,
            "mass_ratio": 1.0 if over else 0.7,
            "semi_major_axis": 10.0,
            "phase_shift": 0.017,
            "additional_light": 0.2,
        },
        "primary": {
            "t_eff": 6500.0,
            "surface_potential": 3.5 if over else 8.0,
            "synchronicity": 1.1 if spot else 1.0,
            "gravity_darkening": 0.32,
            "albedo": 0.5,
            "metallicity": 0.0,
        },
        "secondary": {
            "t_eff": 5000.0,
            "surface_potential": 3.5 if over else 12.0,
            "synchronicity": 1.0,
            "gravity_darkening": 0.32,
            "albedo": 0.5,
            "metallicity": 0.0,
        },
    }
    cfg = {g: {k: {"value": v, "fixed": True} for k, v in group.items()} for g, group in values.items()}
    cfg["primary"]["t_eff"] = {"value": 6300.0, "fixed": False, "min": 6000.0, "max": 7000.0}
    cfg["system"]["inclination"] = {"value": 85.0, "fixed": False, "min": 83.0, "max": 89.0}
    cfg["nuisance"] = {"ln_f": {"value": -2.0, "fixed": not shape, "min": -5.0, "max": 1.0}}
    truth = {f"{g}@{k}": v for g, group in values.items() for k, v in group.items()}
    if over:
        cfg["primary"]["surface_potential"] = {"value": 3.5, "fixed": False, "min": 3.45, "max": 3.55}
        cfg["secondary"]["surface_potential"] = {"constraint": "primary@surface_potential"}
    if spot:
        props = {"longitude": 45.0, "latitude": 70.0, "angular_radius": 20.0, "temperature_factor": 0.9}
        cfg["primary"]["spots"] = [{"label": "cool", **{k: {"value": v, "fixed": True} for k, v in props.items()}}]
        truth.update({f"primary@spot@cool@{k}": v for k, v in props.items()})
        if shape:
            cfg["primary"]["spots"][0]["longitude"] = {"value": 45.0, "fixed": False, "min": 40.0, "max": 50.0}
    if shape:
        cfg["system"]["additional_light"] = {"value": 0.2, "fixed": False, "min": 0.0, "max": 0.6}
        if family == "eccentric":
            cfg["system"]["eccentricity"] = {"value": 0.15, "fixed": False, "min": 0.12, "max": 0.18}
        elif over:
            cfg["primary"]["surface_potential"] = {"value": 3.5, "fixed": False, "min": 3.45, "max": 3.55}
    atm = dict.fromkeys(("primary", "secondary"), "bb")
    ld = {s: {key: [0.4] for key in BANDS} for s in ("primary", "secondary")}
    extended = parameters.extend_json_with_atm_params(truth, atmosphere_model=atm, limb_darkening_coefficients=ld)
    grids = {
        BANDS[0]: np.array([0.5, 0, 0.015, 0.25, 0.6, 1.015]) + 0.017,
        BANDS[1]: np.array([0, 0.015, 0.5, -0.5, 0.75]) + 0.017,
    }
    xs, reducer = parameters.xs_reducer(grids)
    observer = Observer(passband=list(BANDS))
    observer.system_cls = BinarySystem
    flux = predict_light_curves(
        xs, {b: np.ones(len(x)) for b, x in grids.items()}, reducer, 12.0, observer, 1000, extended,
    )
    rng = np.random.default_rng(906)
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
    return FitProblem(
        cfg, data, {"discretization": 12.0, "interp_treshold": 1000, "samples": "uniform"}, atm, ld, truth,
    )


def fitting(device: str) -> dict[str, Any]:
    out = {}
    for family in ["spot", "eccentric", "overcontact"]:
        morphology = "over-contact" if family == "overcontact" else "detached"
        problem = fit_problem(family, shape=True)
        original = prepare_fit(problem, morphology=morphology)
        hybrid = prepare_fit(problem, device=device, general="device_integration", morphology=morphology)
        base = normalized_vector(original)
        row: dict[str, Any] = {"predictions": []}
        for current in [base, np.full(base.shape, 0.65), np.linspace(0.25, 0.7, len(base))]:
            props = physical(original, current)
            a, b = hybrid.predict_light_curves(props), original.predict_light_curves(props)
            row["predictions"].append(
                {
                    "parameters": props,
                    "predictions": {band: compare(a[band], b[band]) for band in BANDS},
                    "posterior": compare(
                        hybrid.ln_probability(current), original.ln_probability(current), atol=1e-6, rtol=0,
                    ),
                    "execution": hybrid.hybrid_synthesis.last_evaluation,
                },
            )
            assert hybrid.hybrid_synthesis.last_evaluation["backend"] == "hybrid_native"
        if family in ("spot", "overcontact"):
            row["public_fit_assessment"] = fit_assessment(
                device, problem=fit_problem(family), general="device_integration", morphology=morphology,
            )
        row["likelihood_costs"] = {}
        for label, fit in [("original_cpu", original), ("hybrid_native", hybrid)]:
            samples = []
            for f in [0.0, 0.02, 0.05]:
                current = (1 - f) * base + f * np.linspace(0.2, 0.8, len(base))
                start = perf_counter()
                fit.ln_probability(current)
                samples.append(perf_counter() - start)
            row["likelihood_costs"][label] = {"samples_seconds": samples, "median_seconds": float(np.median(samples))}
        out[family] = row
    return out


def run(device: str) -> dict[str, Any]:
    started = perf_counter()
    with table_settings():
        curves = curve_cases(device)
        with case_settings({"reflection": False}):
            fits = fitting(device)
    return {
        "device": device,
        "all_correctness_gates_passed": True,
        "curves": curves,
        "fitting": fits,
        "source_hashes": sources(),
        "wall_seconds": perf_counter() - started,
        "contract": "Original CPU physical routers/coverage/approximations, fresh all-band device flux integration, "
        "original Observer and physical fitter mapping/statistics. All transfers included.",
        "limitations": [
            "Broader bridge moves integration only; CPU geometry, spots, reflection, radiance/LD and coverage remain",
            "Configured native approximations retain original semantics; accuracy is separate from parity",
            "Short chains are callback/move replay, not convergence or ESS/sec",
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
