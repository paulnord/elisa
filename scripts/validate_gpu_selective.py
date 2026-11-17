"""Complete fixed-selective curve, correction/normalization and likelihood study.

GPU use requires an independently restored <=5min allocation. This driver does
not manage production. Finite error/correction controls are never continuum
certificates, physical priors, automatically accepted models or new samplers.
"""

# ruff: noqa: INP001, S101, T201, SLF001, C901, PLR0915, B023 - synchronous diagnostic closures never escape
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import Observer, settings
from elisa.analytics.models import cost_fns, lc
from elisa.analytics.params import parameters
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.experimental.gpu_accuracy import resolution_error
from elisa.experimental.gpu_mesh_audit import mesh_measures, roche_measures
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_reference import observe
from elisa.experimental.gpu_selective_mesh import choose_plan, seed_system
from elisa.experimental.gpu_surface_error import NestedBinaryLightCurve, compare_surfaces
from elisa.observer.utils import adjust_flux_for_distance, normalize_light_curve
from scripts.validate_gpu_atmosphere import BANDS, table_settings
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_binary import contact_phases, fixture
from scripts.validate_gpu_fit import compare, make_problem, normalized_vector, physical, prepare_fit
from scripts.validate_gpu_surface_error import assert_envelope, asset_revision, mapped


def raw_curve(snapshot: Any, binary: Any) -> dict[str, np.ndarray]:
    l3 = binary.additional_light
    curves = {b: v + v.mean() * l3 / (1 - l3) for b, v in snapshot.totals.items()}
    return adjust_flux_for_distance(curves, binary.distance)


def relative_change(a: Any, b: Any) -> dict[str, float]:
    a, b = np.asarray(a), np.asarray(b)
    return {"max_abs_relative": float(np.max(np.abs((a - b) / b))), "mean_relative": float(np.mean(a) / np.mean(b) - 1)}


def finite_check(coarse: Any, fine: Any, sigma: Any, **kwargs: Any) -> dict[str, Any]:
    report = compare_surfaces(coarse, fine, sigma, allow_same_level=True, **kwargs)
    assert_envelope(report)
    stopped = False
    try:
        compare_surfaces(coarse, fine, sigma, allow_same_level=True, require_budget=True, **kwargs)
    except ResolutionNotQualifiedError:
        stopped = True
    return {"report": report, "numerical_budget_abort": stopped}


def curves(device: str, case: str, revision: str) -> list[dict[str, Any]]:
    names = ["partial_bb"] if case == "partial_bb" else ["total_bb", "partial_ck04"]
    selected = ["cpu", "gpu"] if device == "gpu" else ["cpu"]
    results = []
    for name in names:
        binary = (
            fixture({"inclination": 85 if name == "partial_bb" else 90, "discretization": 12})
            if (name != "partial_ck04")
            else table_fixture({"inclination": 88})
        )
        observer = Observer(passband=list(BANDS), system=binary)
        phases = contact_phases(binary)
        snapshots, rows = {}, []
        t = perf_counter()
        angular = {n: [roche_measures(binary, n, order) for order in [16, 32]] for n in ["primary", "secondary"]}
        angular_seconds = perf_counter() - t
        for controls in angular.values():
            compare(
                [controls[0][k] for k in ["area", "volume"]],
                [controls[1][k] for k in ["area", "volume"]],
                rtol=2e-11,
                atol=0,
            )

        def evaluate(label: str, mesh: Any) -> None:
            t = perf_counter()
            expected_raw = observe(observer, phases, normalize=False, mesh=mesh)[1]
            reference_seconds = perf_counter() - t
            # Actual original raw prediction, followed by original maximum
            # normalization; direct original normalized controls remain below.
            expected = normalize_light_curve(expected_raw, kind="maximum", top_fraction_to_average=0.0)[0]
            system = seed_system(binary)
            info = mesh.apply(system)
            geometry = {n: mesh_measures(getattr(system, n), n) for n in ["primary", "secondary"]}
            for n, row in geometry.items():
                row["relative_to_angular32"] = {
                    k: row[k] / angular[n][1]["area" if "area" in k else "volume"] - 1
                    for k in ["raw_area", "raw_volume", "corrected_area", "corrected_volume"]
                }
            row = {
                "name": label,
                "mesh": asdict(mesh),
                "geometry": geometry,
                "selected_quarter_roots": {n: info[n].get("selected_quarter_roots") for n in info},
                "reference_original_cpu_raw_seconds": reference_seconds,
                "evaluations": {},
            }
            if label == "uniform2":
                direct = observe(observer, phases, normalize=True, mesh=mesh)[1]
                row["original_raw_to_direct_normalized_control"] = {b: compare(expected[b], direct[b]) for b in BANDS}
            snapshots[label] = {}
            for d in selected:
                engine = NestedBinaryLightCurve(
                    list(BANDS), mesh, device=d, radiance="cpu" if name == "partial_ck04" else "device",
                )
                t = perf_counter()
                snapshot = engine.snapshot(binary, phases, observer=observer, source_revision=revision)
                raw = raw_curve(snapshot, binary)
                elapsed = perf_counter() - t
                snapshots[label][d] = snapshot
                row["evaluations"][d] = {
                    "complete_snapshot_and_raw_seconds": elapsed,
                    "curve_seconds": snapshot.curve_seconds,
                    "face_counts": snapshot.face_counts,
                    "raw_parity": {b: compare(raw[b], expected_raw[b], atol=0) for b in BANDS},
                    "normalized_parity": {b: compare(snapshot.fluxes[b], expected[b]) for b in BANDS},
                    "raw_fluxes": {b: v.tolist() for b, v in raw.items()},
                    "normalized_fluxes": {b: v.tolist() for b, v in snapshot.fluxes.items()},
                }
            rows.append(row)

        evaluate("uniform2", NestedMesh(2))
        evaluate("uniform3", NestedMesh(3))
        t = perf_counter()
        pilot = [snapshots[n]["cpu"] for n in ["uniform2", "uniform3"]]
        main = choose_plan(binary, *pilot, 1e-4, fraction=0.5)
        plan_seconds = perf_counter() - t
        plans = [("selective50", main)]
        if name == "partial_bb":
            plans += [
                (f"selective{int(fraction * 100)}", choose_plan(binary, *pilot, 1e-4, fraction=fraction))
                for fraction in [0.25, 0.75]
            ]
            rng = np.random.default_rng(12012)
            roots = tuple(
                tuple(
                    sorted(
                        int(i)
                        for i in rng.permutation(pilot[0].groups[BANDS[0]][j].shape[1] // 4)[: len(main.roots[j])]
                    ),
                )
                for j in range(2)
            )
            plans += [
                ("random50_root_count_control", replace(main, roots=roots)),
                ("selective50_coarse_correction", replace(main, correction="coarse")),
                ("selective50_fine_correction", replace(main, correction="fine")),
            ]
        for label, mesh in plans:
            evaluate(label, mesh)
        comparisons = {}
        for label, _ in plans:
            comparisons[label] = {}
            for d in selected:
                a, b = snapshots[label][d], snapshots["uniform3"][d]
                comparisons[label][d] = finite_check(a, b, 1e-4, ln_f=-2.0)
                comparisons[label][d]["raw_change_to_uniform3"] = {
                    band: relative_change(raw_curve(a, binary)[band], raw_curve(b, binary)[band]) for band in BANDS
                }
        base_pair = {
            d: finite_check(snapshots["uniform2"][d], snapshots["uniform3"][d], 1e-4, ln_f=-2.0) for d in selected
        }
        # Different finite triangulation; never the implementation oracle or truth.
        binary.primary.discretization_factor = binary.secondary.discretization_factor = np.radians(1.5)
        withheld = observe(Observer(passband=list(BANDS), system=binary), phases)[1]
        cross = {
            label: {b: resolution_error(snapshot[selected[-1]].fluxes[b], withheld[b], 1e-4, ln_f=-2.0) for b in BANDS}
            for label, snapshot in snapshots.items()
        }
        results.append(
            {
                "case": name,
                "phases": phases.tolist(),
                "angular_controls": angular,
                "angular_controls_seconds": angular_seconds,
                "plan_freezing_seconds": plan_seconds,
                "models": rows,
                "selective_to_uniform3": comparisons,
                "uniform2_to_uniform3": base_pair,
                "different_native_1p5deg_comparisons": cross,
                "limit": "Fixed pilot masks; fresh full coupled models. Geometric angular convergence and finite "
                "curve/enclosure controls are separate. No continuum certificate or fit/ESS claim.",
            },
        )
    return results


def prepare_binary(fit: Any, props: Any) -> tuple[Any, Any]:
    phases, values = time_layer_resolver(fit.x_data_reduced, **props)
    extended = parameters.extend_json_with_atm_params(
        values, atmosphere_model=fit.atmosphere_model, limb_darkening_coefficients=fit.limb_darkening_coefficients,
    )
    extended["secondary@discretization_factor"] = fit.discretization
    return phases, lc.prepare_binary(discretization=fit.discretization, **extended)


def observations(device: str, revision: str) -> list[dict[str, Any]]:
    selected = ["cpu", "gpu"] if device == "gpu" else ["cpu"]
    results = []
    for jd in [False, True]:
        fit = prepare_fit(make_problem(jd=jd, free_nuisance=True), phase_mode="observations", mesh_mode="paired")
        base = normalized_vector(fit)
        phases, binary = prepare_binary(fit, physical(fit, base))
        t = perf_counter()
        pilot = [
            NestedBinaryLightCurve(list(fit.y_data), NestedMesh(level), device="cpu").snapshot(
                binary,
                phases,
                observer=fit.observer,
                source_revision=revision,
            )
            for level in [2, 3]
        ]
        frozen = choose_plan(binary, *pilot, 1e-4, fraction=0.5)
        pilot_seconds = perf_counter() - t
        numerical = {"selective": frozen, "uniform3": NestedMesh(3)}
        engines = {
            d: {label: NestedBinaryLightCurve(list(fit.y_data), mesh, device=d) for label, mesh in numerical.items()}
            for d in selected
        }
        proposals = [base]
        if not jd:
            changed = base.copy()
            for key, value in {
                "primary@t_eff": 0.45,
                "system@inclination": 0.5,
                "system@additional_light": 0.45,
                "nuisance@ln_f": 0.6,
            }.items():
                changed[list(fit.fitable).index(key)] = value
            proposals.append(changed)
        candidates = []
        for vector in proposals:
            props = physical(fit, vector)
            phases, binary = prepare_binary(fit, props)
            prior = fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas)
            row = {"physical": props, "prior": float(prior), "reference": {}, "evaluations": {}}
            expected = {}
            for label, mesh in numerical.items():

                def synthesis(
                    phases: Any, discretization: float, observer: Any, *, _mesh: Any = mesh, **kwargs: Any,
                ) -> dict[str, np.ndarray]:
                    observer._system = lc.prepare_binary(discretization=discretization, **kwargs)
                    return observe(observer, phases, mesh=_mesh)[1]

                t = perf_counter()
                expected[label] = fit.predict_light_curves(props, synthesis=synthesis)
                density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, expected[label], props["nuisance@ln_f"]) + prior
                row["reference"][label] = {
                    "complete_prediction_density_seconds": perf_counter() - t,
                    "log_posterior": float(density),
                }
            for d in selected:
                snapshots, models = {}, {}
                for label in numerical:
                    t = perf_counter()
                    snapshot = engines[d][label].snapshot(
                        binary, phases, observer=fit.observer, source_revision=revision,
                    )
                    prediction = mapped(snapshot, fit)
                    density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, prediction, props["nuisance@ln_f"]) + prior
                    elapsed = perf_counter() - t
                    models[label] = {
                        "complete_snapshot_mapping_density_seconds": elapsed,
                        "face_counts": snapshot.face_counts,
                        "prediction_parity": {b: compare(prediction[b], expected[label][b]) for b in prediction},
                        "posterior_parity": compare(
                            density,
                            row["reference"][label]["log_posterior"],
                            rtol=0,
                            atol=1e-6,
                        ),
                    }
                    snapshots[label] = snapshot
                finite = finite_check(
                    snapshots["selective"],
                    snapshots["uniform3"],
                    fit.y_err,
                    reducers=fit.x_data_reducer,
                    mean_normalize=True,
                    observed=fit.y_data,
                    ln_f=props["nuisance@ln_f"],
                )
                actual = row["reference"]["uniform3"]["log_posterior"] - row["reference"]["selective"]["log_posterior"]
                parity = compare(finite["report"]["joint"]["delta_log_likelihood"], actual, rtol=0, atol=1e-6)
                row["evaluations"][d] = {"models": models, "finite_change": finite, "likelihood_change_parity": parity}
            candidates.append(row)
        assert np.isneginf(fit.ln_prior(np.full(len(base), -0.1), fit.norm_init_vector, fit.sigmas))
        results.append(
            {
                "jd": jd,
                "fixed_plan": asdict(frozen),
                "pilot_and_plan_seconds": pilot_seconds,
                "candidates": candidates,
                "out_of_prior": "Original -inf, no nested evaluation",
                "limit": "Fixed numerical plan across temperature/inclination/thirdlight/noise proposals; "
                "seed topology changes abort outside candidate handlers. No new fit or sampler backend.",
            },
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--part", choices=["curves", "observations"], required=True)
    parser.add_argument("--case", choices=["partial_bb", "others"], default="partial_bb")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    revision = asset_revision()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        results = (
            curves(args.device, args.case, revision) if args.part == "curves" else observations(args.device, revision)
        )
    report = {
        "device": args.device,
        "part": args.part,
        "case": args.case,
        "source_revision": revision,
        "all_implementation_parity_gates_passed": True,
        "wall_seconds": perf_counter() - started,
        "results": results,
        "contract": "Fresh full selective curves/snapshots/likelihoods, including CPU roots/coverage/grouping and "
        "all per-call transfers. Constructors in main wall, outside model timers. Main wall starts after imports. "
        "Numerical plan is fixed; no continuum, qualified fitting or MCMC/ESS claim.",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: report[k] for k in ["part", "case", "wall_seconds", "all_implementation_parity_gates_passed"]}),
    )


if __name__ == "__main__":
    main()
