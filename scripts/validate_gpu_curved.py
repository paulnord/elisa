"""Complete true-Roche quadrature, finite field/boundary controls and likelihood.

GPU use requires an independently restored bounded allocation. This driver never
pauses/restores production. Original finite thermal/reflection/radiance fields
are retained; convergence of the continuous fields is a separate open question.
"""

# ruff: noqa: INP001, S101, T201 - standalone scientific diagnostic
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import Observer, settings
from elisa.analytics.models import cost_fns, lc
from elisa.experimental.curved_geometry import CurvedRule, RocheGeometry, duffy_nodes
from elisa.experimental.gpu_curved import CurvedBinaryLightCurve, CurvedCurveResult, curved_difference
from elisa.experimental.gpu_mesh_audit import roche_measures
from elisa.experimental.gpu_nested_mesh import NestedMesh
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_reference import observe
from elisa.experimental.gpu_selective_mesh import seed_system
from scripts.validate_gpu_atmosphere import BANDS, table_settings
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_binary import contact_phases, fixture
from scripts.validate_gpu_fit import compare, make_problem, normalized_vector, physical, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision, mapped


def reload_result(payload: dict[str, Any]) -> CurvedCurveResult:
    values = dict(payload)
    values["phases"] = np.asarray(values["phases"])
    for name in ["fluxes", "raw_fluxes", "component_fluxes"]:
        values[name] = {b: np.asarray(v) for b, v in values[name].items()}
    return CurvedCurveResult(**values)


def finite(a: Any, b: Any, sigma: Any, **kwargs: Any) -> dict[str, Any]:
    r = curved_difference(a, b, sigma, **kwargs)
    for row in r["bands"].values():
        assert np.all(np.abs(row["actual_change"]) <= row["component_envelope"])
        assert row["reconstruction_max_abs"] <= row["roundoff_guard_max"]
    if r["finite_log_likelihood_absolute_bound"] is not None:
        assert abs(r["actual"]["delta_log_likelihood"]) <= r["finite_log_likelihood_absolute_bound"]
    stopped = False
    try:
        curved_difference(a, b, sigma, require_budget=True, **kwargs)
    except ResolutionNotQualifiedError:
        stopped = True
    return {"report": r, "numerical_budget_abort": stopped}


def curve_profile(args: Any, revision: str) -> dict[str, Any]:
    inc = {"partial_bb": 85, "total_bb": 90, "partial_ck04": 88}[args.case]
    b = fixture({"inclination": inc, "discretization": 12}) if args.case != "partial_ck04" else table_fixture(
        {"inclination": inc},
    )
    phases = contact_phases(b)
    observer = Observer(passband=list(BANDS), system=b)
    rule = CurvedRule(args.order, args.depth)
    selected = "gpu" if args.device == "gpu" else "cpu"
    results, costs = {}, {}
    for d in ["reference", selected]:
        t = perf_counter()
        engine = CurvedBinaryLightCurve(list(BANDS), rule, field_levels=args.field_levels, device=d,
                                       radiance="cpu" if args.case == "partial_ck04" else "device")
        results[d] = engine(b, phases, observer=observer, source_revision=revision)
        costs[d] = perf_counter()-t
    reference, actual = results["reference"], results[selected]
    assert actual.signature == reference.signature
    assert actual.diagnostics["phase_records"] == reference.diagnostics["phase_records"]
    parity = {k: {band: compare(getattr(actual, k)[band], getattr(reference, k)[band],
                               atol=2e-12 if k == "fluxes" else 0) for band in BANDS}
              for k in ["fluxes", "raw_fluxes", "component_fluxes"]}
    finite_change = None
    if args.comparison:
        previous = json.loads(args.comparison.read_text())["result"]
        assert previous["case"] == args.case
        finite_change = {d: finite(reload_result(previous["models"][d]), results[d], 1e-4, ln_f=-2.)
                         for d in ["reference", selected]}
    # Independent area audit. It certifies neither eclipses nor the finite field.
    system = seed_system(b)
    NestedMesh(args.field_levels).apply(system)
    area = {}
    for name, x in [("primary", 0.), ("secondary", 1.)]:
        star = getattr(system, name)
        p = star.points-np.array([x, 0, 0])
        p /= np.linalg.norm(p, axis=1)[:, None]
        vectors, weights = duffy_nodes(p[star.faces], args.order)
        _, _, jacobian = RocheGeometry(b, name).sample(vectors)
        reference_area = roche_measures(b, name, 32)
        area[name] = {"curved_area_SMA2": float(np.sum(weights*jacobian)), "angular32": reference_area,
                      "relative_area_change": float(np.sum(weights*jacobian)/reference_area["area"]-1)}
    native = observe(observer, phases, mesh=NestedMesh(args.field_levels))[1]
    from elisa.experimental.gpu_accuracy import resolution_error  # noqa: PLC0415

    native_change = {band: resolution_error(actual.fluxes[band], native[band], 1e-4, ln_f=-2.) for band in BANDS}
    return {"case": args.case, "rule": asdict(rule), "field_levels": args.field_levels,
            "models": {d: asdict(r) for d, r in results.items()}, "complete_constructor_curve_seconds": costs,
            "parity": parity, "finite_change": finite_change, "independent_curved_area_controls": area,
            "same_finite_fields_original_faceted_curve_change": native_change,
            "limit": "Curved geometry replaces original faceted integration. CPU parity oracle is the same new "
            "quadrature applied to original finite fields, not an unchanged original whole-curve prediction. "
            "Surface order, implicit-boundary depth and upstream field grids are separately controlled."}


def observation_profile(args: Any, revision: str) -> list[dict[str, Any]]:
    rows = []
    selected = "gpu" if args.device == "gpu" else "cpu"
    for jd in [False, True]:
        fit = prepare_fit(make_problem(jd=jd, free_nuisance=True), phase_mode="observations", mesh_mode="paired")
        base = normalized_vector(fit)
        proposals = [base]
        if not jd:
            changed = base.copy()
            for key, v in {"primary@t_eff": .45, "system@inclination": .5,
                           "system@additional_light": .45, "nuisance@ln_f": .6}.items():
                changed[list(fit.fitable).index(key)] = v
            proposals.append(changed)
        engine = CurvedBinaryLightCurve(list(fit.y_data), CurvedRule(3, 3), field_levels=1, device=selected)
        original = CurvedBinaryLightCurve(list(fit.y_data), CurvedRule(3, 3), field_levels=1, device="reference")
        coarse = CurvedBinaryLightCurve(list(fit.y_data), CurvedRule(3, 2), field_levels=1, device="reference")
        candidates = []
        for vector in proposals:
            props = physical(fit, vector)
            saved = []
            def synthesis(phases: Any, discretization: float, observer: Any, *, _original: Any = original,
                          _saved: list = saved, **kwargs: Any) -> dict[str, Any]:
                model = lc.prepare_binary(discretization=discretization, **kwargs)
                r = _original(model, phases, observer=observer, source_revision=revision)
                _saved.append(r)
                return r.fluxes

            prior = fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas)
            t = perf_counter()
            expected = fit.predict_light_curves(props, synthesis=synthesis)
            density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, expected, props["nuisance@ln_f"])+prior
            reference_seconds = perf_counter()-t
            phases, binary = prepare_binary(fit, props)
            t = perf_counter()
            result = engine(binary, phases, observer=fit.observer, source_revision=revision)
            prediction = mapped(result, fit)
            actual = cost_fns.likelihood_fn(fit.y_data, fit.y_err, prediction, props["nuisance@ln_f"])+prior
            seconds = perf_counter()-t
            c = coarse(binary, phases, observer=fit.observer, source_revision=revision)
            report = finite(c, saved[0], fit.y_err, reducers=fit.x_data_reducer, mean_normalize=True,
                            observed=fit.y_data, ln_f=props["nuisance@ln_f"])
            candidates.append({"physical": props, "reference_prediction_density_seconds": reference_seconds,
                               "selected_prediction_density_seconds": seconds, "device": selected,
                               "prediction_parity": {b: compare(prediction[b], expected[b]) for b in prediction},
                               "posterior_parity": compare(actual, density, atol=1e-6, rtol=0),
                               "boundary_preflight": report})
        assert np.isneginf(fit.ln_prior(np.full(len(base), -.1), fit.norm_init_vector, fit.sigmas))
        rows.append({"jd": jd, "candidates": candidates,
                     "out_of_prior": "Original -inf before curved synthesis",
                     "limit": "Original phase/JD/serializer/constraint/prior/thirdlight/mean/noise semantics with "
                     "new finite curved integration; no registered sampler backend or complete field certificate."})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--part", choices=["curves", "observations"], required=True)
    p.add_argument("--case", choices=["partial_bb", "total_bb", "partial_ck04"], default="partial_bb")
    p.add_argument("--field-levels", type=int, default=1)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--comparison", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = perf_counter()
    revision = asset_revision()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = curve_profile(args, revision) if args.part == "curves" else observation_profile(args, revision)
    report = {"part": args.part, "device": args.device, "source_revision": revision,
              "all_implementation_parity_gates_passed": True, "result": result, "wall_seconds": perf_counter()-started,
              "contract": "Complete fresh curves/densities, including original finite fields, CPU true geometry, "
              "adaptive partition/implicit eclipse masks and all transfers. Constructor costs in curve "
              "timers, reused constructors in observation mainwall. Mainwall starts after imports. "
              "No continuum field or boundary error certificate; no full-fit, convergence or ESS claim."}
    def convert(value: Any) -> Any:
        return value.tolist() if isinstance(value, np.ndarray) else value.item()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=convert)+"\n")
    keys = ["part", "device", "wall_seconds", "all_implementation_parity_gates_passed"]
    print(json.dumps({k: report[k] for k in keys}))


if __name__ == "__main__":
    main()
