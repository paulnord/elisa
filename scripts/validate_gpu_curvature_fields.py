"""Complete curvature-consistent finite fields, shared curves and likelihood controls.

GPU execution requires independently restored bounded allocations; this driver
never changes production. Fresh curved-cell field models share each clipped packet.
"""

# ruff: noqa: INP001, S101, T201, PLR2004 - standalone scientific driver
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import Observer, settings
from elisa.analytics.models import cost_fns, lc
from elisa.experimental.boundary_geometry import BoundaryRule
from elisa.experimental.gpu_curvature_fields import CurvatureFieldLightCurve, CurvatureFieldSpec
from elisa.experimental.gpu_mesh_audit import roche_measures
from scripts.validate_gpu_atmosphere import BANDS, table_settings
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_binary import contact_phases, fixture
from scripts.validate_gpu_boundary import finite, reload_result
from scripts.validate_gpu_fit import compare, make_problem, normalized_vector, physical, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision, mapped


def specs(args: Any) -> tuple[CurvatureFieldSpec, ...]:
    return tuple(CurvatureFieldSpec(v) for v in args.levels)



def field_controls(result: Any, sigma: Any, **kwargs: Any) -> dict[str, Any]:
    keys = list(result.curves)
    return {f"{a}_to_{b}": finite(result.curves[a], result.curves[b], sigma, **kwargs)
            for a, b in pairwise(keys)}


def parity(reference: Any, selected: Any) -> dict[str, Any]:
    assert reference.quadrature_sha256 == selected.quadrature_sha256
    assert reference.diagnostics["phase_records"] == selected.diagnostics["phase_records"]
    return {name: {key: {b: compare(getattr(selected.curves[name], key)[b],
                                   getattr(reference.curves[name], key)[b], atol=2e-12 if key == "fluxes" else 0)
                        for b in reference.curves[name].fluxes}
                   for key in ["fluxes", "raw_fluxes", "component_fluxes"]} for name in reference.curves}


def engine(args: Any, bands: list[str], backend: str, *, coarse: bool = False) -> CurvatureFieldLightCurve:
    return CurvatureFieldLightCurve(bands, BoundaryRule(args.order, args.depth-int(coarse)),
                                   levels=(args.levels[-1],) if coarse else tuple(args.levels),
                                   integration_level=args.integration_level, device=backend,
                                   radiance="cpu" if args.case == "partial_ck04" else "device")


def curve_profile(args: Any, revision: str) -> dict[str, Any]:
    inc = {"partial_bb": 85, "total_bb": 90, "partial_ck04": 88, "small_occultor": 85}[args.case]
    options = {"inclination": inc, "discretization": 12}
    if args.case == "small_occultor":
        options["potentials"] = [8, 100]
    binary = table_fixture({"inclination": inc}) if args.case == "partial_ck04" else fixture(options)
    phases = contact_phases(binary)
    observer = Observer(passband=list(BANDS), system=binary)
    selected = "gpu" if args.device == "gpu" else "cpu"
    results, costs = {}, {}
    for backend in ["reference", selected]:
        t = perf_counter()
        results[backend] = engine(args, list(BANDS), backend)(binary, phases, observer=observer,
                                                            source_revision=revision)
        costs[backend] = perf_counter()-t
    gates = parity(results["reference"], results[selected])
    controls = {d: field_controls(r, 1e-4, ln_f=-2.) for d, r in results.items()}
    numerical_change = None
    if args.comparison:
        before = json.loads(args.comparison.read_text())["result"]
        assert before["case"] == args.case
        numerical_change = {d: {v.key: finite(reload_result(before["models"][d]["curves"][v.key]),
                                             r.curves[v.key], 1e-4, ln_f=-2.) for v in specs(args)
                               if v.key in before["models"][d]["curves"]}
                            for d, r in results.items()}
        assert all(numerical_change[d] for d in results)
    independent_area = {name: roche_measures(binary, name, 32) for name in ["primary", "secondary"]}
    for r in results.values():
        for stars in r.diagnostics["field_geometry"].values():
            for name, cell in stars.items():
                assert abs(cell["intrinsic_luminosity_ratio"]-1) < 3e-13
                assert abs(cell["curved_area_SMA2"]/independent_area[name]["area"]-1) < 1e-9
    legacy_change = None
    if args.legacy_comparison:
        before = json.loads(args.legacy_comparison.read_text())["result"]
        assert before["case"] == args.case
        # Same actual clipped packet hash, not merely matched integer depth.
        legacy_change = {}
        for d, r in results.items():
            assert r.quadrature_sha256 == before["models"][d]["quadrature_sha256"]
            legacy_change[d] = finite(reload_result(before["models"][d]["curves"]["field2_correction2"]),
                                      r.curves["curvature_field2"], 1e-4, ln_f=-2.)
    return {"case": args.case, "rule": asdict(BoundaryRule(args.order, args.depth)),
            "integration_level": args.integration_level, "variants": [asdict(v) for v in specs(args)],
            "models": {d: asdict(r) for d, r in results.items()}, "matched_complete_batch_curve_seconds": costs,
            "parity": gates, "field_controls": controls, "boundary_or_partition_change": numerical_change,
            "independent_angular32_area": independent_area, "legacy_field_method_change": legacy_change,
            "limit": "Curvature-consistent finite fields on identical common clipped quadrature. "
            "CPU reference uses original gradient/thermal/reflection/radiance/elementflux formulas on new geometry. "
            "New field geometry/integration differ from original faceted whole curves. No continuum/fit qualification."}


def observation_profile(args: Any, revision: str) -> dict[str, Any]:  # noqa: C901 - bounded coarse/fine protocol
    fit = prepare_fit(make_problem(jd=args.proposal == "jd", free_nuisance=True),
                      phase_mode="observations", mesh_mode="paired")
    vector = normalized_vector(fit)
    if args.proposal == "changed":
        for key, value in {"primary@t_eff": .45, "system@inclination": .5,
                           "system@additional_light": .45, "nuisance@ln_f": .6}.items():
            vector[list(fit.fitable).index(key)] = value
    selected = "gpu" if args.device == "gpu" else "cpu"
    results, costs, predictions, densities = {}, {}, {}, {}
    for backend in ["reference", selected]:
        t = perf_counter()
        props = physical(fit, vector)
        instance = engine(args, list(fit.y_data), backend)
        saved = []
        def synthesis(phases: Any, discretization: float, observer: Any, *, _engine: Any = instance,
                      _saved: list = saved, **kwargs: Any) -> dict[str, Any]:
            binary = lc.prepare_binary(discretization=discretization, **kwargs)
            result = _engine(binary, phases, observer=observer, source_revision=revision)
            _saved.append(result)
            return result.curves[specs(args)[-1].key].fluxes

        prior = fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas)
        prediction = fit.predict_light_curves(props, synthesis=synthesis)
        density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, prediction, props["nuisance@ln_f"])+prior
        costs[backend] = perf_counter()-t
        results[backend], predictions[backend], densities[backend] = saved[0], prediction, density
    gates = parity(results["reference"], results[selected])
    # Independent original mapping controls for every variant. Reusing this
    # candidate's computed curve is only a mapping oracle, never timed as a
    # fresh synthesis or used to cache a later physical proposal.
    for r in results["reference"].curves.values():
        def cached(phases: Any, discretization: float, observer: Any, *, _r: Any = r,  # noqa: ARG001
                   **kwargs: Any) -> Any:  # noqa: ARG001
            return _r.fluxes

        original = fit.predict_light_curves(props, synthesis=cached)
        actual = mapped(r, fit)
        for band in original:
            compare(actual[band], original[band])
    controls = {d: field_controls(r, fit.y_err, reducers=fit.x_data_reducer, mean_normalize=True,
                                 observed=fit.y_data, ln_f=props["nuisance@ln_f"]) for d, r in results.items()}
    boundary = None
    boundary_source = "not_evaluated_in_this_coarse_control"
    if args.boundary_reference:
        previous = json.loads(args.boundary_reference.read_text())
        assert previous["part"] == "observations"
        assert previous["all_implementation_parity_gates_passed"]
        assert previous["source_revision"] == revision
        assert previous["result"]["proposal"] == args.proposal
        coarse_curve = reload_result(previous["result"]["models"]["reference"]["curves"][specs(args)[-1].key])
        boundary_source = "separately_bounded_complete_comparison_profile"
    elif not args.no_boundary_preflight:
        phases, binary = prepare_binary(fit, props)
        coarse = engine(args, list(fit.y_data), "reference", coarse=True)(
            binary, phases, observer=fit.observer, source_revision=revision,
        )
        coarse_curve = coarse.curves[specs(args)[-1].key]
        boundary_source = "extra_original_CPU_control_in_same_study_wall"
    if boundary_source != "not_evaluated_in_this_coarse_control":
        boundary = finite(coarse_curve, results["reference"].curves[specs(args)[-1].key], fit.y_err,
                          reducers=fit.x_data_reducer, mean_normalize=True, observed=fit.y_data,
                          ln_f=props["nuisance@ln_f"])
    assert np.isneginf(fit.ln_prior(np.full(len(vector), -.1), fit.norm_init_vector, fit.sigmas))
    return {"proposal": args.proposal, "physical": props, "rule": asdict(instance.rule),
            "integration_level": args.integration_level, "matched_complete_comparison_callback_seconds": costs,
            "prediction_parity": {b: compare(predictions[selected][b], predictions["reference"][b])
                                  for b in fit.y_data},
            "posterior_parity": compare(densities[selected], densities["reference"], atol=1e-6, rtol=0),
            "field_controls": controls, "boundary_preflight": boundary, "boundary_source": boundary_source,
            "models": {d: asdict(r) for d, r in results.items()}, "all_variant_curve_parity": gates,
            "quadrature_sha256": {d: r.quadrature_sha256 for d, r in results.items()},
            "out_of_prior": "Original -inf before synthesis",
            "contract": "Matched complete comparison callback: physical serialization/constraints, constructor, "
            "prior, original prediction layer and density. Both compute every fresh comparison curve on shared "
            "quadrature and return the fine curve through original mapping. Coarse preflight may use a separately "
            "bounded profile; its cost must also be counted. Mapping "
            "oracles add to mainwall. No cross-candidate prediction cache or ordinary single-model MCMC speedup claim."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--part", choices=["curves", "observations"], required=True)
    p.add_argument("--case", choices=["partial_bb", "total_bb", "partial_ck04", "small_occultor"], default="partial_bb")
    p.add_argument("--integration-level", type=int, default=2)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2])
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--proposal", choices=["base", "changed", "jd"], default="base")
    p.add_argument("--comparison", type=Path)
    p.add_argument("--legacy-comparison", type=Path)
    p.add_argument("--boundary-reference", type=Path)
    p.add_argument("--no-boundary-preflight", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = perf_counter()
    revision = asset_revision()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = curve_profile(args, revision) if args.part == "curves" else observation_profile(args, revision)
    report = {"part": args.part, "device": args.device, "source_revision": revision,
              "all_implementation_parity_gates_passed": True, "result": result, "wall_seconds": perf_counter()-started,
              "contract": "Fresh shared-quadrature comparison batches, all CPU bridges/transfers/host outputs. "
              "Compilation incurred included; caches can warm. Mainwall starts after imports; commandwall includes "
              "startup. No continuum field/boundary certificate, optimizedfit or chain/ESS claim."}
    def convert(value: Any) -> Any:
        return value.tolist() if isinstance(value, np.ndarray) else value.item()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=convert)+"\n")
    keys = ["part", "device", "wall_seconds", "all_implementation_parity_gates_passed"]
    print(json.dumps({k: report[k] for k in keys}))


if __name__ == "__main__":
    main()
