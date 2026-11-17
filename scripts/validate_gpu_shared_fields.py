"""Shared-quadrature complete field/correction and likelihood comparisons.

GPU execution requires independently restored bounded allocations; this driver
never changes production. Three fresh field models share each clipped packet.
"""

# ruff: noqa: INP001, S101, T201, PLR2004 - standalone scientific driver
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
from elisa.experimental.boundary_geometry import BoundaryRule
from elisa.experimental.gpu_shared_fields import FieldVariant, SharedFieldLightCurve, shared_area_moments
from scripts.validate_gpu_atmosphere import BANDS, table_settings
from scripts.validate_gpu_atmosphere import fixture as table_fixture
from scripts.validate_gpu_binary import contact_phases, fixture
from scripts.validate_gpu_boundary import finite, reload_result
from scripts.validate_gpu_fit import compare, make_problem, normalized_vector, physical, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision, mapped

VARIANTS = (FieldVariant(1, 1), FieldVariant(1, 2), FieldVariant(2, 2))
PAIRS = {"correction_only": (0, 1), "field_grid_fixed_correction": (1, 2), "total_field_change": (0, 2)}


def field_controls(result: Any, sigma: Any, **kwargs: Any) -> dict[str, Any]:
    curves = [result.curves[v.key] for v in VARIANTS]
    controls = {name: finite(curves[a], curves[b], sigma, **kwargs) for name, (a, b) in PAIRS.items()}
    reconstruction = {}
    for band in curves[0].fluxes:
        values = [c.fluxes[band] for c in curves]
        residual = (values[2]-values[0])-((values[1]-values[0])+(values[2]-values[1]))
        max_residual = float(np.max(np.abs(residual)))
        assert max_residual <= 32*np.finfo(float).eps
        reconstruction[band] = max_residual
    return {"pairs": controls, "normalized_telescoping_max_abs": reconstruction,
            "contract": "All changes use identical clipped coordinates, weights, normals and masks. "
            "Correction-only holds raw field topology/cached state; fixed-correction field refinement rebuilds "
            "original finite physics. Finite comparisons only, no continuum certificate."}


def parity(reference: Any, selected: Any) -> dict[str, Any]:
    assert reference.quadrature_sha256 == selected.quadrature_sha256
    assert reference.diagnostics["phase_records"] == selected.diagnostics["phase_records"]
    return {v.key: {key: {b: compare(getattr(selected.curves[v.key], key)[b],
                                   getattr(reference.curves[v.key], key)[b], atol=2e-12 if key == "fluxes" else 0)
                          for b in reference.curves[v.key].fluxes}
                   for key in ["fluxes", "raw_fluxes", "component_fluxes"]} for v in VARIANTS}


def engine(args: Any, bands: list[str], backend: str, *, coarse: bool = False) -> SharedFieldLightCurve:
    return SharedFieldLightCurve(bands, BoundaryRule(args.order, args.depth-int(coarse)),
                                (VARIANTS[-1],) if coarse else VARIANTS, integration_level=args.integration_level,
                                device=backend, radiance="cpu" if args.case == "partial_ck04" else "device")


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
                                             r.curves[v.key], 1e-4, ln_f=-2.) for v in VARIANTS}
                            for d, r in results.items()}
    moments = shared_area_moments(binary, args.integration_level, VARIANTS)
    for name in ["primary", "secondary"]:
        areas = [moments[v.key][name]["curved_area_SMA2"] for v in VARIANTS]
        assert np.max(areas)/np.min(areas)-1 <= 3e-15
        assert all(abs(moments[v.key][name]["original_intrinsic_closure_relative"]) < 3e-13 for v in VARIANTS)
    return {"case": args.case, "rule": asdict(BoundaryRule(args.order, args.depth)),
            "integration_level": args.integration_level, "variants": [asdict(v) for v in VARIANTS],
            "models": {d: asdict(r) for d, r in results.items()}, "matched_complete_batch_curve_seconds": costs,
            "parity": gates, "field_controls": controls, "boundary_or_partition_change": numerical_change,
            "cpu_intrinsic_shared_area_moments": moments,
            "limit": "Three fresh finite field models on identical common clipped quadrature. "
            "CPU reference uses original finite state/reflection/radiance and elementflux arithmetic. "
            "New integration differs from original faceted whole curves. No continuum/physical fit qualification."}


def observation_profile(args: Any, revision: str) -> dict[str, Any]:
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
            return result.curves[VARIANTS[-1].key].fluxes

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
    phases, binary = prepare_binary(fit, props)
    coarse = engine(args, list(fit.y_data), "reference", coarse=True)(
        binary, phases, observer=fit.observer, source_revision=revision,
    )
    boundary = finite(coarse.curves[VARIANTS[-1].key], results["reference"].curves[VARIANTS[-1].key], fit.y_err,
                      reducers=fit.x_data_reducer, mean_normalize=True, observed=fit.y_data,
                      ln_f=props["nuisance@ln_f"])
    assert np.isneginf(fit.ln_prior(np.full(len(vector), -.1), fit.norm_init_vector, fit.sigmas))
    return {"proposal": args.proposal, "physical": props, "rule": asdict(instance.rule),
            "integration_level": args.integration_level, "matched_complete_comparison_callback_seconds": costs,
            "prediction_parity": {b: compare(predictions[selected][b], predictions["reference"][b])
                                  for b in fit.y_data},
            "posterior_parity": compare(densities[selected], densities["reference"], atol=1e-6, rtol=0),
            "field_controls": controls, "boundary_preflight": boundary, "all_variant_curve_parity": gates,
            "quadrature_sha256": {d: r.quadrature_sha256 for d, r in results.items()},
            "out_of_prior": "Original -inf before synthesis",
            "contract": "Matched complete comparison callback: physical serialization/constraints, constructor, "
            "prior, original prediction layer and density. Both compute three fresh comparison curves on shared "
            "quadrature and return the fine curve through original mapping. Extra coarse preflight and mapping "
            "oracles add to mainwall. No cross-candidate prediction cache or ordinary single-model MCMC speedup claim."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--part", choices=["curves", "observations"], required=True)
    p.add_argument("--case", choices=["partial_bb", "total_bb", "partial_ck04", "small_occultor"], default="partial_bb")
    p.add_argument("--integration-level", type=int, default=2)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--proposal", choices=["base", "changed", "jd"], default="base")
    p.add_argument("--comparison", type=Path)
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
