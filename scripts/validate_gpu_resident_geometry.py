"""Complete fresh physical-proposal batches, original mapping and likelihood.

GPU commands belong in registered hold-aware <=300s units. This driver never
manages production. All complete comparison curves and final densities return
on host. No prediction cache across batches, fit qualification or ESS claim.
"""
# ruff: noqa: INP001, S101, T201, PLR2004 - scientific standalone driver
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.analytics.models import cost_fns
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, InferenceBudget
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_resident_geometry import ResidentGeometryLightCurve
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, normalized_vector, physical, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision


def evaluate(instance: Any, fit: Any, count: int, iteration: int, revision: str) -> dict:
    start = perf_counter()
    properties, binaries, phases, priors = [], [], [], []
    for i in range(count):
        vector = normalized_vector(fit)
        for key, value in {"primary@t_eff": .38+.007*i+.003*iteration,
                           "system@inclination": .42+.009*i+.004*iteration,
                           "system@mass_ratio": .48+.003*i+.002*iteration,
                           "primary@surface_potential": .48+.004*i+.003*iteration,
                           "system@additional_light": .22+.002*i+.001*iteration,
                           "nuisance@ln_f": .4+.003*i+.002*iteration}.items():
            vector[list(fit.fitable).index(key)] = value
        props = physical(fit, vector)
        p, binary = prepare_binary(fit, props)
        properties.append(props)
        binaries.append(binary)
        phases.append(p)
        priors.append(fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas))
    prepared = perf_counter()
    batch = instance.batch(binaries, phases, observer=fit.observer, source_revision=revision)
    integrated = perf_counter()
    rows = []
    for i, r in enumerate(batch.results):
        fine = r.curves["curvature_field1"]
        def synthesis(phases: Any, discretization: float, observer: Any,  # noqa: ARG001
                      *, _expected: Any = phases[i], _fine: Any = fine,
                      **kwargs: Any) -> dict:  # noqa: ARG001 - original prediction callback contract
            # Same candidate's fresh complete curve, consumed inside this enclosing
            # timer. Only mapping is deferred until the whole geometry batch exists.
            np.testing.assert_array_equal(phases, _expected)
            return _fine.fluxes
        prediction = fit.predict_light_curves(properties[i], synthesis=synthesis)
        density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, prediction, properties[i]["nuisance@ln_f"])+priors[i]
        control = InferenceBudget().compare(r.curves["curvature_field0"], fine, fit.y_err,
            observed=fit.y_data, reducers=fit.x_data_reducer, mean_normalize=True, require_budget=False)
        aborted = False
        try:
            InferenceBudget().compare(r.curves["curvature_field0"], fine, fit.y_err,
                observed=fit.y_data, reducers=fit.x_data_reducer, mean_normalize=True)
        except ResolutionNotQualifiedError:
            aborted = True
        assert aborted != control["within_finite_budget"]
        rows.append({"physical": properties[i], "quadrature_sha256": r.quadrature_sha256,
                     "prediction": prediction, "posterior": density,
                     "curves": {k: {attr: getattr(c, attr) for attr in
                                ["fluxes", "raw_fluxes", "component_fluxes"]} for k, c in r.curves.items()},
                     "timings": r.timings, "accuracy_control": control,
                     "accuracy_abort": aborted, "phase_records": r.diagnostics["phase_records"]})
    done = perf_counter()
    return {"rows": rows, "enclosing_complete_batch_seconds": done-start,
            "physical_serialization_constructor_prior_seconds": prepared-start,
            "complete_synthesis_seconds": integrated-prepared,
            "original_mapping_density_accuracy_controls_seconds": done-integrated,
            "synthesis_timings": batch.timings, "geometry_groups": batch.geometry_groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["reference", "cpu", "gpu"], required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--jd", action="store_true")
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "gpu" and not args.verified_bounded_gpu_window:
        msg = "GPU execution requires a registered hold-aware bounded window"
        raise RuntimeError(msg)
    if not 1 <= args.batch <= 128 or args.repeats < 1:
        msg = "Use1-128proposals and at least one changed batch"
        raise ValueError(msg)
    started = perf_counter()
    revision = asset_revision()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        fit = prepare_fit(make_problem(jd=args.jd, free_nuisance=True, shape_parameters=True),
                          phase_mode="observations", mesh_mode="paired")
        input_setup = perf_counter()
        instance = ResidentGeometryLightCurve(list(fit.y_data), CurvedEdgeRule(3, 1), levels=(0, 1),
                                              integration_level=1, device=args.backend)
        constructed = perf_counter()
        samples = []
        for iteration in range(args.repeats+1):
            samples.append(evaluate(instance, fit, args.batch, iteration, revision))
            report = {"backend": args.backend, "batch": args.batch, "jd": args.jd, "samples": samples,
                      "source_revision": revision, "fixed_input_setup_seconds": input_setup-started,
                      "constructor_seconds": constructed-input_setup, "main_wall_seconds": perf_counter()-started,
                      "host_peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "affinity": sorted(os.sched_getaffinity(0)),
                      "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "contract": "First complete call includes encountered compilation; subsequent physically changed "
                      "batches retain all topology, reflection, radiance, observer geometry/clipping, transfers, "
                      "host comparison curves, original mean/l3/JD mapping, density and accuracy checks. "
                      "Serial original-CPU finite curvature reference versus separate JAX CPU/GPU. "
                      "Two finite field curves per candidate; not a production single-model likelihood cost. "
                      "No continuum, fit, observational-adequacy or chain/ESS qualification."}
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False,
                default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())+"\n")
            print(json.dumps({"iteration": iteration, "backend": args.backend, "batch": args.batch,
                              "complete_seconds": samples[-1]["enclosing_complete_batch_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
