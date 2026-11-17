"""Conservative reconstructed fields in complete fresh physical likelihood batches.

GPU commands belong in registered hold-aware <=300s units. This driver never
manages production. All complete comparison curves and final densities return
on host. No prediction cache across batches, fit qualification or ESS claim.
"""

# ruff: noqa: INP001, S101, T201, PLR2004, ARG001 - scientific standalone driver
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

from elisa import const, settings
from elisa import units as astro_units
from elisa.analytics.models import cost_fns
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, InferenceBudget
from elisa.experimental.gpu_fused_observer import FusedObserverLightCurve
from elisa.experimental.gpu_observer_edges import ObserverDeviceLightCurve
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.gpu_reconstructed_observer import ReconstructedObserverLightCurve
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, normalized_vector, physical, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision


def evaluate(instance: Any, fit: Any, count: int, iteration: int, revision: str) -> dict:
    start = perf_counter()
    properties, binaries, phases, priors = [], [], [], []
    for i in range(count):
        vector = normalized_vector(fit)
        for key, value in {
            "primary@t_eff": 0.38 + 0.007 * i + 0.003 * iteration,
            "system@inclination": 0.42 + 0.009 * i + 0.004 * iteration,
            "system@mass_ratio": 0.48 + 0.003 * i + 0.002 * iteration,
            "primary@surface_potential": 0.48 + 0.004 * i + 0.003 * iteration,
            "system@additional_light": 0.22 + 0.002 * i + 0.001 * iteration,
            "nuisance@ln_f": 0.4 + 0.003 * i + 0.002 * iteration,
        }.items():
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
        fine = r.curves[instance.variants[-1].key]

        def synthesis(
            phases: Any,
            discretization: float,
            observer: Any,
            *,
            _expected: Any = phases[i],
            _fine: Any = fine,
            **kwargs: Any,
        ) -> dict:
            # Same candidate's fresh complete curve, consumed inside this enclosing
            # timer. Only mapping is deferred until the whole geometry batch exists.
            np.testing.assert_array_equal(phases, _expected)
            return _fine.fluxes

        prediction = fit.predict_light_curves(properties[i], synthesis=synthesis)
        density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, prediction, properties[i]["nuisance@ln_f"]) + priors[i]
        control = InferenceBudget().compare(
            r.curves[instance.variants[0].key],
            fine,
            fit.y_err,
            observed=fit.y_data,
            reducers=fit.x_data_reducer,
            mean_normalize=True,
            require_budget=False,
        )
        aborted = False
        try:
            InferenceBudget().compare(
                r.curves[instance.variants[0].key],
                fine,
                fit.y_err,
                observed=fit.y_data,
                reducers=fit.x_data_reducer,
                mean_normalize=True,
            )
        except ResolutionNotQualifiedError:
            aborted = True
        assert aborted != control["within_finite_budget"]
        rows.append(
            {
                "physical": properties[i],
                "quadrature_sha256": r.quadrature_sha256,
                "prediction": prediction,
                "posterior": density,
                "curves": {
                    k: {attr: getattr(c, attr) for attr in ["fluxes", "raw_fluxes", "component_fluxes"]}
                    for k, c in r.curves.items()
                },
                "finite_metadata": {
                    k: {
                        attr: getattr(c, attr)
                        for attr in ["phases", "signature", "rule", "field_levels", "additional_light"]
                    }
                    for k, c in r.curves.items()
                },
                "field_controls": r.diagnostics["ancestry_controls"],
                "timings": r.timings,
                "accuracy_control": control,
                "accuracy_abort": aborted,
                "phase_records": r.diagnostics["phase_records"],
            },
        )
    done = perf_counter()
    return {
        "rows": rows,
        "enclosing_complete_batch_seconds": done - start,
        "physical_serialization_constructor_prior_seconds": prepared - start,
        "complete_synthesis_seconds": integrated - prepared,
        "original_mapping_density_accuracy_controls_seconds": done - integrated,
        "synthesis_timings": batch.timings,
        "geometry_groups": batch.geometry_groups,
        "observer_stats": instance.last_observer_stats,
        "field_stats": getattr(instance, "last_field_stats", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["reference", "cpu", "gpu"], required=True)
    parser.add_argument(
        "--engine", choices=["reconstructed", "fused", "observer", "area-thermal"], default="reconstructed",
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--slice-order", type=int, default=4)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--iteration-start", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--jd", action="store_true")
    parser.add_argument("--without-irradiation", action="store_true")
    parser.add_argument("--levels", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--integration-level", type=int, default=1)
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "gpu" and not args.verified_bounded_gpu_window:
        msg = "GPU execution requires a registered hold-aware bounded window"
        raise RuntimeError(msg)
    if not 1 <= args.batch <= 128 or args.repeats < 0:
        msg = "Use1-128proposals and nonnegative repeat count; zero records only a first complete call"
        raise ValueError(msg)
    started = perf_counter()
    revision = asset_revision()
    with (
        table_settings(),
        settings.override(REFLECTION_EFFECT=not args.without_irradiation, REFLECTION_EFFECT_ITERATIONS=2),
    ):
        # Generate the same original irradiated observational fixture for both physical models.
        with settings.override(REFLECTION_EFFECT=True):
            problem = make_problem(jd=args.jd, free_nuisance=True, shape_parameters=True)
        fit = prepare_fit(
            problem,
            phase_mode="observations",
            mesh_mode="paired",
        )
        input_setup = perf_counter()
        engine = {
            "reconstructed": ReconstructedObserverLightCurve,
            "fused": FusedObserverLightCurve,
            "area-thermal": FusedObserverLightCurve,
            "observer": ObserverDeviceLightCurve,
        }[args.engine]
        instance = engine(
            list(fit.y_data),
            CurvedEdgeRule(args.order, args.depth, slice_order=args.slice_order),
            levels=tuple(args.levels),
            integration_level=args.integration_level,
            device=args.backend,
            **({"area_thermal": True} if args.engine == "area-thermal" else {}),
        )
        constructed = perf_counter()
        samples = []
        for iteration in range(args.iteration_start, args.iteration_start + args.repeats + 1):
            samples.append(evaluate(instance, fit, args.batch, iteration, revision))
            report = {
                "backend": args.backend,
                "irradiation": not args.without_irradiation,
                "observations_irradiated": True,
                "batch": args.batch,
                "jd": args.jd,
                "levels": args.levels,
                "iteration_start": args.iteration_start,
                "integration_level": args.integration_level,
                "engine": args.engine,
                "depth": args.depth,
                "slice_order": args.slice_order,
                "order": args.order,
                "observations": fit.y_data,
                "uncertainties": fit.y_err,
                "reducers": fit.x_data_reducer,
                "samples": samples,
                "source_revision": revision,
                "physical_constants": {"G_SI": const.G, "solar_mass_kg": float(astro_units.solMass.to(astro_units.kg))},
                "fixed_input_setup_seconds": input_setup - started,
                "constructor_seconds": constructed - input_setup,
                "main_wall_seconds": perf_counter() - started,
                "host_peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "affinity": sorted(os.sched_getaffinity(0)),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "contract": "First complete call includes encountered compilation; subsequent physically changed "
                "batches retain all topology, reflection, radiance, observer geometry/clipping, transfers, "
                "host comparison curves, original mean/l3/JD mapping, density and accuracy checks. "
                "Serial original-CPU finite curvature reference versus separate JAX CPU and native nested GPU; "
                "engine recorded. "
                "Selected finite field representation is recorded; reconstructed uses conservative positive "
                "C0 intensity/LD moments. "
                "Two finite field curves per candidate; not a production single-model likelihood cost. "
                "No continuum, fit, observational-adequacy or chain/ESS qualification.",
            }
            args.output.write_text(
                json.dumps(
                    report,
                    indent=2,
                    allow_nan=False,
                    default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item(),
                )
                + "\n",
            )
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "backend": args.backend,
                        "batch": args.batch,
                        "complete_seconds": samples[-1]["enclosing_complete_batch_seconds"],
                    },
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
