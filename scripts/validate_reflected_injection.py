"""Check the curved reflected likelihood at the known physical generating parameters."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import const, settings, units
from elisa.analytics.models import cost_fns
from elisa.analytics.params import parameters
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule, InferenceBudget
from elisa.experimental.gpu_nodal_source import EmittingAreaLightCurve
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts.validate_common_donor_rays import CommonRays
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, physical, prepare_fit
from scripts.validate_gpu_selective import prepare_binary
from scripts.validate_gpu_surface_error import asset_revision


def run(args):
    if args.backend == "gpu" and not args.verified_bounded_gpu_window:
        raise ValueError("GPU synthesis requires a verified bounded unit")
    start = perf_counter()
    problem = make_problem(free_nuisance=True, shape_parameters=True)
    fit = prepare_fit(problem, phase_mode="observations", mesh_mode="paired")
    target = {**problem.truth, "nuisance@ln_f": -2.0}
    vector = np.array(parameters.vector_normalizer([target[k] for k in fit.fitable], fit.fitable, fit.normalization))
    props = physical(fit, vector)
    resolved = parameters.extend_json_with_atm_params(
        dict(props), atmosphere_model=fit.atmosphere_model, limb_darkening_coefficients=fit.limb_darkening_coefficients
    )
    for key, value in target.items():
        if isinstance(value, (str, dict)):
            if resolved[key] != value:
                raise ValueError("Generating atmosphere or limb-darkening fixture changed")
        else:
            np.testing.assert_allclose(resolved[key], value, rtol=0, atol=1e-12)
    phases, binary = prepare_binary(fit, props)
    levels = (2, 3)
    model = EmittingAreaLightCurve(
        list(fit.y_data),
        CurvedEdgeRule(3, args.depth, slice_order=8),
        device=args.backend,
        levels=levels,
        integration_level=3,
        receiver_points=3,
        donor_level=args.donor_level,
    )
    revision = asset_revision()
    protocol = dict(
        physical=props,
        passbands=list(fit.y_data),
        phases=phases.tolist(),
        backend=args.backend,
        source_revision=revision,
        integration_level=3,
        depth=args.depth,
    )
    if args.packets is not None:
        packets = CommonRays(model._packet, args.packets, protocol, args.mode)
        model._packet = packets
    result = model.batch([binary], [phases], observer=fit.observer, source_revision=revision).results[0]
    low, fine = [result.curves[v.key] for v in model.variants]

    expected_phases = phases

    def synthesis(phases, *args, **kwargs):
        np.testing.assert_array_equal(phases, expected_phases)
        return fine.fluxes

    prediction = fit.predict_light_curves(props, synthesis=synthesis)
    density = cost_fns.likelihood_fn(fit.y_data, fit.y_err, prediction, props["nuisance@ln_f"]) + fit.ln_prior(
        vector, fit.norm_init_vector, fit.sigmas
    )
    try:
        control = InferenceBudget().compare(
            low, fine, fit.y_err, observed=fit.y_data, reducers=fit.x_data_reducer, mean_normalize=True
        )
        aborted = False
    except ResolutionNotQualifiedError as error:
        control = error.report
        aborted = True
    row = dict(
        physical=props,
        prediction=prediction,
        posterior=float(density),
        accuracy_control=control,
        accuracy_abort=aborted,
        phase_records=result.diagnostics["phase_records"],
        quadrature_sha256=result.quadrature_sha256,
        curves={
            k: {a: getattr(c, a) for a in ["fluxes", "raw_fluxes", "component_fluxes"]}
            for k, c in result.curves.items()
        },
        finite_metadata={
            k: {a: getattr(c, a) for a in ["phases", "signature", "rule", "field_levels", "additional_light"]}
            for k, c in result.curves.items()
        },
    )
    report = dict(
        engine="emitting-area",
        backend=args.backend,
        levels=levels,
        integration_level=3,
        order=3,
        slice_order=8,
        depth=args.depth,
        donor_level=args.donor_level,
        receiver_points=3,
        irradiation=True,
        observations_irradiated=True,
        source_revision=revision,
        physical_constants={"G_SI": const.G, "solar_mass_kg": float(units.solMass.to(units.kg))},
        batch=1,
        jd=False,
        iteration_start="generating_parameters",
        reducers=fit.x_data_reducer,
        observations=fit.y_data,
        uncertainties=fit.y_err,
        samples=[{"rows": [row]}],
        total_seconds=perf_counter() - start,
    )
    if args.packets is not None:
        report["archived_geometry"] = packets.finish()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=lambda a: a.tolist()) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "gpu", "cpu"), required=True)
    parser.add_argument("--depth", type=int, choices=(3, 4), required=True)
    parser.add_argument("--donor-level", type=int, choices=(1, 2), required=True)
    parser.add_argument("--verified-bounded-gpu-window", action="store_true")
    parser.add_argument("--packets", type=Path)
    parser.add_argument("--mode", choices=("record", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.mode is None) != (args.packets is None):
        parser.error("Select both a packet directory and its record or replay mode")
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        run(args)
