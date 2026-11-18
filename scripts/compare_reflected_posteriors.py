"""Replay the original fitter's density at each corner of a reflected resolution check."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.analytics.params import parameters
from elisa.experimental.gpu_binary import HybridBinaryLightCurve
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts.compare_gpu_joint_budget import curve
from scripts.compare_reflection_resolution import compare
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, physical, prepare_fit
from scripts.validate_gpu_surface_error import asset_revision


def mapped_density(fit, props, prior, model):
    flux = HybridBinaryLightCurve._observed_fluxes(model.component_fluxes, model, normalize=True)
    for band, values in flux.items():
        np.testing.assert_allclose(values, model.fluxes[band], atol=2e-12, rtol=3e-13)

    def synthesis(phases, *args, **kwargs):
        np.testing.assert_array_equal(phases, model.phases)
        return flux

    prediction = fit.predict_light_curves(props, synthesis=synthesis)
    return prediction, float(prior + fit.likelihood_fn(prediction, props["nuisance@ln_f"]))


def replay(profiles, *, require_budget=True):
    """Return a usable finite density only after checking all eight corners of one proposal."""
    checked = compare(profiles)
    ref = profiles[0]
    if (
        len(checked) != 1
        or ref["engine"] != "emitting-area"
        or ref["source_revision"] != asset_revision()
        or ref["levels"] != [2, 3]
        or ref["integration_level"] != 3
        or ref["receiver_points"] != 3
        or ref["slice_order"] != 8
        or any(not p["observations_irradiated"] for p in profiles)
        or any(len(p["samples"]) != 1 or p["batch"] != 1 for p in profiles)
        or {(p["depth"], p["donor_level"]) for p in profiles} != {(3, 1), (3, 2), (4, 1), (4, 2)}
    ):
        raise ValueError("Expected a single matched m028 emitting-area proposal and its full refinement grid")
    control = checked[0]["control"]
    if require_budget and not control["within_finite_budget"]:
        raise ResolutionNotQualifiedError("Reflected proposal exceeds the combined finite likelihood budget", control)

    fit = prepare_fit(
        make_problem(jd=ref["jd"], free_nuisance=True, shape_parameters=True),
        phase_mode="observations",
        mesh_mode="paired",
    )
    for attr, key in (("y_data", "observations"), ("y_err", "uncertainties"), ("x_data_reducer", "reducers")):
        for band, v in getattr(fit, attr).items():
            np.testing.assert_array_equal(v, ref[key][band])
    props = checked[0]["physical"]
    vector = np.asarray(parameters.vector_normalizer([props[k] for k in fit.fitable], fit.fitable, fit.normalization))
    reconstructed = physical(fit, vector)
    if props.keys() != reconstructed.keys():
        raise ValueError("Physical keys differ from original fit setup")
    for k, v in reconstructed.items():
        np.testing.assert_allclose(v, props[k], rtol=1e-14, atol=1e-14)
    prior = float(fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas))
    if not np.isfinite(prior):
        raise ValueError("Physical replay requires a proposal within the original prior")
    densities = {}
    for p in profiles:
        row = p["samples"][0]["rows"][0]
        for key in row["curves"]:
            c = curve(row, key)
            np.testing.assert_equal(c.additional_light, props["system@additional_light"])
            prediction, value = mapped_density(fit, props, prior, c)
            if c.field_levels == max(ref["levels"]):
                for band, v in prediction.items():
                    np.testing.assert_allclose(v, row["prediction"][band], rtol=3e-13, atol=2e-12)
                np.testing.assert_allclose(value, row["posterior"], rtol=0, atol=1e-6)
            densities[f"field{c.field_levels}_donor{p['donor_level']}_depth{p['depth']}"] = value
    fine = densities["field3_donor2_depth4"]
    return dict(
        physical=props,
        fitable=list(fit.fitable),
        normalized=vector.tolist(),
        prior=prior,
        finite_pass=control["within_finite_budget"],
        nominal_bound=control["combined_log_likelihood_bound"],
        usable_log_posterior=fine if control["within_finite_budget"] else None,
        corner_log_posteriors=densities,
        maximum_measured_absolute_difference=max(abs(x - fine) for x in densities.values()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs=4, type=Path)
    parser.add_argument("--audit", action="store_true", help="Record corner diagnostics even if the accuracy gate stops")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [path.read_bytes() for path in args.profiles]
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = replay([json.loads(v) for v in inputs], require_budget=not args.audit)
    result["input_sha256"] = [hashlib.sha256(v).hexdigest() for v in inputs]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
