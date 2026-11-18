"""Replay complete physical proposals with ELISa's original facet observer and fitter."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from elisa.analytics.params import parameters
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit
from scripts.validate_gpu_surface_error import asset_revision


def compare(profile, facet_degrees):
    if not profile["irradiation"] or profile["source_revision"] != asset_revision():
        raise ValueError("The original comparison requires the same reflected fixture and table files")
    problem = make_problem(jd=profile["jd"], free_nuisance=True, shape_parameters=True)
    problem.setup["discretization"] = facet_degrees
    fit = prepare_fit(problem)
    for key, actual in (("observations", fit.y_data), ("uncertainties", fit.y_err), ("reducers", fit.x_data_reducer)):
        if set(profile[key]) != set(actual):
            raise ValueError("Mismatched observed passbands")
        for band, value in actual.items():
            np.testing.assert_array_equal(value, profile[key][band])
    rows = []
    for sample in profile["samples"]:
        for row in sample["rows"]:
            props = row["physical"]
            vector = np.array(
                parameters.vector_normalizer([props[k] for k in fit.fitable], fit.fitable, fit.normalization)
            )
            start = perf_counter()
            density = fit.ln_probability(vector)
            elapsed = perf_counter() - start
            prediction = fit.predict_light_curves(props)
            rebuilt = fit.ln_prior(vector, fit.norm_init_vector, fit.sigmas) + fit.likelihood_fn(
                prediction, props["nuisance@ln_f"]
            )
            np.testing.assert_allclose(rebuilt, density, atol=1e-6, rtol=0)
            rows.append(
                dict(
                    physical=props,
                    original_log_posterior=float(density),
                    spatial_log_posterior=row["posterior"],
                    original_facet_one_density_seconds=elapsed,
                    bands={
                        b: dict(
                            original_prediction=v.tolist(),
                            maximum_absolute_difference=float(np.max(abs(v - row["prediction"][b]))),
                        )
                        for b, v in prediction.items()
                    },
                )
            )
    return dict(
        facet_discretization_deg=fit.discretization,
        reflected_iterations=2,
        rows=rows,
        contract="Original facet density on identical inputs. The spatial calculation has its own curved finite surface and returns two field resolutions plus their accuracy comparison.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--facet-degrees", type=int, choices=(6, 8, 12), default=12)
    args = parser.parse_args()
    source = args.profile.read_bytes()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = compare(json.loads(source), args.facet_degrees)
    result["profile_sha256"] = hashlib.sha256(source).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
