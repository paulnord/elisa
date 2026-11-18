"""Recheck the mesh sensitivity of every attempted fixed-facet MCMC likelihood."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from scripts.benchmark_mcmc_scaling import FLUX_ATOL, FLUX_RTOL, LOGP_ATOL, PHYSICS, read_problem, sha256, write_json

import numpy as np

from elisa import settings
from elisa.analytics.params import parameters
from elisa.experimental.gpu_accuracy import resolution_error
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import prepare_fit

FITS = ()
DEGREES = (12.0, 8.0, 6.0)


def start_worker(fixture):
    global FITS
    base = read_problem(fixture)
    fits = []
    for degrees in DEGREES:
        problem = deepcopy(base)
        problem.setup["discretization"] = degrees
        fits.append(prepare_fit(problem))
    FITS = tuple(fits)


def check(row):
    index, position, saved_flux, saved_density = row
    target = FITS[0]
    renormalized, props = target._prepare_likelihood_parameters(position)
    ln_f = parameters.prepare_nuisance_properties_set(renormalized, target.fitable, target.fixed)["nuisance@ln_f"]
    models = [fit.predict_light_curves(props) for fit in FITS]
    flux = [np.concatenate([m[b] for b in target.y_data]) for m in models]
    density = np.array([fit.likelihood_fn(m, ln_f) for fit, m in zip(FITS, models, strict=True)])
    prior = target.ln_prior(position, target.norm_init_vector, target.sigmas)
    np.testing.assert_allclose(flux[0], saved_flux, rtol=FLUX_RTOL, atol=FLUX_ATOL)
    np.testing.assert_allclose(prior + density[0], saved_density, rtol=0, atol=LOGP_ATOL)
    error = np.concatenate([target.y_err[b] for b in target.y_data])
    observed = np.concatenate([target.y_data[b] for b in target.y_data])
    comparisons = [resolution_error(x, flux[-1], error, observed=observed, max_distance_sigma=0.1)
                   for x in flux[:-1]]
    return dict(index=index, distances=[d["distance_sigma"] for d in comparisons],
                passes=all(d["within_diagnostic_budget"] for d in comparisons),
                delta_log_likelihood_to_finest=(density[:-1] - density[-1]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("reference", type=Path, help="completed CPU reference run directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    run = json.loads((args.reference / "result.json").read_text())
    if run["backend"] != "cpu" or run["fixture_sha256"] != sha256(args.fixture):
        raise ValueError("Use the original CPU trace from exactly this fixture")
    with np.load(args.reference / "trace.npz") as trace:
        rows = [(i, pos, flux, density) for i, (pos, flux, density) in
                enumerate(zip(trace["positions"], trace["flux"], trace["posterior"], strict=True))
                if np.isfinite(density)]
        distinct = len(np.unique(trace["positions"], axis=0))
    with table_settings(), settings.override(**PHYSICS, NUMBER_OF_MCMC_PROCESSES=1):
        # Linux/Ada. Workers only execute serial original CPU curves; JAX is never initialized.
        from elisa.observer.mp_manager import get_mp_context
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_mp_context(),
                                 initializer=start_worker, initargs=(fixture,)) as pool:
            results = list(pool.map(check, rows))
    distances = np.array([v["distances"] for v in results])
    deltas = np.abs([v["delta_log_likelihood_to_finest"] for v in results])
    write_json(args.output, dict(fixture_sha256=run["fixture_sha256"], trace_sha256=run["trace_sha256"],
        mesh_degrees=DEGREES, joint_nominal_sigma_limit=0.1, attempted=run["posterior_calls"], distinct=distinct,
        evaluated=len(rows), qualified=sum(v["passes"] for v in results),
        distance_min=distances.min(axis=0).tolist(), distance_max=distances.max(axis=0).tolist(),
        absolute_delta_log_likelihood_range=[deltas.min(axis=0).tolist(), deltas.max(axis=0).tolist()],
        native_predictions_and_densities_reproduced=True, rows=results))


if __name__ == "__main__":
    main()
