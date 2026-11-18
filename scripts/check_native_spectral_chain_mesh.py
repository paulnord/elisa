"""Check all proposals of the measured native spectral sampler against the 2.5°, 2° and 1.5° meshes."""

import argparse
from pathlib import Path

from emcee.moves import StretchMove
import numpy as np

from elisa import settings
from elisa.experimental.gpu_accuracy import resolution_error
from scripts.benchmark_mcmc_scaling import PHYSICS, sha256
from scripts.study_native_mesh_refinement import Native, read, stack, store, table_settings


def replay(fixture, trace, posteriors):
    """Compare decisions on the same trial, preceding fine state, stretch factor and random draw."""
    random = np.random.RandomState(fixture["move_seed"])
    move = StretchMove()
    walkers = np.array(fixture["initial"])
    lookup = {tuple(x): i for i, x in enumerate(trace["positions"])}
    selected = np.array([lookup[tuple(x)] for x in walkers])
    seen = list(selected)
    acceptances = np.zeros(len(walkers), dtype=int)
    output = []
    for step, expected in enumerate(trace["chain"], 1):
        random.choice([0], p=[1.0])  # emcee selects its one configured move before splitting.
        sides = np.arange(len(walkers)) % 2
        random.shuffle(sides)
        for side in (0, 1):
            indices = np.flatnonzero(sides == side)
            coordinates, factors = move.get_proposal(walkers[indices], [walkers[sides != side]], random)
            candidates = np.array([lookup[tuple(x)] for x in coordinates])
            seen.extend(candidates)
            thresholds = np.log(random.rand(len(indices)))
            old = selected[indices]
            margins = {
                k: factors + density[candidates] - density[old] - thresholds for k, density in posteriors.items()
            }
            accepted = margins["1.5"] > 0
            for j, walker in enumerate(indices):
                output.append(
                    dict(
                        step=step,
                        walker=int(walker),
                        current=int(old[j]),
                        proposed=int(candidates[j]),
                        log_stretch_factor=float(factors[j]),
                        log_uniform=float(thresholds[j]),
                        margins={k: float(m[j]) for k, m in margins.items()},
                    )
                )
            acceptances[indices] += accepted
            selected[indices[accepted]] = candidates[accepted]
            walkers[indices[accepted]] = coordinates[accepted]
        np.testing.assert_array_equal(walkers, expected)
        np.testing.assert_allclose(posteriors["1.5"][selected], trace["chain_logp"][step - 1], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(acceptances, trace["accepted"])
    np.testing.assert_array_equal(np.sort(seen), np.arange(len(lookup)))
    return output


def evaluate(fixture, native, spectra):
    trace = {k: np.asarray(v) for k, v in spectra["reference_trace"].items()}
    if (
        fixture["burn_in"] != 0
        or trace["chain"].shape != (2, 32, 6)
        or trace["positions"].shape != (96, 6)
        or trace["flux"].shape != (96, 28)
        or any(not np.isfinite(v).all() for v in trace.values())
    ):
        raise ValueError("Expected the two-step all-finite native trace on the frozen three-band fixture")
    levels = native["finest_triple"]["primary_degrees"]
    if levels != [2.5, 2, 1.5] or spectra["primary_degrees"] != levels[-1]:
        raise ValueError("Unexpected native reference mesh")
    models = {f"{d:g}": Native(fixture, d) for d in levels[:-1]}
    fit = models["2"].fit
    bands = list(fit.y_data)
    cuts = np.cumsum([len(fit.y_data[k]) for k in bands])[:-1]
    sigma, observed = stack(fit.y_err), stack(fit.y_data)
    np.testing.assert_array_equal(sigma, native["nominal_sigma"])
    limit = native["limits"]["nominal_joint_sigma"]
    archived = {tuple(x["position"]): i for i, x in enumerate(native["sampled_points"])}
    densities = {"1.5": trace["posterior"]}
    densities.update({k: np.empty(len(trace["positions"])) for k in models})
    rows = []
    for i, (x, fine) in enumerate(zip(trace["positions"], trace["flux"], strict=True)):
        _, ln_f = models["2"].props(x)
        prior = fit.ln_prior(x, fit.norm_init_vector, fit.sigmas)
        band_fluxes = dict(zip(bands, np.split(fine, cuts), strict=True))
        np.testing.assert_allclose(fit.likelihood_fn(band_fluxes, ln_f) + prior, densities["1.5"][i], atol=1e-6, rtol=0)
        j = archived.get(tuple(x))
        if j is not None:
            np.testing.assert_allclose(fine, native["native_predictions"]["1.5"]["flux"][j], atol=2e-12, rtol=3e-13)
            np.testing.assert_allclose(
                densities["1.5"][i], native["native_predictions"]["1.5"]["log_posterior"][j], atol=1e-6, rtol=0
            )
        comparisons = {}
        for level, model in models.items():
            if j is None:
                result = model.measure(x)
                flux, density = result["flux"], result["posterior"]
            else:
                previous = native["native_predictions"][level]
                flux, density = previous["flux"][j], previous["log_posterior"][j]
            densities[level][i] = density
            nominal = resolution_error(flux, fine, sigma, observed=observed, max_distance_sigma=limit)
            fitted = resolution_error(flux, fine, sigma, observed=observed, ln_f=ln_f)
            change = density - densities["1.5"][i]
            np.testing.assert_allclose(change, fitted["delta_log_likelihood"], atol=1e-6, rtol=0)
            comparisons[level] = dict(
                distance_sigma=nominal["distance_sigma"],
                passed=nominal["within_diagnostic_budget"],
                nominal_delta_log_likelihood=nominal["delta_log_likelihood"],
                fitted_delta_log_posterior=float(change),
            )
        rows.append(
            dict(
                position=x.tolist(),
                archived=(j is not None),
                comparisons=comparisons,
                passes_both=all(c["passed"] for c in comparisons.values()),
            )
        )
    decisions = replay(fixture, trace, densities)
    previous = native["samplers"]
    initial_split = decisions[: len(previous["walkers"])]
    np.testing.assert_array_equal([row["walker"] for row in initial_split], previous["walkers"])
    np.testing.assert_array_equal([row["log_uniform"] for row in initial_split], previous["log_uniforms"])
    np.testing.assert_array_equal([row["log_stretch_factor"] for row in initial_split], previous["stretch_log_factors"])
    for level in densities:
        margins = [row["margins"][level] for row in initial_split]
        np.testing.assert_allclose(margins, previous["decisions"][level]["margin"], atol=2e-6, rtol=0)
        np.testing.assert_array_equal(np.greater(margins, 0), previous["decisions"][level]["accepted"])
    return dict(
        meshes=levels,
        nominal_joint_limit=limit,
        positions=rows,
        conditional_decisions=decisions,
        qualifying_positions=sum(r["passes_both"] for r in rows),
        reused_positions=sum(r["archived"] for r in rows),
        fine_acceptances=sum(d["margins"]["1.5"] > 0 for d in decisions),
        changed_decisions={
            k: sum((d["margins"][k] > 0) != (d["margins"]["1.5"] > 0) for d in decisions) for k in models
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("native_mesh_results", type=Path)
    parser.add_argument("spectral_results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    native, spectra = read(args.native_mesh_results), read(args.spectral_results)
    if (
        sha256(args.fixture) != native["fixture_sha256"]
        or native["fixture_sha256"] != spectra["fixture_sha256"]
        or sha256(args.native_mesh_results) != spectra["native_mesh_results_sha256"]
    ):
        raise ValueError("The fit fixture or native mesh reference has changed")
    with (
        table_settings(),
        settings.override(
            **PHYSICS, NUMBER_OF_MCMC_PROCESSES=1, MAX_DISCRETIZATION_FACTOR=12, MESH_GENERATOR="trapezoidal"
        ),
    ):
        result = evaluate(read(args.fixture), native, spectra)
    result.update(
        fixture_sha256=sha256(args.fixture),
        native_mesh_sha256=sha256(args.native_mesh_results),
        spectral_results_sha256=sha256(args.spectral_results),
    )
    store(args.output, result)


if __name__ == "__main__":
    main()
