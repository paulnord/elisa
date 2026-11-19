"""Audit consecutive physical sampler generations and measure movement in radii and inclination."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from elisa import settings
from scripts import sample_reflected_binary as sampler
from scripts.compare_reemission_workers import compare
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, physical, prepare_fit
from scripts.validate_gpu_selective import prepare_binary


def verify_continuation(previous, current):
    if previous.keys() != current.keys():
        raise ValueError("Cohort membership changed between physical generations")
    for seed, state in current.items():
        if sampler.digest(state) != sampler.digest(previous[seed]):
            raise ValueError("Chain history, selected state or RNG changed between generations")


def states(root, file):
    values = [sampler.read_record(p) for p in sorted(root.glob("walker-*/" + file))]
    result = {s["seed"]: s for s in values}
    if not values or len(result) != len(values):
        raise ValueError("Missing or repeated independent stream")
    return result


def coordinates(fit, vector):
    props = physical(fit, np.asarray(vector))
    _, model = prepare_binary(fit, props)
    return dict(
        primary_radius_over_separation=float(model.primary.equivalent_radius),
        secondary_radius_over_separation=float(model.secondary.equivalent_radius),
        inclination_deg=float(props["system@inclination"]),
    )


class OriginalFacet:
    """Replay each actual starting/proposed pair with ELISa's original density at one mesh size."""

    def __init__(self, degrees, reference):
        problem = make_problem(free_nuisance=True, shape_parameters=True)
        problem.setup["discretization"] = degrees
        self.fit = prepare_fit(problem)
        self.cache = {}
        for key in ("y_data", "y_err", "x_data_reducer"):
            actual, other = getattr(self.fit, key), getattr(reference, key)
            if set(actual) != set(other):
                raise ValueError("Facet fitter passbands changed")
            for band in actual:
                np.testing.assert_array_equal(actual[band], other[band])
        if list(self.fit.fitable) != list(reference.fitable) or self.fit.normalization != reference.normalization:
            raise ValueError("Facet fitter physical coordinates changed")

    def density(self, vector):
        key = tuple(vector)
        if key not in self.cache:
            v = np.asarray(vector)
            density = self.fit.ln_probability(v)
            props = physical(self.fit, v)
            prediction = self.fit.predict_light_curves(props)
            independent = self.fit.ln_prior(v, self.fit.norm_init_vector, self.fit.sigmas) + self.fit.likelihood_fn(
                prediction, props["nuisance@ln_f"]
            )
            np.testing.assert_allclose(density, independent, rtol=0, atol=1e-6)
            self.cache[key] = float(density)
        return self.cache[key]

    def decision(self, start, proposed, log_u):
        delta = self.density(proposed) - self.density(start)
        return dict(fitted_log_posterior_change=delta, conditional_accept=log_u < min(0.0, delta))


def assess(pairs, facet_degrees=()):
    fit, truth = fixture()
    facets = {d: OriginalFacet(d, fit) for d in facet_degrees}
    prior = {}
    traces, rounds, jumps = {}, [], []
    coord_cache = {}

    def point(state):
        key = tuple(state["position"])
        if key not in coord_cache:
            coord_cache[key] = coordinates(fit, key)
        return dict(step=state["step"], log_posterior=state["density"], coordinates=coord_cache[key])

    for reference, gpu in pairs:
        before = {k: states(p, "start.json") for k, p in (("reference", reference), ("gpu", gpu))}
        if prior:
            for k, starts in before.items():
                verify_continuation(prior[k], starts)
        audit = compare(reference, gpu)
        after = {k: states(p, "state.json") for k, p in (("reference", reference), ("gpu", gpu))}
        for row in audit["rows"]:
            seed = row["seed"]
            start, end = before["reference"][seed], after["reference"][seed]
            if row["status"] == "accuracy_stop" and len(rounds) + 1 < len(pairs):
                raise ValueError("Cannot continue an unresolved physical draw")
            for name in ("position", "rng", "scales", "step", "seed", "target"):
                if sampler.digest(end[name]) != sampler.digest(after["gpu"][seed][name]):
                    raise ValueError("CPU/GPU selected chain states differ")
            np.testing.assert_allclose(end["density"], after["gpu"][seed]["density"], rtol=0, atol=1e-6)
            track = traces.setdefault(seed, [point(start)])
            if row["status"] != "accuracy_stop":
                a, b = point(start)["coordinates"], point(end)["coordinates"]
                jumps.append({k: b[k] - a[k] for k in a})
                track.append(dict(**point(end), status=row["status"]))
            if "draw" in row:
                draw = row["draw"]
                row["original_facets"] = {
                    str(d): native.decision(start["position"], draw["proposed"], draw["log_uniform"])
                    for d, native in facets.items()
                }
        prior = after
        rounds.append(audit)
    inclination = list(fit.fitable).index("system@inclination")
    lower, upper = fit.normalization["system@inclination"]
    final_states = prior["reference"]
    accepted = Counter(row["status"] for group in rounds for row in group["rows"])
    return dict(
        rounds=rounds,
        selected=traces,
        status_counts=dict(accepted),
        injected_coordinates=coordinates(fit, truth),
        inclination_proposal_sigma_deg={
            seed: (upper - lower) * st["scales"][inclination] for seed, st in final_states.items()
        },
        original_facet_densities={
            str(d): dict(
                injected=native.density(truth),
                final_selected={seed: native.density(st["position"]) for seed, st in final_states.items()},
            )
            for d, native in facets.items()
        },
        rms_jump={k: float(np.sqrt(np.mean([j[k] ** 2 for j in jumps]))) for k in jumps[0]} if jumps else {},
        stationary_streams=sum(all(v["coordinates"] == tr[0]["coordinates"] for v in tr) for tr in traces.values()),
        finite_check="Every in-prior draw evaluated two fields, emitter groupings, eclipse depths and interior rules",
        scope="Observed displacement per transition across the supplied generations; these short traces do not estimate posterior uncertainty or effective samples.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, type=Path, action="append", required=True, metavar=("CPU", "GPU"))
    p.add_argument("--original-facets", nargs="*", type=int, choices=(6, 12), default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        sampler.write_record(args.output, assess(args.pair, args.original_facets))
