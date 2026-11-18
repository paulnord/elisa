"""Compare original ELISa finite meshes, proposal decisions and conditional radius fits."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import resource
from copy import deepcopy
from pathlib import Path
from time import perf_counter

# The m032 benchmark sets numerical thread limits before importing NumPy.
from scripts.benchmark_mcmc_scaling import PHYSICS, read_problem, sha256
import numpy as np
from emcee.moves import StretchMove
from scipy.optimize import least_squares

from elisa import settings
from elisa.analytics.models.lc import prepare_binary
from elisa.analytics.params import parameters
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import prepare_fit


SHAPE = ("system@inclination", "primary@surface_potential", "secondary@surface_potential")
STEPS = np.array([0.002, 0.001, 0.002])
BOUNDS = ([83.0, 7.8, 10.0], [89.0, 8.2, 14.0])


def read(path):
    return json.loads(path.read_bytes())


def store(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def stack(bands):
    return np.concatenate(list(bands.values()))


def shape_numbers(binary):
    return [[float(getattr(star, name)) for star in (binary.primary, binary.secondary)]
            for name in ("equivalent_radius", "side_radius")]


def fixed_radii(physical):
    # The volume calculation itself uses a sampled contour. Use the same
    # contour spacing for both stars and all fitted light-curve resolutions.
    cfg = dict(physical)
    result = []
    for degrees in (1.0, 0.5):
        cfg["secondary@discretization_factor"] = degrees
        result.append(shape_numbers(prepare_binary(discretization=degrees, **cfg)))
    return np.asarray(result)


def first_batch(fixture):
    initial = np.asarray(fixture["initial"])
    random = np.random.RandomState(fixture["move_seed"])
    random.choice([0], p=[1.0])
    sides = np.arange(len(initial)) % 2
    random.shuffle(sides)
    walkers = np.flatnonzero(sides == 0)
    moves, factors = StretchMove().get_proposal(initial[walkers], [initial[sides != 0]], random)
    log_uniforms = np.log(random.rand(len(walkers)))
    return walkers, moves, factors, log_uniforms


def study_positions(fixture):
    walkers, moves, factors, log_uniforms = first_batch(fixture)
    q = [("centre", fixture["centre"])]
    q += [(f"initial-{i:02d}", x) for i, x in enumerate(fixture["initial"])]
    q += [(f"proposal-{w:02d}", x.tolist()) for w, x in zip(walkers, moves, strict=True)]
    return q, walkers, factors, log_uniforms


class Native:
    def __init__(self, fixture, degrees):
        problem = read_problem(fixture)
        problem.setup["discretization"] = degrees
        self.fit = prepare_fit(problem)
        self.centre = np.asarray(fixture["centre"])
        self.true_shape = np.array([fixture["truth"][k] for k in SHAPE])
        self.ln_f = fixture["truth"]["nuisance@ln_f"]

    def props(self, position):
        renormalized, props = self.fit._prepare_likelihood_parameters(np.asarray(position))
        noise = parameters.prepare_nuisance_properties_set(renormalized, self.fit.fitable, self.fit.fixed)
        return props, noise["nuisance@ln_f"]

    def shape_props(self, shape):
        props, _ = self.props(self.centre)
        props.update(zip(SHAPE, shape, strict=True))
        return props

    def measure(self, position):
        position = np.asarray(position)
        props, ln_f = self.props(position)
        started = perf_counter()
        bands = self.fit.predict_light_curves(props)
        elapsed = perf_counter() - started
        log_likelihood = self.fit.likelihood_fn(bands, ln_f)
        prior = self.fit.ln_prior(position, self.fit.norm_init_vector, self.fit.sigmas)
        binary = self.fit.observer._system
        return dict(flux=stack(bands).tolist(), posterior=float(prior + log_likelihood),
                    log_likelihood=float(log_likelihood), seconds=elapsed,
                    primary_secondary_degrees=[float(np.degrees(star.discretization_factor))
                                               for star in (binary.primary, binary.secondary)])


def point_measurements(fixture, degrees, begin, end):
    model = Native(fixture, degrees)
    points, *_ = study_positions(fixture)
    if not 0 <= begin < end <= len(points):
        raise ValueError("Invalid range of fixture coordinates")
    rows = []
    for label, position in points[begin:end]:
        rows.append(dict(label=label, position=position, **model.measure(position)))
    return dict(rows=rows, nominal_sigma=stack(model.fit.y_err).tolist(),
                observed=stack(model.fit.y_data).tolist(), passbands=list(model.fit.y_data))


def noise_draw(fixture):
    rng = np.random.default_rng(fixture["data_seed"])
    return {name: rng.normal(0, np.asarray(band["error"])) for name, band in fixture["data"].items()}


def injection(fixture, centre_result):
    result = deepcopy(fixture)
    noise = noise_draw(fixture)
    centre = centre_result["rows"][0]
    assert centre["label"] == "centre" and centre["position"] == fixture["centre"]
    flux = np.asarray(centre["flux"])
    offset = 0
    for name, band in result["data"].items():
        size = len(band["y"])
        band["y"] = (flux[offset:offset + size] + noise[name]).tolist()
        offset += size
    assert offset == len(flux)
    return result


def central_jacobian(fn, x, steps):
    columns = []
    for i, step in enumerate(steps):
        perturb = np.zeros(len(x))
        perturb[i] = step
        columns.append((fn(x + perturb) - fn(x - perturb)) / (2 * step))
    return np.column_stack(columns)


def recovery(fixture, degrees, step_scale):
    model = Native(fixture, degrees)
    truth, fit = model.true_shape, model.fit
    data = stack(fit.y_data)
    sigma = stack(fit.y_err) * np.sqrt(1 + np.exp(2 * model.ln_f))
    cache = {}

    def residual(x):
        key = tuple(x)
        if key not in cache:
            flux = stack(fit.predict_light_curves(model.shape_props(x)))
            cache[key] = (flux - data) / sigma
        return cache[key]

    def jac(x):
        return central_jacobian(residual, x, STEPS * step_scale)

    origin = residual(truth)
    opt = least_squares(residual, truth, jac=jac, bounds=BOUNDS, x_scale="jac",
                        max_nfev=45, ftol=1e-8, xtol=1e-9, gtol=1e-7)
    jacobian = jac(opt.x)
    _, s, vt = np.linalg.svd(jacobian, full_matrices=False)
    if s[-1] <= s[0] * np.finfo(float).eps * max(jacobian.shape):
        raise RuntimeError("Rank-deficient conditional fit")
    covariance = (vt.T / s**2) @ vt

    def quantities(x):
        eq, side = fixed_radii(model.shape_props(x))[-1]
        return np.array([x[0], *eq, *side])

    g = central_jacobian(quantities, opt.x, STEPS)
    q_covariance = g @ covariance @ g.T
    contours = fixed_radii(model.shape_props(opt.x))
    def original_log_likelihood(standardized):
        prediction = standardized*sigma + data
        edges = np.cumsum([len(x) for x in fit.y_data.values()])[:-1]
        bands = dict(zip(fit.y_data, np.split(prediction, edges), strict=True))
        return float(fit.likelihood_fn(bands, model.ln_f))

    likelihood = [original_log_likelihood(row) for row in (origin, opt.fun)]
    np.testing.assert_allclose(2*(likelihood[1]-likelihood[0]), origin@origin-opt.fun@opt.fun, atol=1e-6, rtol=1e-10)
    return dict(parameters=list(SHAPE), parameters_fit=opt.x.tolist(), parameters_truth=truth.tolist(),
                parameter_sigma=np.sqrt(np.diag(covariance)).tolist(), covariance=covariance.tolist(),
                quantities=["inclination_deg", "r1_volume_over_a", "r2_volume_over_a",
                            "r1_side_over_a", "r2_side_over_a"],
                fit=quantities(opt.x).tolist(), truth=quantities(truth).tolist(),
                conditional_sigma=np.sqrt(np.diag(q_covariance)).tolist(),
                quantity_covariance=q_covariance.tolist(), equivalent_radius_1deg_minus_0p5deg=(contours[0,0]-contours[1,0]).tolist(),
                chi2_truth=float(origin @ origin), chi2_fit=float(opt.fun @ opt.fun), residual=opt.fun.tolist(),
                native_log_likelihood_truth_and_fit=likelihood,
                calls=len(cache), jacobian_singular_values=s.tolist(), solver_status=int(opt.status),
                solver_message=str(opt.message), optimality=float(opt.optimality), active_bounds=opt.active_mask.tolist(),
                iterations_evaluated=int(opt.nfev), success=bool(opt.success), finite_difference_scale=step_scale)


def profile_curve(fixture, degrees):
    model = Native(fixture, degrees)
    cold = model.measure(model.centre)
    repeats = [model.measure(model.centre) for _ in range(3)]
    prof = cProfile.Profile()
    with prof:
        result = model.measure(model.centre)
    stats = pstats.Stats(prof)
    records = [dict(file=key[0].split("/src/elisa/")[-1].split("/")[-1], function=key[2],
                    calls=v[1], own_s=v[2], cumulative_s=v[3]) for key, v in stats.stats.items()]
    stages = {}
    for fn in ("apply_reflection_effect", "compute_surface_coverage", "partial_visible_faces_surface_coverage",
               "build_mesh", "build_faces", "predict_light_curves"):
        stages[fn] = sum(row["cumulative_s"] for row in records if row["function"] == fn)
    for row in [*repeats, result]:
        np.testing.assert_array_equal(row["flux"], cold["flux"])
    return dict(cold_s=cold["seconds"], warm_s=[r["seconds"] for r in repeats], profiled_seconds=stats.total_tt,
                stages=stages, top_own_seconds=sorted(records, key=lambda r:r["own_s"], reverse=True)[:14],
                profiling_scope="One complete native prediction and its density in one process. Coverage includes partial coverage.",
                profiled_flux_equal=True)


def crossed_meshes(fixture, coarse, fine):
    """Vary each star's spacing separately on the same generating-point geometry."""
    if coarse <= fine:
        raise ValueError("Choose a coarser primary spacing to cross with the fine one")
    native = [Native(fixture, d) for d in (coarse, fine)]
    ends = [model.measure(model.centre) for model in native]
    secondary = [end["primary_secondary_degrees"][1] for end in ends]
    sigma = stack(native[0].fit.y_err)
    arrays = {}
    for i, model in enumerate(native):
        for j, spacing in enumerate(secondary):
            props, _ = model.props(model.centre)
            props["secondary@discretization_factor"] = spacing
            direct = stack(model.fit.predict_light_curves(props))
            if i == j:
                np.testing.assert_allclose(direct, ends[i]["flux"], rtol=3e-13, atol=2e-12)
            arrays[i,j] = direct
    primary = arrays[1,0] - arrays[0,0]
    companion = arrays[0,1] - arrays[0,0]
    interaction = arrays[1,1] - arrays[1,0] - arrays[0,1] + arrays[0,0]
    terms = dict(primary_only=primary, secondary_only=companion, interaction=interaction,
                 both=arrays[1,1]-arrays[0,0])
    return dict(coarse_primary_secondary_deg=ends[0]["primary_secondary_degrees"],
                fine_primary_secondary_deg=ends[1]["primary_secondary_degrees"],
                distances_sigma={k:float(np.linalg.norm(v/sigma)) for k,v in terms.items()},
                flux_differences={k:v.tolist() for k,v in terms.items()}, nominal_sigma=sigma.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["points", "profile", "recover", "crossed"])
    parser.add_argument("fixture", type=Path)
    parser.add_argument("degrees", type=float)
    parser.add_argument("output", type=Path)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--injection-centre", type=Path)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--coarse", type=float, default=3.0)
    args = parser.parse_args()
    fixture = read(args.fixture)
    injection_degrees = 12.0
    if args.injection_centre:
        centre = read(args.injection_centre)
        assert centre["fixture_sha256"] == sha256(args.fixture)
        fixture = injection(fixture, centre)
        injection_degrees = centre["primary_degrees"]
    start = perf_counter()
    with table_settings(), settings.override(**PHYSICS, NUMBER_OF_MCMC_PROCESSES=1,
                                             MAX_DISCRETIZATION_FACTOR=12, MESH_GENERATOR="trapezoidal"):
        if args.action == "points":
            result = point_measurements(fixture, args.degrees, args.begin, args.end)
        elif args.action == "recover":
            result = recovery(fixture, args.degrees, args.step_scale)
        elif args.action == "profile":
            result = profile_curve(fixture, args.degrees)
        else:
            result = crossed_meshes(fixture, args.coarse, args.degrees)
    result.update(action=args.action, primary_degrees=args.degrees, injection_degrees=injection_degrees,
                  fixture_sha256=sha256(args.fixture), driver_sha256=sha256(Path(__file__)), elapsed_s=perf_counter()-start,
                  linux_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  cpu_affinity=sorted(os.sched_getaffinity(0)))
    store(args.output, result)


if __name__ == "__main__":
    main()
