# Copyright 2026 ELISa contributors
"""Seeded real-data MCMC benchmark for the tutorial 12 KIC 4851217 fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import emcee
import numpy as np
import scipy
from astropy import physical_constants

# Match the constants convention used throughout the ELISa regression tests.
physical_constants.set("astropyconst20")

from astropy import units as u  # noqa: E402

import elisa  # noqa: E402
from elisa import settings  # noqa: E402
from elisa.analytics import LCBinaryAnalyticsTask, LCData  # noqa: E402
from elisa.analytics.params import parameters  # noqa: E402
from elisa.analytics.params.parameters import BinaryInitialParameters  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "jupyter_tutorials/demo_data/lc_data/KIC04851217.lc.pulse_removed.dat"
)
PARAMETERS_PATH = REPOSITORY_ROOT / "jupyter_tutorials/demo_data/assets/lc_least_squares.json"
PERIOD_DAYS = 2.47028
PRIMARY_MINIMUM_BJD = 54953.900507000006

INVALID_LOG_PROBABILITY_FLOOR = -1e100
ACCEPTANCE_RANGE = (0.1, 0.7)
MIN_EFFECTIVE_SAMPLES = 1000.0
AUTOCORRELATION_MULTIPLE = 50.0
MIN_STEPS_FOR_AUTOCORRELATION = 10
EXPECTED_FREE_PARAMETERS = (
    "primary@surface_potential",
    "primary@t_eff",
    "secondary@surface_potential",
    "secondary@t_eff",
    "system@eccentricity",
    "system@argument_of_periastron",
    "system@inclination",
    "system@mass_ratio",
    "system@additional_light",
    "nuisance@ln_f",
)


@dataclass(frozen=True)
class MCMCProfile:
    """Execution settings for one reproducible benchmark profile."""

    name: str
    nwalkers: int
    nsteps: int
    burn_in: int
    discretization: float
    interpolation_points: int
    samples: str
    initialization: str
    initial_scale: float


PROFILES = {
    "smoke": MCMCProfile(
        name="smoke",
        nwalkers=20,
        nsteps=2,
        burn_in=1,
        discretization=10.0,
        interpolation_points=24,
        samples="uniform",
        initialization="local",
        initial_scale=0.002,
    ),
    "tutorial": MCMCProfile(
        name="tutorial",
        nwalkers=100,
        nsteps=200,
        burn_in=0,
        discretization=10.0,
        interpolation_points=100,
        samples="adaptive",
        initialization="prior",
        initial_scale=0.0,
    ),
    "long": MCMCProfile(
        name="long",
        nwalkers=100,
        nsteps=1000,
        burn_in=200,
        discretization=10.0,
        interpolation_points=100,
        samples="adaptive",
        initialization="local",
        initial_scale=0.002,
    ),
}


def tutorial_mcmc_parameters() -> dict[str, Any]:
    """Recreate tutorial 12's MCMC input from its least-squares result."""
    initial = json.loads(PARAMETERS_PATH.read_text(encoding="utf-8"))
    initial.pop("r_squared", None)

    # These are the three explicit changes made in tutorial 12 before sampling.
    initial["system"]["phase_shift"]["fixed"] = True
    initial["system"]["mass_ratio"] = {
        "value": initial["system"]["mass_ratio"]["value"],
        "fixed": False,
        "sigma": 0.04,
        "min": 0.001,
        "max": 2.0,
        "unit": None,
    }
    initial["primary"]["t_eff"] = {
        "value": initial["primary"]["t_eff"]["value"],
        "fixed": False,
        "sigma": 380,
        "min": 3500,
        "max": 10000,
        "unit": u.K,
    }
    initial["nuisance"] = {
        "ln_f": {
            "value": -5,
            "fixed": False,
            "min": -10,
            "max": 2,
        },
    }
    return initial


def normalized_parameter_space(
    initial: dict[str, Any],
) -> tuple[tuple[str, ...], np.ndarray, dict[str, tuple[float, float]]]:
    """Return free labels, normalized start, and physical normalization bounds."""
    parsed = BinaryInitialParameters(**deepcopy(initial))
    fitable = parsed.get_fitable(jsonify=False)
    normalization = {
        label: bounds
        for label, bounds in parsed.get_normalization_map().items()
        if label in fitable
    }
    values = np.array([parameter.value for parameter in fitable.values()], dtype=float)
    center = np.asarray(
        parameters.vector_normalizer(values, fitable.keys(), normalization),
        dtype=float,
    )
    return tuple(fitable), center, normalization


def seeded_initial_state(
    center: np.ndarray,
    profile: MCMCProfile,
    seed: int,
) -> np.ndarray:
    """Build the explicit walker state supplied to standard ELISa/emcee."""
    rng = np.random.default_rng(seed)
    shape = (profile.nwalkers, center.size)
    if profile.initialization == "prior":
        state = rng.uniform(0.0, 1.0, shape)
    elif profile.initialization == "local":
        state = rng.normal(center, profile.initial_scale, shape)
        state = np.clip(state, 1e-6, 1.0 - 1e-6)
    else:
        msg = f"Unknown initialization mode: {profile.initialization}"
        raise ValueError(msg)
    state[0] = center
    return state


def chain_diagnostics(  # noqa: C901, PLR0912, PLR0915
    sampler: emcee.EnsembleSampler,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize execution checks and conservative convergence diagnostics."""
    chain = sampler.get_chain()
    log_probability = sampler.get_log_prob()
    acceptance = np.asarray(sampler.acceptance_fraction)
    nsteps, nwalkers, ndim = chain.shape

    if log_probability.shape != (nsteps, nwalkers):
        msg = f"Unexpected log-probability shape: {log_probability.shape}."
        raise AssertionError(msg)
    if ndim != len(labels):
        msg = f"Chain has {ndim} dimensions for {len(labels)} labels."
        raise AssertionError(msg)
    if not np.all(np.isfinite(log_probability)):
        msg = "The production chain contains non-finite log probabilities."
        raise AssertionError(msg)
    if np.any((chain < 0.0) | (chain > 1.0)):
        msg = "The normalized production chain left the declared [0, 1] bounds."
        raise AssertionError(msg)

    usable = log_probability > INVALID_LOG_PROBABILITY_FLOOR
    if not np.any(usable):
        msg = "No production sample had a usable (non-sentinel) log probability."
        raise AssertionError(msg)

    moved = (
        np.any(np.abs(np.diff(chain, axis=0)) > 0.0, axis=(0, 2))
        if nsteps > 1
        else np.zeros(nwalkers, dtype=bool)
    )
    accepted = acceptance > 0.0
    if nsteps > 1 and not np.any(moved & accepted):
        msg = "No walker both moved and accepted a production proposal."
        raise AssertionError(msg)

    tau: np.ndarray | None = None
    tau_error: str | None = None
    if nsteps >= MIN_STEPS_FOR_AUTOCORRELATION:
        try:
            candidate = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
            if np.all(np.isfinite(candidate)) and np.all(candidate > 0.0):
                tau = candidate
            else:
                tau_error = "non-finite or non-positive autocorrelation estimate"
        except (emcee.autocorr.AutocorrError, FloatingPointError, ValueError) as exc:
            tau_error = str(exc)
    else:
        tau_error = f"requires at least {MIN_STEPS_FOR_AUTOCORRELATION} production steps"

    mean_acceptance = float(np.mean(acceptance))
    acceptance_ok = ACCEPTANCE_RANGE[0] <= mean_acceptance <= ACCEPTANCE_RANGE[1]
    convergence_reasons: list[str] = []
    effective_samples: list[float] | None = None
    if tau is None:
        convergence_reasons.append(f"autocorrelation unavailable: {tau_error}")
    else:
        effective = nwalkers * nsteps / tau
        effective_samples = effective.tolist()
        if np.any(nsteps < AUTOCORRELATION_MULTIPLE * tau):
            convergence_reasons.append(
                f"chain is shorter than {AUTOCORRELATION_MULTIPLE:.0f} autocorrelation times",
            )
        if np.any(effective < MIN_EFFECTIVE_SAMPLES):
            convergence_reasons.append(
                f"effective samples below {MIN_EFFECTIVE_SAMPLES:.0f} in at least one dimension",
            )
    if not acceptance_ok:
        convergence_reasons.append(
            f"mean acceptance outside [{ACCEPTANCE_RANGE[0]}, {ACCEPTANCE_RANGE[1]}]",
        )

    return {
        "chain_shape": list(chain.shape),
        "log_probability_shape": list(log_probability.shape),
        "all_log_probabilities_finite": True,
        "usable_log_probability_count": int(np.count_nonzero(usable)),
        "sentinel_log_probability_count": int(usable.size - np.count_nonzero(usable)),
        "log_probability_range": [float(np.min(log_probability)), float(np.max(log_probability))],
        "normalized_chain_range": [float(np.min(chain)), float(np.max(chain))],
        "mean_acceptance_fraction": mean_acceptance,
        "acceptance_fraction_range": [float(np.min(acceptance)), float(np.max(acceptance))],
        "accepted_walker_count": int(np.count_nonzero(accepted)),
        "moved_walker_count": int(np.count_nonzero(moved)),
        "autocorrelation_time": None if tau is None else tau.tolist(),
        "effective_samples": effective_samples,
        "converged": not convergence_reasons,
        "convergence_reasons": convergence_reasons,
    }


def run_mcmc(profile: MCMCProfile, seed: int) -> dict[str, Any]:
    """Run standard ELISa's public tutorial fitting API and return a JSON report."""
    observations = LCData.load_from_file(OBSERVATIONS_PATH, x_unit=None, y_unit=None)
    observations.convert_to_phases(PERIOD_DAYS, PRIMARY_MINIMUM_BJD, centre=0.0)
    initial = tutorial_mcmc_parameters()
    labels, center, normalization = normalized_parameter_space(initial)
    if labels != EXPECTED_FREE_PARAMETERS:
        msg = f"Tutorial free-parameter order changed: {labels}."
        raise AssertionError(msg)
    if profile.nwalkers < 2 * len(labels):
        msg = f"{profile.nwalkers} walkers are insufficient for {len(labels)} dimensions."
        raise ValueError(msg)

    initial_state = seeded_initial_state(center, profile, seed)
    initial_state_sha256 = hashlib.sha256(initial_state.tobytes()).hexdigest()
    # emcee seeds its internal legacy RandomState from NumPy's global state.
    np.random.seed(seed)  # noqa: NPY002
    settings.configure(NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=1)

    task = LCBinaryAnalyticsTask(
        data={"Kepler": observations},
        method="mcmc",
        expected_morphology="detached",
    )
    started = perf_counter()
    task.fit(
        x0=initial,
        nsteps=profile.nsteps,
        nwalkers=profile.nwalkers,
        burn_in=profile.burn_in,
        initial_state=initial_state,
        discretization=profile.discretization,
        interp_treshold=profile.interpolation_points,
        progress=False,
        save=False,
        samples=profile.samples,
    )
    elapsed_seconds = perf_counter() - started

    sampler = task.fit_cls.fit_method_instance.last_sampler
    diagnostics = chain_diagnostics(sampler, labels)
    expected_shape = [profile.nsteps, profile.nwalkers, len(labels)]
    if diagnostics["chain_shape"] != expected_shape:
        msg = f"Expected production chain shape {expected_shape}, found {diagnostics['chain_shape']}."
        raise AssertionError(msg)
    return {
        "profile": asdict(profile),
        "seed": seed,
        "elapsed_seconds": elapsed_seconds,
        "observation_count": len(observations.x_data),
        "free_parameters": list(labels),
        "physical_bounds": {key: list(value) for key, value in normalization.items()},
        "normalized_initial_center": center.tolist(),
        "initial_state_sha256": initial_state_sha256,
        "planned_likelihood_proposals": profile.nwalkers * (profile.burn_in + profile.nsteps),
        "diagnostics": diagnostics,
        "versions": {
            "python": platform.python_version(),
            "elisa": elisa.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "emcee": emcee.__version__,
        },
        "interpretation": {
            "execution_correct": True,
            "historical_saved_chain_used": False,
            "scientifically_converged": diagnostics["converged"],
        },
    }


def _profile_from_args(args: argparse.Namespace) -> MCMCProfile:
    profile = PROFILES[args.profile]
    replacements = {
        field: getattr(args, field)
        for field in ("nwalkers", "nsteps", "burn_in", "interpolation_points")
        if getattr(args, field) is not None
    }
    return replace(profile, **replacements)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--seed", type=int, default=4851217)
    parser.add_argument("--nwalkers", type=int)
    parser.add_argument("--nsteps", type=int)
    parser.add_argument("--burn-in", dest="burn_in", type=int)
    parser.add_argument("--interpolation-points", dest="interpolation_points", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_mcmc(_profile_from_args(args), args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    sys.stdout.write(f"{rendered}\n")
    if args.output is not None:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
