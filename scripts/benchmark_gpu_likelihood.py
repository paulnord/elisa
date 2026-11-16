"""Synthetic fixed-binary pilot, not a full-physics GPU benchmark.

Run with JAX_ENABLE_X64=True. No tables are downloaded: the demonstration uses
ELISa's real Roche surface with blackbody atmospheres and explicit linear LD.
"""

# ruff: noqa: INP001 - standalone command-line script
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from elisa import BinarySystem, Star, settings
from elisa.experimental.gpu_likelihood import FixedBinaryLikelihood


def main() -> None:
    """Run an explicit-device correctness check and small sampling experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    settings.configure(LIMB_DARKENING_LAW="linear", NUMBER_OF_PROCESSES=1)
    common = {
        "mass": 1,
        "surface_potential": 5,
        "synchronicity": 1,
        "albedo": 0.5,
        "gravity_darkening": 0.32,
        "atmosphere": "bb",
        "limb_darkening_coefficients": {"bolometric": [0.5]},
        "discretization_factor": 10,
    }
    binary = BinarySystem(
        Star(t_eff=6500, **common),
        Star(t_eff=5500, **common),
        inclination=88,
        period=3,
        eccentricity=0,
        argument_of_periastron=90,
    )
    phases = np.linspace(-0.1, 1.1, 128)
    errors = np.full(phases.size, 0.005)
    started = time.perf_counter()
    prep = FixedBinaryLikelihood.from_binary(
        binary, phases, "bolometric", np.ones(phases.size), errors, device=args.device,
    )
    preparation_seconds = time.perf_counter() - started
    truth = np.array([[0.15, -2.0]])
    rng = np.random.default_rng(732)
    observed = prep.predict(truth)[0] + rng.normal(0, errors * np.sqrt(1 + np.exp(-4)))
    observed_mean = observed.mean()
    observed = observed / observed_mean
    errors = errors / observed_mean
    pilot = FixedBinaryLikelihood(prep.baseline, observed, errors, support_mean=prep.mean_ratio, device=args.device)
    proposals = rng.uniform([0.01, -5], [0.9, 1], size=(32, 2))
    started = time.perf_counter()
    result = pilot.device_log_probability(proposals)
    result.block_until_ready()
    compile_and_first_seconds = time.perf_counter() - started
    oracle = pilot.numpy_reference(proposals)
    np.testing.assert_allclose(np.asarray(result), oracle, rtol=1e-10, atol=1e-8)
    started = time.perf_counter()
    for _ in range(100):
        pilot.device_log_probability(proposals).block_until_ready()
    batch_seconds = (time.perf_counter() - started) / 100
    sampler = pilot.make_sampler(32)
    sampler.random_state = np.random.RandomState(733).get_state()
    positions = rng.uniform([0.1, -3], [0.2, -1], size=(32, 2))
    started = time.perf_counter()
    sampler.run_mcmc(positions, args.steps)
    sampling_seconds = time.perf_counter() - started
    print(  # noqa: T201 - JSON command-line report
        json.dumps(
            {
                "scope": "Fixed physical binary; only additional_light and ln_f vary. Not full GPU ELISa.",
                "device": str(pilot.device),
                "platform": pilot.device.platform,
                "dtype": str(result.dtype),
                "preparation_seconds": preparation_seconds,
                "compile_and_first_batch_seconds": compile_and_first_seconds,
                "warm_32_candidate_batch_seconds_including_input_transfer": batch_seconds,
                "max_abs_log_likelihood_difference": float(np.max(np.abs(np.asarray(result) - oracle))),
                "sampling_seconds": sampling_seconds,
                "steps": args.steps,
                "walkers": 32,
                "mean_acceptance": float(sampler.acceptance_fraction.mean()),
                "convergence": "Not established by this short smoke benchmark; no ESS or scientific fit claim.",
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
