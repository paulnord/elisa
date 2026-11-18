"""Audit guarded Metropolis transitions against each finite resolution and original facet fits."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from scripts.sample_reflected_binary import read_record
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import make_problem, prepare_fit


def audit(cpu, gpu):
    for k in ("target", "seed", "scales", "step", "rng", "position"):
        if cpu[k] != gpu[k]:
            raise ValueError(f"Checkpointed backends differ in {k}")
    if (cpu["backend"], gpu["backend"]) != ("reference", "gpu"):
        raise ValueError("Supply serial CPU and GPU chains in order")
    rng = np.random.default_rng(cpu["seed"])
    position = np.array(cpu["initial"]["normalized"])
    current = cpu["initial"]
    results = []
    for a, b in zip(cpu["events"], gpu["events"], strict=True):
        for k in ("step", "accepted", "proposed", "selected", "log_u"):
            if a[k] != b[k]:
                raise ValueError(f"Backend transition differs in {k}")
        # Generate the physical draws again without using the checkpoint transition helper.
        proposed = position + np.asarray(cpu["scales"]) * rng.standard_normal(len(position))
        log_u = float(np.log1p(-rng.uniform()))
        np.testing.assert_array_equal(proposed, a["proposed"])
        np.testing.assert_equal(log_u, a["log_u"])
        tested = a["report"]
        if tested is None:
            if np.all((proposed >= 0) & (proposed <= 1)) or a["accepted"]:
                raise ValueError("Incorrect outside-prior rejection")
            results.append(dict(step=a["step"], outside_prior=True))
            continue
        new, old = tested["corner_log_posteriors"], current["corner_log_posteriors"]
        deltas = {k: new[k] - old[k] for k in old}
        fine_delta = tested["usable_log_posterior"] - current["usable_log_posterior"]
        accepted = log_u < min(0.0, fine_delta)
        if accepted != a["accepted"] or not tested["finite_pass"] or not b["report"]["finite_pass"]:
            raise ValueError("Failed guarded Metropolis comparison")
        for k, value in new.items():
            np.testing.assert_allclose(value, b["report"]["corner_log_posteriors"][k], rtol=0, atol=1e-6)
        # Each side can vary independently across the eight measured models.
        extremes = [min(new.values()) - max(old.values()), max(new.values()) - min(old.values())]
        margins = [min(0.0, v) - log_u for v in extremes]
        results.append(
            dict(
                step=a["step"],
                normalized=proposed.tolist(),
                log_u=log_u,
                accepted=accepted,
                delta_log_posterior=fine_delta,
                per_corner_delta=deltas,
                cross_corner_acceptance_margin=margins,
                uniform_decision=all((v > 0) == accepted for v in margins),
                nominal_bound=tested["nominal_bound"],
                physical=tested["physical"],
            )
        )
        if accepted:
            position, current = proposed, tested
        np.testing.assert_array_equal(position, a["selected"])
        np.testing.assert_equal(current["usable_log_posterior"], a["selected_density"])
    if rng.bit_generator.state != cpu["rng"]:
        raise ValueError("Restored random number state differs from independent replay")
    return results


def original(chain, degrees):
    problem = make_problem(free_nuisance=True, shape_parameters=True)
    problem.setup["discretization"] = degrees
    fit = prepare_fit(problem)
    rng = np.random.default_rng(chain["seed"])
    position = np.asarray(chain["initial"]["normalized"])
    start = perf_counter()
    current = fit.ln_probability(position)
    init = perf_counter() - start
    fixed_position = position
    rows = []
    for event in chain["events"]:
        trial = position + rng.standard_normal(len(position)) * chain["scales"]
        log_u = np.log1p(-rng.uniform())
        started = perf_counter()
        density = fit.ln_probability(trial)
        seconds = perf_counter() - started
        delta = density - current
        accept = bool(log_u < min(0.0, delta))
        if accept:
            current, position = density, trial
        fixed_delta = fit.ln_probability(np.asarray(event["proposed"])) - fit.ln_probability(fixed_position)
        fixed_position = np.asarray(event["selected"])
        rows.append(
            dict(
                step=event["step"],
                proposed=trial.tolist(),
                accepted=accept,
                selected=position.tolist(),
                selected_log_posterior=float(current),
                delta_log_posterior=float(delta),
                proposal_seconds=seconds,
                conditional_delta_on_curved_path=float(fixed_delta),
                conditional_accept_on_curved_path=bool(log_u < min(0.0, fixed_delta)),
            )
        )
    return dict(facet_degrees=degrees, initial_density_seconds=init, rows=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("gpu", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cpu, gpu = read_record(args.reference), read_record(args.gpu)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = dict(transitions=audit(cpu, gpu), original_facet=[original(cpu, d) for d in (12, 8, 6)])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
