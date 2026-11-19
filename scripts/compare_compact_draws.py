"""Audit a paired physical proposal before either compact-interior chain advances."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from elisa import settings
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_stationary_eclipse import inputs, run
from scripts.compare_reemission_workers import acceptance_margins
from scripts.compare_stationary_eclipse import compare, read_grid
from scripts.validate_gpu_atmosphere import table_settings


def selected_corners(state):
    for event in reversed(state["events"]):
        if event["accepted"]:
            return event["evidence"]["corner_log_posteriors"]
    return state["initial_corner_log_posteriors"]


def audit(reference, reference_blocks, gpu, gpu_blocks):
    starts = [sampler.read_record(p) for p in (reference, gpu)]
    if [s["backend"] for s in starts] != ["reference", "gpu"] or "compact_interior" not in starts[0]["target"]:
        raise ValueError("Expected reference and GPU checkpoints for the compact-interior target")
    for key in ("target", "rng", "seed", "position", "scales", "step"):
        if starts[0][key] != starts[1][key]:
            raise ValueError(f"Different starting {key}")
    np.testing.assert_allclose(starts[0]["density"], starts[1]["density"], atol=1e-6, rtol=0)
    prepared = [inputs(p, False, compact=True) for p in (reference, gpu)]
    np.testing.assert_array_equal(prepared[0][2], prepared[1][2])
    if prepared[0][4] != prepared[1][4]:
        raise ValueError("Original physical priors differ")
    vector, log_uniform, _, _ = sampler.trial(starts[0])
    result = dict(seed=starts[0]["seed"], step=starts[0]["step"] + 1, proposed=vector, log_uniform=log_uniform)
    if prepared[0][5] is None:
        return dict(result, status="outside_prior", accepted=False, comparisons={})
    cases = [(reference, reference_blocks), (gpu, gpu_blocks)]
    comparisons = {}
    for rule in ("six", "duffy"):
        for depth in (3, 4):
            grids = [read_grid(path, cp, False, rule, depth, compact=True) for cp, path in cases]
            comparisons[rule + str(depth)] = compare(*grids)
    assessments = [
        run(
            SimpleNamespace(
                task="assess", checkpoint=cp, output=path, fields=None, compact_interior=True, injection=False
            )
        )["evidence"]
        for cp, path in cases
    ]
    if any(e is None for e in assessments):
        if not all(e is None for e in assessments):
            raise ArithmeticError("The resource stops differ")
        return dict(result, status="node_cap", accepted=None, comparisons=comparisons)
    for rule in ("six", "duffy"):
        for corner in map(str, range(8)):
            x, y = (e["corner_log_posteriors"][rule][corner] for e in assessments)
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=0)
    reports = [e["report"] for e in assessments]
    for field in ("distance_sigma", "log_likelihood_bound", "max_measured_nominal_log_likelihood_change"):
        np.testing.assert_allclose(reports[0][field], reports[1][field], atol=1e-6, rtol=0)
    if reports[0]["pass_finite_check"] != reports[1]["pass_finite_check"]:
        raise ArithmeticError("Backends differ on the resolution check")
    if not reports[0]["pass_finite_check"]:
        return dict(result, status="accuracy_stop", accepted=None, comparisons=comparisons)
    decisions, margins = [], []
    for state, e in zip(starts, assessments, strict=True):
        margin = acceptance_margins(dict(log_uniform=log_uniform), selected_corners(state), e)
        accepted = log_uniform < min(0.0, e["report"]["usable_log_posterior"] - state["density"])
        if (accepted and margin[0] <= 0) or (not accepted and margin[1] >= 0):
            raise ArithmeticError("Transition is sensitive to the finite resolution choice")
        decisions.append(bool(accepted))
        margins.append(margin)
    if decisions[0] != decisions[1]:
        raise ArithmeticError("CPU and GPU sampler decisions differ")
    return dict(result, status="qualified", accepted=decisions[0], margins=margins, comparisons=comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "reference_blocks", "gpu", "gpu_blocks", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        sampler.write_record(args.output, audit(args.reference, args.reference_blocks, args.gpu, args.gpu_blocks))
