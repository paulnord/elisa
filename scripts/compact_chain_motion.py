"""Replay portable compact-target chains and summarize movement in the two fractional radii and inclination."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from elisa import settings
from scripts import sample_reflected_binary as sampler
from scripts.assess_walker_chains import coordinates
from scripts.compare_reemission_workers import acceptance_margins
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings


def replay_history(final, initial):
    """Reproduce the random draws and recheck all finite-resolution decisions before counting a jump."""
    position = list(initial)
    density = final["initial"]["usable_log_posterior"]
    corners = final["initial_corner_log_posteriors"]
    if not final["initial"]["pass_finite_check"] or not np.isfinite(density):
        raise ValueError("Unqualified starting point")
    rng = np.random.PCG64(final["seed"]).state
    trace = [position]
    decisions = []
    for step, event in enumerate(final["events"], start=1):
        payload = dict(final, rng=rng, position=position, step=step - 1)
        proposal, u, rng, _ = sampler.trial(payload)
        if not np.array_equal(proposal, event["proposed"]) or step != event["step"] or u != event["log_u"]:
            raise ValueError("Physical proposal differs from the recorded random stream")
        e = event["evidence"]
        if event["outside_prior"]:
            if e is not None:
                raise ValueError("Outside-prior event contains a physical density")
            accepted = False
        else:
            report = e["report"]
            d = report["usable_log_posterior"]
            if not report["pass_finite_check"] or not np.isfinite(d):
                raise ValueError("Completed physical event lacks its resolution check")
            lo, hi = acceptance_margins(dict(log_uniform=u), corners, e)
            accepted = u < min(0.0, d - density)
            if not np.isfinite([lo, hi]).all() or (lo <= 0 if accepted else hi >= 0):
                raise ValueError("Unresolved acceptance margin in the selected history")
            if accepted:
                position, corners, density = proposal.tolist(), e["corner_log_posteriors"], d
        if (accepted, position, density) != (event["accepted"], event["selected"], event["selected_density"]):
            raise ValueError("Event differs from the reconstructed movement")
        trace.append(position)
        decisions.append(bool(accepted))
    if (len(decisions), position, density, rng) != (final["step"], final["position"], final["density"], final["rng"]):
        raise ValueError("Final state differs from its history")
    return trace, decisions


def summarize(paths):
    """Describe actual selected jumps. Rejected steps contribute zero."""
    output = {}
    for key in paths[0][0]:
        series = [np.asarray([v[key] for v in path]) for path in paths]
        jumps = [np.diff(v) for v in series]
        pooled = np.concatenate(jumps)
        spans = [np.ptp(v) for v in series]
        output[key] = dict(
            median_within_chain_range=float(np.median(spans)),
            max_within_chain_range=float(max(spans)),
            streams_with_movement=int(np.count_nonzero(spans)),
            streams_moving_both_directions=sum(bool(np.any(d > 0) and np.any(d < 0)) for d in jumps),
            rms_selected_jump=float(np.sqrt(np.mean(pooled**2))) if pooled.size else None,
            selected_endpoint_range=[float(min(v[-1] for v in series)), float(max(v[-1] for v in series))],
        )
    return output


def measure(checkpoints):
    fit, initial = fixture()

    @lru_cache(maxsize=None)
    def physical_position(pos):
        return coordinates(fit, pos)

    paths = []
    rows = []
    seen = set()
    for a, b in zip(checkpoints["reference"], checkpoints["gpu"], strict=True):
        if (
            a["seed"] in seen
            or a["backend"] != "reference"
            or b["backend"] != "gpu"
            or any(a[k] != b[k] for k in ("target", "scales", "seed"))
        ):
            raise ValueError("Expected paired independent streams")
        seen.add(a["seed"])
        trace, decisions = replay_history(a, initial)
        if (trace, decisions) != replay_history(b, initial):
            raise ArithmeticError("CPU and GPU physical histories differ")
        coords = [physical_position(tuple(v)) for v in trace]
        paths.append(coords)
        rows.append(dict(seed=a["seed"], decisions=decisions, selected=coords))
    return dict(streams=rows, movement=summarize(paths))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        sampler.write_record(args.output, measure(sampler.read_record(args.checkpoints)))
