"""Compare independent seeded CPU/GPU physical chains, retaining every finite-resolution stop."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_reflected_transitions import audit
from scripts.sample_reflected_binary import digest, read_record


def compare_pairs(pairs):
    if len(pairs) < 2:
        raise ValueError("Supply at least two independently seeded CPU/GPU pairs")
    target = None
    seeds = set()
    chains = []
    for reference, device in pairs:
        if reference["seed"] in seeds:
            raise ValueError("Seeds must differ across pairs")
        seeds.add(reference["seed"])
        identity = digest([reference[k] for k in ("target", "scales")])
        if target is not None and identity != target:
            raise ValueError("Chains use different model, data, prior or proposal scales")
        target = identity
        transitions = audit(reference, device)
        if not transitions:
            raise ValueError("An empty chain cannot test transition equivalence")
        position = np.array(reference["initial"]["normalized"])
        squared_jumps = []
        max_delta = 0.0
        for step, gpu_step, verdict in zip(reference["events"], device["events"], transitions, strict=True):
            previous = position
            position = np.asarray(step["selected"])
            squared_jumps.append(float(np.sum((position - previous) ** 2)))
            if step["report"] is None:
                if not verdict["outside_prior"]:
                    raise ValueError("Missing physical likelihood")
                continue
            if not step["report"]["finite_pass"] or not gpu_step["report"]["finite_pass"]:
                raise ValueError("An accuracy stop cannot be counted as a transition")
            max_delta = max(
                max_delta,
                *(
                    abs(value - gpu_step["report"]["corner_log_posteriors"][key])
                    for key, value in step["report"]["corner_log_posteriors"].items()
                ),
            )
        chains.append(
            dict(
                seed=reference["seed"],
                transitions=transitions,
                squared_jumps_normalized=squared_jumps,
                final_normalized=position.tolist(),
                final_log_posterior=reference["density"],
                maximum_corner_backend_difference=max_delta,
            )
        )
    return dict(target_sha256=target, chains=chains, posterior_convergence_assessed=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", type=Path, required=True, metavar=("CPU", "GPU"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = compare_pairs([(read_record(a), read_record(b)) for a, b in args.pair])
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
