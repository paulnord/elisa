"""Measure the same fresh, guarded physical trial on serial CPU, JAX CPU and GPU."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elisa import settings
from scripts.compare_reflected_posteriors import replay
from scripts.reflected_trial import evaluate_depth, fixture
from scripts.sample_reflected_binary import context, digest, read_record
from scripts.validate_gpu_atmosphere import table_settings


def run(args):
    state = read_record(args.checkpoint)
    if not 1 <= args.step <= state["step"]:
        raise ValueError("Select a committed physical trial")
    event = state["events"][args.step - 1]
    if event["report"] is None:
        raise ValueError("An outside-prior rejection has no finite forward model")
    fit, _ = fixture()
    if digest(state["target"]) != digest(context(fit)):
        raise ValueError("Mismatch with the original checkpointed likelihood")
    if args.command == "evaluate":
        if args.backend == "gpu" and not args.verified_bounded_gpu_window:
            raise ValueError("GPU synthesis requires a verified bounded unit")
        return evaluate_depth(fit, np.array(event["proposed"]), args.depth, args.backend)
    profiles = [p for path in args.parts for p in json.loads(path.read_bytes())["profiles"]]
    if any(p["backend"] != args.backend for p in profiles):
        raise ValueError("The two depth files must use the selected backend")
    result = replay(profiles)
    if result["physical"] != event["report"]["physical"]:
        raise ValueError("Timing replay changed the proposal")
    np.testing.assert_allclose(result["usable_log_posterior"], event["report"]["usable_log_posterior"], atol=1e-6, rtol=0)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("evaluate", "finish"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--backend", choices=("reference", "cpu", "gpu"), required=True)
    p.add_argument("--depth", type=int, choices=(3, 4))
    p.add_argument("--parts", nargs=2, type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verified-bounded-gpu-window", action="store_true")
    args = p.parse_args()
    if args.command == "finish" and not args.parts or args.command == "evaluate" and not args.depth:
        p.error("evaluate requires a depth and finish requires both depth results")
    start = perf_counter()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        result = run(args)
    result["evaluation_seconds"] = perf_counter() - start
    args.output.write_text(json.dumps(result, allow_nan=False, default=lambda a: a.tolist()) + "\n")
