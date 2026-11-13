"""Compare the six-line history exclusion against the original physical RV fit."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import multiprocessing as mp
import statistics
import subprocess
from pathlib import Path

import numpy as np
import realistic_mcmc_transport as bench
from elisa import settings
from elisa.analytics.binary_fit import mcmc

BASE = "6dc055ba989ecb558c5df0530bee00a57a155549"


def baseline_fit():
    source = subprocess.check_output(
        ["git", "show", BASE + ":src/elisa/analytics/binary_fit/mcmc.py"], text=True
    )
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "MCMCFit")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_fit")
    namespace = {}
    exec(compile("from __future__ import annotations\n" + ast.unparse(fn),
                 "<baseline _fit>", "exec"), mcmc.__dict__, namespace)
    return namespace["_fit"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", choices=["fork", "spawn"], required=True)
    parser.add_argument("--output", default="minimal-history-results")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    candidate = mcmc.MCMCFit._fit
    bench._ORIGINAL_FIT = baseline_fit()
    bench.configure(args.context)
    settings.configure(NUMBER_OF_MCMC_PROCESSES=1)
    fresh, cfg, data, options, metadata = bench.make_rv_problem()
    p0 = bench.initial_state(fresh, cfg, data, options)
    for vector in p0[:2]:
        assert np.isfinite(fresh.ln_probability(vector))
    prior = copy.deepcopy(fresh)
    bench.run_fit(prior, cfg, data, options, p0, 8, 10, "serial", candidate, 6026)
    enlarged = copy.deepcopy(prior)
    bench.grow_retained_chain(enlarged, 10000)
    templates = {"fresh": fresh, "reused_actual": prior, "retained_10000": enlarged}
    reference, ref_chain, ref_log = bench.run_fit(
        copy.deepcopy(fresh), cfg, data, options, p0, 120, 10, "serial", candidate, 19302
    )
    report = {
        "completed": False, "context": args.context, "environment": bench.environment(),
        "model": metadata, "walkers": 32, "workers": 2, "production_steps": 120, "burn_in": 10,
        "rounds": 2,
        "notes": [
            "Retained 10000 history is tiled real samples, not a 10000-step science run.",
            "Wall time includes setup, pool startup, sampling and result summarization.",
            "Observation generation, template cloning and chain persistence are excluded.",
            "Fresh/reused cases compare complete physical RV fits; short chains do not establish convergence.",
        ],
        "records": [],
    }
    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    for round_idx in range(2):
        histories = list(templates)
        if round_idx % 2:
            histories.reverse()
        for history in histories:
            variants = ["baseline", "initialized"] if round_idx % 2 == 0 else ["initialized", "baseline"]
            for variant in variants:
                record, chain, log_prob = bench.run_fit(
                    copy.deepcopy(templates[history]), cfg, data, options, p0,
                    120, 10, variant, candidate, 19302
                )
                np.testing.assert_array_equal(chain, ref_chain)
                np.testing.assert_array_equal(log_prob, ref_log)
                assert record["final_result"] == reference["final_result"]
                assert not mp.active_children()
                record.update(variant="minimal" if variant == "initialized" else "baseline",
                              history=history, round=round_idx)
                report["records"].append(record)
                save()
                print("MEASUREMENT", json.dumps({key: record[key] for key in
                      ("variant", "history", "round", "wall_seconds")}), flush=True)
    report["summaries"] = []
    for history in templates:
        base = [r for r in report["records"] if r["history"] == history and r["variant"] == "baseline"]
        minimal = [r for r in report["records"] if r["history"] == history and r["variant"] == "minimal"]
        report["summaries"].append({
            "history": history,
            "baseline_median_seconds": statistics.median(r["wall_seconds"] for r in base),
            "minimal_median_seconds": statistics.median(r["wall_seconds"] for r in minimal),
            "paired_speedup": [b["wall_seconds"] / m["wall_seconds"] for b, m in zip(base, minimal)],
            "exact_chain_log_probability_and_result": True,
        })
    report["completed"] = True
    save()
    print("MINIMAL_HISTORY_SUMMARY", json.dumps(report["summaries"]), flush=True)


if __name__ == "__main__":
    main()
