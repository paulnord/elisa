"""Check paired physical walker pools and summarize usable likelihoods and resolution stops."""

import argparse
import json
from pathlib import Path

import numpy as np

from elisa import settings
from scripts import sample_reflected_binary as sampler
from scripts.benchmark_reemission_walkers import completed
from scripts.compare_reemission_draws import audit
from scripts.validate_gpu_atmosphere import table_settings


def starting_corners(state, initial):
    if state["step"]:
        record = next((e for e in reversed(state["events"]) if e["accepted"]), None)
        if record is not None:
            return record["evidence"]["corner_log_posteriors"]
    return {
        rule: initial[state["backend"] + suffix]["corner_log_posteriors"]
        for rule, suffix in (("six", "_n6"), ("duffy", "_n9"))
    }


def acceptance_margins(draw, initial, evidence):
    old = [value for rule in initial.values() for value in rule.values()]
    new = [value for rule in evidence["corner_log_posteriors"].values() for value in rule.values()]
    if len(old) != 16 or len(new) != 16:
        raise ValueError("Expected sixteen starting and proposed resolutions")
    corners = [min(0.0, y - x) - draw["log_uniform"] for x in old for y in new]
    return [min(corners), max(corners)]


def compare(reference, gpu):
    prior = json.loads((sampler.ROOT / "docs/ada_validation/background-m039.json").read_bytes())
    roots = {"reference": reference, "gpu": gpu}
    pools = {b: sampler.read_record(r / "pool.json") for b, r in roots.items()}
    batches = {b: sorted(r.glob("walker-*/start.json")) for b, r in roots.items()}
    workers = pools["reference"]["workers"]
    if pools["gpu"]["workers"] != workers or any(p["evaluated"] != len(batches[b]) for b, p in pools.items()):
        raise ValueError("Use fresh, matched worker counts")
    if any(len({row["pid"] for row in p["rows"]}) != workers for p in pools.values()):
        raise ValueError("The requested number of worker processes did not evaluate physical proposals")
    rows = []
    for a, b in zip(batches["reference"], batches["gpu"], strict=True):
        states = dict(reference=sampler.read_record(a), gpu=sampler.read_record(b))
        step = states["reference"]["step"] + 1
        records = {k: completed(p, states[k]) for k, p in (("reference", a), ("gpu", b))}
        if records["reference"]["status"] != records["gpu"]["status"]:
            raise ArithmeticError("Backend decisions differ")
        if records["reference"]["status"] == "outside_prior":
            # No stellar forward model or usable likelihood exists for a prior rejection.
            av = sampler.trial(states["reference"])
            bv = sampler.trial(states["gpu"])
            np.testing.assert_array_equal(av[0], bv[0])
            if av[1:3] != bv[1:3]:
                raise ValueError("Mismatched outside-prior trials")
            rows.append(dict(seed=states["gpu"]["seed"], status="outside_prior", proposed=av[0], step=step))
            continue
        pair = audit(a, a.with_name("state.json"), b, b.with_name("state.json"), step)
        row = dict(
            seed=states["reference"]["seed"],
            step=step,
            status=records["reference"]["status"],
            draw=pair["draw"],
            parity=pair["parity"],
            independent_density_error=pair["max_independent_density_difference"],
        )
        for backend, state in states.items():
            ev = pair[backend]
            margin = acceptance_margins(pair["draw"], starting_corners(state, prior["finite_models"]), ev)
            passed = ev["report"]["pass_finite_check"]
            if passed != (row["status"] != "accuracy_stop"):
                raise ValueError("Sampler status differs from its resolution check")
            density = ev["report"]["usable_log_posterior"]
            row[backend] = dict(
                starting_density=state["density"],
                proposed_density=density,
                density_change=None if density is None else density - state["density"],
                log_likelihood_bound=ev["report"]["log_likelihood_bound"],
                distance_sigma=ev["report"]["distance_sigma"],
                max_measured=ev["report"]["max_measured_nominal_log_likelihood_change"],
                margin_over_256_corner_pairs=margin,
                corner_densities=ev["corner_log_posteriors"],
                selected_position=sampler.read_record(roots[backend] / a.parent.name / "state.json")["position"],
            )
        rows.append(row)
    return dict(workers=workers, rows=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("reference", "gpu", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        sampler.write_record(args.output, compare(args.reference, args.gpu))
