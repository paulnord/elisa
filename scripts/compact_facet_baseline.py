"""Time original ELISa facet likelihoods at the same starts and proposals as a curved fit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np

from elisa import settings
from scripts import sample_reflected_binary as sampler
from scripts.assess_walker_chains import OriginalFacet
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings


def prepare(degrees):
    global fitter
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        fitter = OriginalFacet(degrees, fixture()[0]).fit


def evaluate(state):
    vector, u, _, _ = sampler.trial(state)
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        begin = perf_counter()
        start = float(fitter.ln_probability(np.asarray(state["position"])))
        proposed = float(fitter.ln_probability(vector))
        elapsed = perf_counter() - begin
    if not np.isfinite(start) or (not np.isfinite(proposed) and proposed != -np.inf):
        raise ArithmeticError("Invalid original facet density")
    return dict(
        seed=state["seed"],
        start=start,
        proposed=proposed if np.isfinite(proposed) else None,
        conditional_accept=bool(u < min(0.0, proposed - start)),
        likelihood_pair_seconds=elapsed,
    )


def timed(fixture_path, workers, degrees):
    t = perf_counter()
    states = sampler.read_record(fixture_path)["reference"]
    if len(states) < workers or len({s["seed"] for s in states}) != len(states):
        raise ValueError("Supply an independent proposal per worker")
    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn"), initializer=prepare, initargs=(degrees,)
    ) as pool:
        rows = list(pool.map(evaluate, states, chunksize=1))
    return dict(workers=workers, degrees=degrees, complete_seconds=perf_counter() - t, proposals=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, choices=(8, 16), required=True)
    p.add_argument("--degrees", type=int, choices=(6, 12), required=True)
    args = p.parse_args()
    sampler.write_record(args.output, timed(args.fixture, args.workers, args.degrees))
