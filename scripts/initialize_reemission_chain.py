"""Start a new physical chain only after both interior rules pass the sixteen-model check."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from elisa import settings
from elisa.analytics.params import parameters
from scripts import sample_reflected_binary as sampler
from scripts.check_interior_resolution import compare
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings


def initialize(path, backend, six, duffy, seed, scale):
    sets = [[json.loads(p.read_bytes()) for p in files] for files in (six, duffy)]
    report = compare(*sets, None, node_reemission=True)
    if report["usable_log_posterior"] is None:
        from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
        raise ResolutionNotQualifiedError("Starting position exceeds the sixteen-model resolution budget", report)
    profiles = [v for parts in sets for part in parts for v in part["profiles"]]
    fit, _ = fixture()
    props = profiles[0]["samples"][0]["rows"][0]["physical"]
    position = parameters.vector_normalizer([props[k] for k in fit.fitable], fit.fitable, fit.normalization)
    sampler.validate_physics(fit, np.asarray(position), profiles, backend)
    if not np.isfinite(report["usable_log_posterior"]) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Initialization requires finite density and finite positive proposal scale")
    inventory = json.loads((sampler.ROOT / "docs/ada_validation/background-m039-source-hashes.json").read_bytes())
    sources = {name: value for name, value in inventory.items() if name.startswith(("src/elisa/", "scripts/"))}
    for name, value in sources.items():
        if hashlib.sha256((sampler.ROOT / name).read_bytes()).hexdigest() != value:
            raise ValueError(f"Changed source for the qualified starting model: {name}")
    target = dict(engine="secondary-node-reemission", source=sampler.digest(sources), original_fit=sampler.context(fit),
                  field_levels=[2, 3], donor_levels=[1, 2], eclipse_depths=[3, 4], selective_shadow=True,
                  receiver_points=3, reflections=2, slice_order=12, junction_depth=4, integration_level=3,
                  interiors=["symmetric-degree-4", "duffy-3x3"], distance_limit=report["limit_distance"],
                  likelihood_limit=report["limit_log_likelihood"])
    result = dict(target=target, backend=backend, position=list(position), step=0, events=[], initial=report,
                  initial_input_hashes=[sampler.digest(v) for parts in sets for v in parts],
                  density=report["usable_log_posterior"], seed=seed, scales=[scale]*len(position),
                  rng=np.random.PCG64(seed).state)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if sampler.digest(sampler.read_record(path)) != sampler.digest(result):
                raise ValueError("A different chain state owns this checkpoint")
        else:
            sampler.write_record(path, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("reference", "gpu"), required=True)
    p.add_argument("--six", type=Path, nargs=2, required=True)
    p.add_argument("--duffy", type=Path, nargs=2, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--scale", type=float, default=0.0006)
    args = p.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        initialize(args.checkpoint, args.backend, args.six, args.duffy, args.seed, args.scale)
