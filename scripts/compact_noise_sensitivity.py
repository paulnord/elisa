"""Evaluate a correlated-residual sensitivity check at a qualified physical position."""

import argparse
from pathlib import Path

import numpy as np

from elisa import settings
from scripts import sample_reemission_chain as chain
from scripts import sample_reflected_binary as sampler
from scripts.audit_reflected_noise import evaluate
from scripts.compact_interior_fit import compact_inputs, qualified_evidence
from scripts.receiver_phase_parts import join
from scripts.validate_gpu_atmosphere import table_settings
from scripts.validate_gpu_fit import physical


def sensitivity(checkpoint, directory, *, injection=False):
    evidence = qualified_evidence(checkpoint, directory, injection)
    state, fit, vector, key, prior, base = compact_inputs(checkpoint, injection)
    step = state["step"] + 1
    parts = [
        chain.read_block(
            chain.block_path(directory / "checkpoint.json", step, "six", 4, part),
            chain.block_binding(key, "six", 4, part),
        )
        for part in (0, 1)
    ]
    profiles = join(base, key, parts, symmetric_interior=4, node_reemission=True)["profiles"]
    profile, = [p for p in profiles if p["donor_level"] == 2]
    reported = dict(prior=prior, physical=physical(fit, vector))
    independent = evaluate(fit, profile, reported, 0.0, 0.01)
    np.testing.assert_allclose(
        independent["log_posterior"], evidence["report"]["usable_log_posterior"], rtol=0, atol=1e-6
    )
    correlated = evaluate(fit, profile, reported, 0.25, 0.01)
    return dict(rho=0.25, unwrapped_phase_length=0.01, independent=independent, correlated=correlated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--injection", action="store_true")
    args = parser.parse_args()
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        sampler.write_record(args.output, sensitivity(args.checkpoint, args.directory, injection=args.injection))
