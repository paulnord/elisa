"""Boundary cases for a durable random-walk step with a guarded physical likelihood."""

from types import SimpleNamespace

import numpy as np
import pytest

from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts import sample_reflected_binary as sampler


def starting(width):
    return dict(
        target={},
        backend="reference",
        step=0,
        rng=np.random.PCG64(73).state,
        position=[0.5],
        scales=[width],
        density=1.0,
        events=[],
    )


def test_only_original_prior_exclusion_advances_without_a_curve(tmp_path):
    state = starting(100)
    fit = SimpleNamespace(ln_prior=lambda *args: -np.inf, norm_init_vector=[0.5], sigmas=[np.nan])
    args = SimpleNamespace(command="advance", checkpoint=tmp_path / "chain.json", step=1)
    sampler.write_record(args.checkpoint, state)
    vector, _, rng, _ = sampler.trial(state)
    assert vector[0] < 0 or vector[0] > 1
    sampler.take_step(args, fit, sampler.read_record(args.checkpoint))
    after = sampler.read_record(args.checkpoint)
    assert after["step"] == 1 and after["rng"] == rng and after["position"] == [0.5]
    assert after["events"][0]["report"] is None and not after["events"][0]["accepted"]


def test_accuracy_failure_leaves_trial_rng_and_state_intact(tmp_path, monkeypatch):
    state = starting(0.001)
    args = SimpleNamespace(command="advance", checkpoint=tmp_path / "chain.json", step=1)
    sampler.write_record(args.checkpoint, state)
    before = args.checkpoint.read_bytes()
    fit = SimpleNamespace(ln_prior=lambda *args: 0.0, norm_init_vector=[0.5], sigmas=[np.nan])
    *_, key = sampler.trial(state)
    for d in (3, 4):
        sampler.write_record(sampler.part_path(tmp_path, 1, d), dict(key=key, depth=d, profiles=[{"depth": d}]))
    monkeypatch.setattr(sampler, "validate_physics", lambda *args: None)

    def stop(profiles):
        raise ResolutionNotQualifiedError("accuracy stop", {"within_finite_budget": False})

    monkeypatch.setattr(sampler, "replay", stop)
    with pytest.raises(ResolutionNotQualifiedError):
        sampler.take_step(args, fit, sampler.read_record(args.checkpoint))
    assert args.checkpoint.read_bytes() == before
    assert sampler.trial(sampler.read_record(args.checkpoint))[-1] == key
