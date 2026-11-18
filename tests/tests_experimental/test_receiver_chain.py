"""Guard the distinction between prior exclusion and an unavailable physical density."""

from types import SimpleNamespace

import numpy as np
import pytest

from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts import sample_receiver_heating as chain
from scripts import sample_reflected_binary as sampler


@pytest.mark.parametrize("outside", [False, True])
def test_next_generation_preserves_accuracy_stops(tmp_path, monkeypatch, outside):
    state = dict(
        target={},
        backend="reference",
        step=0,
        rng=np.random.PCG64(73).state,
        position=[0.5],
        scales=[100.0 if outside else 0.001],
        density=1.0,
        events=[],
    )
    args = SimpleNamespace(checkpoint=tmp_path / "state.json", step=1, command="advance")
    fit = SimpleNamespace(
        ln_prior=lambda u, *_: 0.0 if 0 <= u[0] <= 1 else -np.inf, norm_init_vector=[0.5], sigmas=[np.nan]
    )
    sampler.write_record(args.checkpoint, state)
    before = args.checkpoint.read_bytes()
    proposal, _, rng_after, key = sampler.trial(state)
    assert outside == (fit.ln_prior(proposal) == -np.inf)
    if outside:
        chain.step(args, fit, sampler.read_record(args.checkpoint))
        saved = sampler.read_record(args.checkpoint)
        assert saved["step"] == 1 and saved["rng"] == rng_after
        assert saved["position"] == state["position"] and saved["density"] == state["density"]
        assert saved["events"][0]["report"] is None and not saved["events"][0]["accepted"]
    else:
        for depth in (3, 4):
            sampler.write_record(
                sampler.part_path(tmp_path, 1, depth), dict(key=key, depth=depth, profiles=[dict(depth=depth)])
            )

        def unqualified(*args):
            raise ResolutionNotQualifiedError("Outside finite resolution limit", {})

        monkeypatch.setattr(chain, "density", unqualified)
        with pytest.raises(ResolutionNotQualifiedError):
            chain.step(args, fit, sampler.read_record(args.checkpoint))
        assert args.checkpoint.read_bytes() == before
        assert sampler.trial(sampler.read_record(args.checkpoint))[-1] == key
