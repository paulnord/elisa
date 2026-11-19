from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from scripts import sample_reemission_chain as chain


def test_transition_stops_before_mutating_state():
    state = dict(position=[0.5], density=1.0, step=0, events=[], rng={"old": True})
    vector = np.array([0.51])

    def advance(prior, passed, density):
        evidence = (
            None if prior == -np.inf else dict(report=dict(pass_finite_check=passed, usable_log_posterior=density))
        )
        return chain.transition(state, 1, vector, -0.4, {"new": True}, prior, evidence)

    with pytest.raises(ResolutionNotQualifiedError):
        advance(0, False, None)
    with pytest.raises(ArithmeticError):
        advance(0, True, np.nan)
    for density, accepted in ((1.5, True), (0.75, True), (0.5, False)):
        out = advance(0, True, density)
        assert out["events"][0]["accepted"] == accepted
        assert out["position"] == (vector.tolist() if accepted else state["position"])
        assert out["density"] == (density if accepted else state["density"])
        assert out["step"] == 1 and out["rng"] == {"new": True}
    rejected = advance(-np.inf, False, None)
    assert rejected["position"] == state["position"] and rejected["events"][0]["outside_prior"]
    assert state == dict(position=[0.5], density=1.0, step=0, events=[], rng={"old": True})


def test_repeated_evaluation_binds_the_trial_and_keeps_errors(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    calls = []

    def synthesis(*args, **kwargs):
        calls.append(kwargs["symmetric_interior"])
        return dict(result=1.0)

    monkeypatch.setattr(chain, "evaluate_block", synthesis)

    def evaluate(key):
        chain.evaluate(path, 1, "six", 3, 0, {"backend": "reference"}, key, {"physical": 1})

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(evaluate, ["same", "same"]))
    assert calls == [4]
    with pytest.raises(ValueError, match="different trial"):
        evaluate("other")
    f = chain.block_path(path, 1, "six", 3, 0)
    data = f.read_bytes()
    f.write_bytes(data.replace(b'"result":1.0', b'"result":2.0'))
    with pytest.raises(
        ValueError,
        match="checksum|Checksum|digest",
    ):
        evaluate("same")
    assert calls == [4]
    with pytest.raises(ValueError, match="bounded"):
        chain.evaluate(path, 1, "duffy", 3, 0, {"backend": "gpu"}, "gpu", {})

    def unsupported(*args, **kwargs):
        raise RuntimeError("unsupported model")

    monkeypatch.setattr(chain, "evaluate_block", unsupported)
    with pytest.raises(RuntimeError, match="unsupported model"):
        chain.evaluate(path, 1, "duffy", 4, 1, {"backend": "reference"}, "same", {})
    assert not chain.block_path(path, 1, "duffy", 4, 1).exists()


def test_audit_requires_independent_backends(tmp_path):
    from scripts.compare_reemission_draws import audit

    ref = tmp_path / "reference.json"
    chain.sampler.write_record(ref, {"backend": "reference"})
    with pytest.raises(ValueError, match="distinct reference and GPU"):
        audit(ref, ref, ref, ref, 1)
