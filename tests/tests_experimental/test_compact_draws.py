import numpy as np
import pytest

from scripts import compare_compact_draws as checks
from scripts import sample_reflected_binary as sampler


def test_resolution_margins_follow_last_accepted_state():
    original, accepted = {"six": {"0": -2}}, {"six": {"0": -1}}
    state = dict(initial_corner_log_posteriors=original, events=[{"accepted": False, "evidence": None}])
    assert checks.selected_corners(state) == original
    state["events"].extend(
        [
            {"accepted": True, "evidence": {"corner_log_posteriors": accepted}},
            {"accepted": False, "evidence": {"corner_log_posteriors": original}},
        ]
    )
    assert checks.selected_corners(state) == accepted


def test_prior_rejection_uses_no_curves(tmp_path, monkeypatch):
    base = dict(
        target={"compact_interior": {}},
        rng=np.random.PCG64(17).state,
        seed=17,
        position=[0.5],
        scales=[0.01],
        step=0,
        density=0,
    )
    paths = [tmp_path / (backend + ".json") for backend in ("reference", "gpu")]
    for backend, p in zip(("reference", "gpu"), paths, strict=True):
        sampler.write_record(p, dict(base, backend=backend))
    monkeypatch.setattr(checks, "inputs", lambda *a, **kw: (None, None, [0.5], None, -np.inf, None))
    result = checks.audit(paths[0], tmp_path, paths[1], tmp_path)
    assert result["status"] == "outside_prior" and result["accepted"] is False and result["comparisons"] == {}
    sampler.write_record(paths[1], dict(base, backend="gpu", step=1))
    with pytest.raises(ValueError, match="starting step"):
        checks.audit(paths[0], tmp_path, paths[1], tmp_path)
