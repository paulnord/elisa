import numpy as np
import pytest

from scripts.compact_cohort import decision


def corners(x):
    return {rule: {str(i): x for i in range(8)} for rule in ("six", "duffy")}


def position():
    return dict(
        step=2, position=[0.5], density=-1.0, events=[], rng={"old": True}, initial_corner_log_posteriors=corners(-1.0)
    )


def evidence(corner, density, passed=True):
    return dict(
        evidence=dict(
            corner_log_posteriors=corners(corner), report=dict(pass_finite_check=passed, usable_log_posterior=density)
        )
    )


@pytest.mark.parametrize(
    "corner,density,status", [(0.0, 0.0, "accepted"), (-2.0, -2.0, "rejected"), (-2.0, 0.0, "decision_stop")]
)
def test_all_start_and_proposal_resolutions_gate_the_transition(corner, density, status):
    state = position()
    result, observed, margins = decision(state, 0.0, np.array([0.55]), -0.5, {"next": True}, evidence(corner, density))
    assert observed == status and len(margins) == 2
    if observed == "decision_stop":
        assert result is state
    else:
        assert result["step"] == 3 and result["rng"] == {"next": True}
        assert result["position"] == ([0.55] if observed == "accepted" else [0.5])


@pytest.mark.parametrize(
    "assessment,status",
    [
        (dict(evidence=None, status="node_cap", failed_blocks=[0]), "node_cap"),
        (evidence(-1.0, None, False), "accuracy_stop"),
    ],
)
def test_failed_draw_retains_the_identical_start_and_rng(assessment, status):
    state = position()
    result, observed, margins = decision(state, 0.0, np.array([0.55]), -0.5, None, assessment)
    assert result is state and observed == status and margins is None


def test_only_original_prior_rejection_advances_without_a_light_curve():
    state = position()
    result, outcome, _ = decision(state, -np.inf, np.array([-2.0]), -0.5, {"next": True}, None)
    assert result["step"] == 3 and outcome == "outside_prior" and result["position"] == state["position"]
    with pytest.raises(ValueError, match="resource stop"):
        decision(state, 0.0, np.array([0.55]), -0.5, None, dict(evidence=None))


def test_resume_checks_saved_rng_before_reusing_a_completed_walker(tmp_path, monkeypatch):
    from scripts import compact_cohort as cohort
    from scripts import sample_reflected_binary as sampler

    state = dict(position(), seed=1, backend="reference", target={}, scales=[0.1], rng=np.random.PCG64(1).state)
    p = tmp_path / "walker-00" / "start.json"
    out = p.parent / "draw"
    out.mkdir(parents=True)
    sampler.write_record(p, state)
    vector, u, rng, key = sampler.trial(state)
    selected, status, margins = decision(state, -np.inf, vector, u, rng, None)
    sampler.write_record(out / "state.json", selected)
    sampler.write_record(out / "result.json", dict(seed=1, step=3, trial=key, status=status, margins=margins))
    monkeypatch.setattr(cohort, "compact_inputs", lambda *_: (state, None, vector, key, -np.inf, None))
    cohort.pool(tmp_path, 1, None, resume=True)
    assert sampler.read_record(tmp_path / "resumed-pool.json")["evaluated"] == 0
    assert not (tmp_path / "pool.json").exists()
    sampler.write_record(out / "state.json", dict(selected, rng=state["rng"]))
    with pytest.raises(ValueError, match="RNG"):
        cohort.pool(tmp_path, 1, None, resume=True)


def test_paired_audit_can_load_a_completed_resumed_cohort(tmp_path):
    from scripts.assess_compact_cohort import checked_paths
    from scripts import sample_reflected_binary as sampler

    rows = []
    for i in range(2):
        p = tmp_path / f"walker-{i:02d}"
        (p / "draw").mkdir(parents=True)
        sampler.write_record(p / "start.json", dict(seed=i))
        row = dict(seed=i, pid=i)
        sampler.write_record(p / "draw/result.json", row)
        rows.append(row)
    sampler.write_record(tmp_path / "resumed-pool.json", dict(workers=1, evaluated=1, rows=rows))
    assert len(checked_paths(tmp_path)) == 2
    sampler.write_record(tmp_path / "pool.json", dict(workers=1, evaluated=2, rows=rows))
    with pytest.raises(ValueError, match="worker count"):
        checked_paths(tmp_path)
