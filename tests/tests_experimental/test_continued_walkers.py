from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from elisa import settings
from elisa.experimental.gpu_protocol import ResolutionNotQualifiedError
from elisa.experimental.shadow_refinement import deeper_shadow_packet
from scripts import sample_reflected_binary as sampler
from scripts.continue_reemission_walkers import initialize, snapshot
from scripts.profile_walker_eclipse import CountedGeometry, PacketProfile
from scripts.reemission_walker_fields import rule_for
from scripts.reflected_trial import fixture
from scripts.validate_gpu_atmosphere import table_settings
from scripts.walker_roche_geometry import WalkerRocheGeometry


def saved():
    return sampler.read_record(sampler.ROOT / "docs/ada_validation/background-m042-starts-reference.json")[:2]


def test_next_generation_preserves_streams_and_is_atomic(tmp_path):
    states = saved()
    dest = tmp_path / "next"
    with table_settings(), settings.override(REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2):
        # pytest selects its own Astropy constants. Bind this I/O test to its actual observations.
        fit, _ = fixture()
        for state in states:
            state["target"]["original_fit"] = json.loads(sampler.encoded(sampler.context(fit)))
        initialize(states, dest)
        assert [sampler.read_record(d) for d in sorted(dest.glob("walker-*/start.json"))] == states
        with pytest.raises(FileExistsError):
            initialize(states[::-1], dest)
        wrong = deepcopy(states)
        wrong[1]["events"] = []
        with pytest.raises(ValueError, match="generation"):
            initialize(wrong, tmp_path / "wrong")
        with pytest.raises(ValueError, match="independently"):
            initialize(states[:1] * 2, tmp_path / "duplicate")
    assert list(tmp_path.iterdir()) == [dest]


def test_cohort_cannot_advance_a_resolution_stop(tmp_path):
    state = saved()[0]
    path = tmp_path / "walker-00"
    path.mkdir()
    step, key = state["step"] + 1, sampler.trial(state)[-1]
    for file in ("state.json", "start.json"):
        sampler.write_record(path / file, state)
    sampler.write_record(
        path / f"assessment-{step:06d}.json",
        dict(
            key=key,
            outside_prior=False,
            evidence={"report": {"pass_finite_check": False}},
        ),
    )
    sampler.write_record(
        path / "result.json",
        dict(status="accuracy_stop", sampled_step=step, trial=key, source_step=state["step"], seed=state["seed"]),
    )
    with pytest.raises(ResolutionNotQualifiedError):
        snapshot(tmp_path)
    assert sampler.read_record(path / "state.json") == state


@pytest.mark.parametrize("eclipse", [False, True])
def test_packet_profiler_keeps_physical_arrays_and_counts_stationary_candidates(eclipse):
    star = SimpleNamespace(surface_potential=12.0, synchronicity=1.0)
    companion = SimpleNamespace(surface_potential=24.0, synchronicity=1.0)
    model = SimpleNamespace(
        mass_ratio=0.8,
        primary=star,
        secondary=companion,
        morphology="detached",
        eccentricity=0.0,
        has_spots=lambda: False,
        has_pulsations=lambda: False,
    )
    triangles = np.array([[[1.0, 0.88, 0.08], [1.0, 1.1, 0.08], [1.0, 1.0, 0.22]]])
    line = np.array([1.0, 0.0, 0.0])
    rule = rule_for("six", 3)
    p = PacketProfile(deeper_shadow_packet)
    measured = p(
        triangles,
        CountedGeometry(model, "primary"),
        line,
        rule,
        CountedGeometry(model, "secondary") if eclipse else None,
    )
    actual = deeper_shadow_packet(
        triangles,
        WalkerRocheGeometry(model, "primary"),
        line,
        rule,
        WalkerRocheGeometry(model, "secondary") if eclipse else None,
    )
    for key in ("parents", "area", "mu"):
        np.testing.assert_array_equal(actual[key], measured[key])
    np.testing.assert_array_equal(actual.rays, measured.rays)
    row = p.rows[0]
    assert row["surface_rays"] > 0
    assert 0 <= row["stationary_rays"] <= row["shadow_rays"]
    assert (row["stationary_rays"] > 0) == eclipse
    for category in ("stage_seconds", "physical_operation_seconds"):
        assert sum(row[category].values()) == pytest.approx(row["packet_seconds"])


def test_generation_audit_rejects_a_dropped_seed_and_rng_substitution():
    from scripts.assess_walker_chains import verify_continuation

    prior = {s["seed"]: s for s in saved()}
    verify_continuation(prior, deepcopy(prior))
    other = deepcopy(prior)
    other.pop(next(iter(other)))
    with pytest.raises(ValueError, match="membership"):
        verify_continuation(prior, other)
    other = deepcopy(prior)
    first, second = other.values()
    second["rng"] = first["rng"]
    with pytest.raises(ValueError, match="RNG"):
        verify_continuation(prior, other)


def test_rehashed_phase_block_cannot_seed_another_generation(tmp_path):
    from scripts import benchmark_reemission_walkers as walkers
    from scripts import sample_reemission_chain as chain

    older = tmp_path / "preceding"
    walkers.initialize(older, "reference", 2)
    start = older / "walker-00/start.json"
    state = sampler.read_record(start)
    final = saved()[0]
    step, key = state["step"] + 1, sampler.trial(state)[-1]
    evidence = final["events"][-1]["evidence"]
    sampler.write_record(start.with_name("state.json"), final)
    sampler.write_record(
        start.with_name(f"assessment-{step:06d}.json"),
        dict(
            key=key,
            evidence=evidence,
            outside_prior=False,
        ),
    )
    sampler.write_record(
        start.with_name("result.json"),
        dict(
            trial=key,
            sampled_step=step,
            source_step=state["step"],
            seed=state["seed"],
            status="rejected",
        ),
    )
    for rule in ("six", "duffy"):
        for depth in (3, 4):
            for part in (0, 1):
                name = chain.block_path(start, step, rule, depth, part)
                sampler.write_record(
                    name, dict(binding=chain.block_binding(key, rule, depth, part), block={"changed": True})
                )
    with pytest.raises(ValueError, match="physical curves"):
        snapshot(older, 1)
