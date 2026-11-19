import numpy as np
import pytest

from scripts import benchmark_reemission_walkers as benchmark
from scripts import sample_reflected_binary as sampler
from scripts.reemission_walker_fields import load_fields, rule_for, save_fields


def test_walkers_continue_the_published_checkpoints_and_draw_independently(tmp_path):
    roots = [tmp_path / k for k in ("reference", "gpu")]
    for root in roots:
        benchmark.initialize(root, root.name, 16)
    proposals = []
    for i in range(16):
        paths = [root / f"walker-{i:02d}" / "start.json" for root in roots]
        states = [sampler.read_record(p) for p in paths]
        assert all(s["step"] == int(i < 2) and s["seed"] == 20261117 + i for s in states)
        a, b = [sampler.trial(s) for s in states]
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:3] == b[1:3] and a[3] != b[3]
        proposals.append(tuple(a[0]))
    assert len(set(proposals)) == 16


def test_field_archive_requires_same_proposal_and_intact_optical_values(tmp_path):
    path = tmp_path / "fields.npz"
    key = dict(trial="first")
    data = np.linspace(1.23456, 2.34567, 36).reshape(2, 6, 3)
    save_fields(path, dict(binding=key), dict(nodes=data))
    assert np.array_equal(load_fields(path, key)[1]["nodes"], data)
    with pytest.raises(ValueError, match="belong"):
        load_fields(path, dict(trial="second"))
    raw = path.read_bytes()
    assert raw.count(data.tobytes()) == 1
    path.write_bytes(raw.replace(data.tobytes(), (data + 1).tobytes()))
    from zipfile import BadZipFile

    with pytest.raises(BadZipFile):
        load_fields(path, key)


def test_worker_limits_and_fresh_directory(tmp_path):
    root = tmp_path / "starts"
    benchmark.initialize(root, "reference", 2)
    with pytest.raises(ValueError, match="independent proposal"):
        benchmark.run_pool(root, 8)
    with pytest.raises(FileExistsError):
        benchmark.initialize(root, "reference", 2)
    for rule, size in (("duffy", 9), ("six", 6)):
        for depth in (3, 4):
            r = rule_for(rule, depth)
            assert (r.interior_size, r.slice_order, r.junction_depth) == (size, 12, 4)
    with pytest.raises(ValueError):
        rule_for("six", 2)


def test_resume_verifies_the_decision_and_rng(tmp_path):
    directory = tmp_path / "batch"
    benchmark.initialize(directory, "reference", 2)
    final = sampler.read_record(directory / "walker-00/start.json")
    start = tmp_path / "start.json"
    import json

    fixtures = json.loads((sampler.ROOT / "docs/ada_validation/background-m040-starts.json").read_bytes())
    state = fixtures["20261117"]["reference"]["data"]
    _, _, _, trial = sampler.trial(state)
    sampler.write_record(start, state)
    evidence = final["events"][0]["evidence"]
    sampler.write_record(tmp_path / "assessment-000001.json", dict(key=trial, evidence=evidence, outside_prior=False))
    sampler.write_record(tmp_path / "state.json", final)
    row = dict(seed=state["seed"], source_step=0, sampled_step=1, trial=trial, status="accepted")
    sampler.write_record(tmp_path / "result.json", row)
    assert benchmark.completed(start, state) == row
    sampler.write_record(tmp_path / "result.json", dict(row, status="rejected"))
    with pytest.raises(ValueError, match="transition"):
        benchmark.completed(start, state)
    sampler.write_record(tmp_path / "result.json", row)
    sampler.write_record(tmp_path / "state.json", dict(final, rng=state["rng"]))
    with pytest.raises(ValueError, match="transition"):
        benchmark.completed(start, state)


@pytest.mark.parametrize("q,spin", [(0.5, 0.5), (1.01, 1.0), (2.0, 1.5)])
@pytest.mark.parametrize("component", ["primary", "secondary"])
def test_tiled_bisection_matches_reference_nodes_and_areas(component, q, spin):
    from types import SimpleNamespace
    from elisa.experimental.boundary_geometry import ConditionedRocheGeometry
    from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
    from scripts.walker_roche_geometry import WalkerRocheGeometry

    star = SimpleNamespace(surface_potential=16.0, synchronicity=spin)
    system = SimpleNamespace(
        mass_ratio=q,
        morphology="detached",
        eccentricity=0.0,
        primary=star,
        secondary=star,
        has_spots=lambda: False,
        has_pulsations=lambda: False,
    )
    native = WalkerRocheGeometry(system, component)
    original = ConditionedRocheGeometry(system, component)
    rays = np.vstack((np.eye(3), -np.eye(3), np.random.default_rng(391).normal(size=(515, 3))))
    for got, want in zip(native.sample(rays), original.sample(rays), strict=True):
        np.testing.assert_array_equal(got, want)
    star.surface_potential = 3.0
    with pytest.raises(UnsupportedCurvedGeometryError, match="bracket"):
        native.sample(rays)
