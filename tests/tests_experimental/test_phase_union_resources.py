from types import SimpleNamespace

import numpy as np
import pytest

from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError
from scripts import compact_phase_union as union
from scripts import sample_reflected_binary as sampler


@pytest.mark.parametrize("capacity", [True, False])
def test_failed_union_replays_only_the_recognized_node_cap(tmp_path, monkeypatch, capacity):
    phases = np.array([0.0, 0.5])
    fit = SimpleNamespace(observer=None, y_data={"B": [1.0]})
    state = {"step": 3}
    reason = "Refined full shadow packet exceeds the original node cap" if capacity else "Invalid Roche bracket"
    calls = []
    closed = []

    class FailedObserver:
        integration_level = 3

        def __init__(self, bands, rule, **kwargs):
            self.rule = rule

        def attach(self, *args):
            pass

        def batch(self, *args, **kwargs):
            raise UnsupportedCurvedGeometryError(reason)

    def split(args):
        calls.append((args.rule, args.depth, args.part))
        return {"status": "node_cap"} if args.part else {"seconds": 1.0}

    monkeypatch.setattr(union, "context", lambda _: (None, {}, None, phases, object()))
    monkeypatch.setattr(union, "binding", lambda *args: {})
    monkeypatch.setattr(union, "load_fields", lambda *args: ({}, {}))
    monkeypatch.setattr(union, "supports", lambda *args: ([], [], phases))
    monkeypatch.setattr(union, "WalkerObserver", FailedObserver)
    monkeypatch.setattr(
        union, "StationarySolve", lambda *a, **kw: SimpleNamespace(geometry=None, close=lambda: closed.append(1))
    )
    monkeypatch.setattr(union, "run", split)
    args = (tmp_path / "start.json", tmp_path, state, fit, "digest", {"source_revision": "fixture"}, None)
    if capacity:
        union.evaluate(*args)
        assert calls == [(rule, depth, part) for rule in ("six", "duffy") for depth in (3, 4) for part in (0, 1)]
        assert len(closed) == 4
        for rule, depth, part in calls:
            record = sampler.read_record(tmp_path / f"{rule}-{depth}-{part}.json")
            assert record == ({"status": "node_cap"} if part else {"seconds": 1.0})
    else:
        with pytest.raises(UnsupportedCurvedGeometryError, match=reason):
            union.evaluate(*args)
        assert not calls and len(closed) == 1
