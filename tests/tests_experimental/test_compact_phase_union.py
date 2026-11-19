from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import compact_phase_union as union


@dataclass
class Rule:
    depth: int = 3


def test_union_preserves_each_block_quadrature_digest(monkeypatch):
    solver = SimpleNamespace(integration_level=3, rule=Rule())
    whole = [0.0, 0.1, 0.5]
    subsets = [[0.0, 0.1], [0.0, 0.5]]
    hashes = union.BlockDigests(subsets, whole, solver)
    header = json.dumps({"level": 3, "rule": {"depth": 3}}).encode()
    expected = [hashlib.sha256(header), hashlib.sha256(header)]
    packet = {"mu": np.arange(4.0)[::2], "area": np.ones(2), "parents": np.arange(2)}
    monkeypatch.setattr(union, "compact_shadow_packet", lambda *args: packet)
    for phase in whole:
        for name in ("primary", "secondary"):
            hashes.packet(None, SimpleNamespace(component=name), None, solver.rule)
            for digest, subset in zip(expected, subsets, strict=True):
                if phase in subset:
                    digest.update(json.dumps({"phase": phase, "component": name, "nodes": 2}).encode())
                    for key in ("mu", "area", "parents"):
                        digest.update(packet[key].tobytes())
    assert hashes.finish() == [d.hexdigest() for d in expected]


def test_union_detects_missing_component():
    solver = SimpleNamespace(integration_level=3, rule=Rule())
    h = union.BlockDigests([[0.0], [0.0]], [0.0], solver)
    with pytest.raises(ArithmeticError, match="all phases"):
        h.finish()


def test_shared_observer_time_is_counted_once_by_the_original_join():
    blocks = [{"stages": {"geometry": 3.0}, "batch": {"total": 5.0}} for _ in range(2)]
    for part, block in enumerate(blocks):
        union.account_union_time(block, part, 5.0)
    assert sum(b["inner_seconds"] for b in blocks) == 5.0
    assert sum(b["stages"].get("geometry", 0.0) for b in blocks) == 3.0
