from dataclasses import replace

import numpy as np
import pytest

from elisa.experimental.compact_interior import CompactDuffyRule, OpticalFaces, compact_panels
from elisa.experimental.curved_geometry import split_triangles
from elisa.experimental.gpu_curved_edges import CurvedEdgeRule


@pytest.fixture
def face():
    return np.array([[[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, 0.0, 0.2]]])


def test_coalesces_one_generation_within_the_optical_face(face):
    parents = split_triangles(face)
    children = split_triangles(parents)
    panels, owners = compact_panels(children, np.arange(16), OpticalFaces(face), 4)
    np.testing.assert_array_equal(panels, parents)
    np.testing.assert_array_equal(owners, [0, 4, 8, 12])
    # The same four panels must stay separate when each has its own optical field.
    retained, ids = compact_panels(parents, np.arange(4), OpticalFaces(parents), 4)
    np.testing.assert_array_equal(retained, parents)
    np.testing.assert_array_equal(ids, np.arange(4))


@pytest.mark.parametrize("damage", ["clipped", "shared_source", "partial"])
def test_does_not_merge_incomplete_or_clipped_quartet(face, damage):
    children = split_triangles(split_triangles(face)[:1])
    source = np.arange(4)
    if damage == "clipped":
        children[0, 0, 0] = np.nextafter(children[0, 0, 0], np.inf)
    elif damage == "shared_source":
        source[2] = source[1]
    else:
        children, source = children[1:], source[1:]
    panels, ids = compact_panels(children, source, OpticalFaces(face), 4)
    np.testing.assert_array_equal(panels, children)
    np.testing.assert_array_equal(ids, source)


def test_unbound_rule_keeps_original_duffy_nodes_and_cap(face):
    rule = CompactDuffyRule(3, 3)
    for actual, original in zip(
        rule.planar_packet(face, np.array([0]), 9),
        CurvedEdgeRule(3, 3).planar_packet(face, np.array([0]), 9),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, original)
    with pytest.raises(ValueError, match="single Duffy"):
        replace(rule, duffy_sibling_levels=2).planar_packet(face, np.array([0]), 9)
    from elisa.experimental.curved_geometry import UnsupportedCurvedGeometryError

    with pytest.raises(UnsupportedCurvedGeometryError, match="node cap"):
        rule.planar_packet(face, np.array([0]), 8)


def test_historical_checkpoint_cannot_advance_on_new_rule(tmp_path):
    from scripts import sample_reflected_binary as sampler
    from scripts.compact_interior_fit import advance

    cp = tmp_path / "state.json"
    state = dict(target={}, step=0, events=[])
    sampler.write_record(cp, state)
    with pytest.raises(ValueError, match="Historical"):
        advance(cp, tmp_path, 1)
    assert sampler.read_record(cp) == state
