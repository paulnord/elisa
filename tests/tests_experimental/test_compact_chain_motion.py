import pytest

from scripts.compact_chain_motion import summarize


def test_movement_keeps_repeated_positions_and_counts_both_directions():
    result = summarize([[{"radius": x} for x in path] for path in ([2, 4, 3, 3], [2, 2, 2])])["radius"]
    assert result == dict(
        median_within_chain_range=1,
        max_within_chain_range=2,
        streams_with_movement=1,
        streams_moving_both_directions=1,
        rms_selected_jump=pytest.approx(1),
        selected_endpoint_range=[2, 3],
    )
