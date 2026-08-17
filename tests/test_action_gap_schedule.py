import pytest

from fastwam.models.wan22.internal_action_branch import build_action_gap_schedule


@pytest.mark.parametrize(
    ("num_steps", "gap", "expected"),
    [
        (10, 2, ["full", "full", "internal", "full", "internal", "full", "internal", "full", "internal", "full"]),
        (10, 4, ["full", "internal", "internal", "full", "internal", "internal", "internal", "full", "internal", "full"]),
        (1, 2, ["full"]),
        (2, 4, ["full", "full"]),
    ],
)
def test_action_gap_schedule(num_steps, gap, expected):
    assert build_action_gap_schedule(num_steps, gap) == expected


@pytest.mark.parametrize(("num_steps", "gap"), [(0, 2), (10, 0), (-1, 2), (10, -1)])
def test_action_gap_schedule_rejects_invalid_values(num_steps, gap):
    with pytest.raises(ValueError):
        build_action_gap_schedule(num_steps, gap)
