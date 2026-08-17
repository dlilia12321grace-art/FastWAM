import numpy as np
import pytest

from fastwam.utils.action_jitter import compute_action_jitter_metrics


def test_action_jitter_separates_replan_boundary_and_gripper():
    actions = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    metrics = compute_action_jitter_metrics(actions, [0, 2])

    assert metrics["num_actions"] == 4
    assert metrics["num_replans"] == 2
    assert metrics["gripper_flip_count"] == 1
    assert metrics["replan_boundary_delta_l2_mean"] == pytest.approx(1.0)
    assert metrics["within_chunk_delta_l2_mean"] == pytest.approx(0.1)


def test_action_jitter_rejects_invalid_shape():
    with pytest.raises(ValueError, match="Expected actions"):
        compute_action_jitter_metrics(np.zeros((4,), dtype=np.float32), [0])
