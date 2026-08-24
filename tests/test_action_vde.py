import pytest
import torch

from fastwam.models.wan22.action_vde import (
    ActionVelocityDecompositionEstimator,
    build_action_vde_schedule,
)


def test_conservative_ten_step_schedule():
    assert build_action_vde_schedule(
        10, warmup_steps=4, anchor_interval=2
    ) == [
        "full", "full", "full", "full", "estimate",
        "full", "estimate", "full", "estimate", "full",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_steps": 1, "warmup_steps": 2, "anchor_interval": 2},
        {"num_steps": 10, "warmup_steps": 1, "anchor_interval": 2},
        {"num_steps": 10, "warmup_steps": 4, "anchor_interval": 1},
    ],
)
def test_schedule_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        build_action_vde_schedule(**kwargs)


def test_estimator_preserves_action_shape_dtype_and_device():
    estimator = ActionVelocityDecompositionEstimator()
    sample = torch.randn(1, 30, 7, dtype=torch.float16)
    estimator.add_anchor(sample, sample * 0.3 + 0.2, 900.0)
    estimator.add_anchor(sample * 0.9, sample * 0.4 + 0.25, 700.0)
    estimate = estimator.estimate(sample * 0.8, 500.0)
    assert estimate is not None
    assert estimate.shape == sample.shape
    assert estimate.dtype == sample.dtype
    assert estimate.device == sample.device
    assert torch.isfinite(estimate).all()


def test_estimator_falls_back_until_two_anchors_exist():
    estimator = ActionVelocityDecompositionEstimator()
    sample = torch.randn(1, 4, 3)
    assert estimator.estimate(sample, 500.0) is None
    estimator.add_anchor(sample, sample + 1.0, 900.0)
    assert estimator.estimate(sample, 500.0) is None
