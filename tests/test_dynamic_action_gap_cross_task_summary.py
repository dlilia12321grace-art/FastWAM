import pytest

from scripts.summarize_dynamic_action_gap_cross_task import aggregate_rows


def test_aggregate_rows_weights_episodes_and_counts_routes():
    rows = [
        {
            "method": "v4_t036",
            "successes": 9,
            "total_episodes": 10,
            "action_denoise_total_ms_mean": 100.0,
            "infer_action_total_ms_mean": 150.0,
            "full_steps": 30,
            "internal_steps": 70,
        },
        {
            "method": "v4_t036",
            "successes": 20,
            "total_episodes": 20,
            "action_denoise_total_ms_mean": 130.0,
            "infer_action_total_ms_mean": 180.0,
            "full_steps": 80,
            "internal_steps": 120,
        },
    ]

    result = aggregate_rows(rows, "v4_t036")

    assert result["successes"] == 29
    assert result["total_episodes"] == 30
    assert result["success_rate"] == pytest.approx(29 / 30)
    assert result["action_denoise_total_ms_mean"] == pytest.approx(120.0)
    assert result["infer_action_total_ms_mean"] == pytest.approx(170.0)
    assert result["internal_ratio"] == pytest.approx(190 / 300)


def test_aggregate_rows_rejects_missing_method():
    with pytest.raises(ValueError, match="No rows"):
        aggregate_rows([], "v4_t036")
