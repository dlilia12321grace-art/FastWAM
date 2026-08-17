import json

import pytest

from scripts.summarize_dynamic_action_gap_smoke import (
    discover_methods,
    summarize_result,
)


def test_summarize_dynamic_routes():
    data = {
        "task_suite": "libero_goal",
        "successes": 4,
        "total_episodes": 5,
        "timing_profile": {
            "summary": {
                "action_denoise_total_ms_mean": 100.0,
                "infer_action_total_ms_mean": 160.0,
            },
            "episodes": [
                {
                    "chunks": [
                        {
                            "dynamic_action_gap": {
                                "steps": [
                                    {
                                        "route": "full",
                                        "reason": "first_step",
                                        "predicted_gap": 0.4,
                                    },
                                    {
                                        "route": "internal",
                                        "reason": "predicted_safe",
                                        "predicted_gap": 0.1,
                                    },
                                ]
                            }
                        }
                    ]
                }
            ],
        },
    }
    summary = summarize_result(data, "dynamic")
    assert summary["success_rate"] == 0.8
    assert summary["full_steps"] == 1
    assert summary["internal_steps"] == 1
    assert summary["internal_ratio"] == 0.5
    assert summary["predicted_gap_mean"] == 0.25
    assert summary["route_reasons"] == {
        "first_step": 1,
        "predicted_safe": 1,
    }


def test_summarize_fixed_schedule_routes():
    data = {
        "task_suite": "libero_spatial",
        "successes": 5,
        "total_episodes": 5,
        "timing_profile": {
            "summary": {},
            "episodes": [
                {
                    "chunks": [
                        {
                            "internal_lora_inference": {
                                "step_modes": ["full", "internal", "internal"]
                            }
                        }
                    ]
                }
            ],
        },
    }
    summary = summarize_result(data, "fixed_gap4")
    assert summary["full_steps"] == 1
    assert summary["internal_steps"] == 2
    assert summary["internal_ratio"] == 2 / 3


def test_discover_methods_includes_complete_v3_candidates(tmp_path):
    for method in ("fixed_gap4", "v3_safe", "v3_balanced"):
        for suite in ("libero_goal", "libero_spatial"):
            path = tmp_path / f"{method}_{suite}" / suite / "gpu0_task0_results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({}), encoding="utf-8")
    # An incomplete candidate must not appear in the comparison table.
    incomplete = (
        tmp_path
        / "v3_fast_libero_goal"
        / "libero_goal"
        / "gpu0_task0_results.json"
    )
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text(json.dumps({}), encoding="utf-8")

    assert discover_methods(tmp_path) == [
        "fixed_gap4",
        "v3_safe",
        "v3_balanced",
    ]


def test_discover_methods_requires_fixed_baseline(tmp_path):
    with pytest.raises(FileNotFoundError, match="fixed_gap4"):
        discover_methods(tmp_path)


def test_discover_methods_finds_custom_complete_candidate(tmp_path):
    for method in ("fixed_gap4", "my_threshold", "collect"):
        for suite in ("libero_goal", "libero_spatial"):
            path = tmp_path / f"{method}_{suite}" / suite / "gpu0_task0_results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({}), encoding="utf-8")

    assert discover_methods(tmp_path) == ["fixed_gap4", "my_threshold"]


def test_discover_methods_supports_other_suites(tmp_path):
    suites = ("libero_object", "libero_10")
    for method in ("fixed_gap4", "v4_t036"):
        for suite in suites:
            path = tmp_path / f"{method}_{suite}" / suite / "gpu0_task3_results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({}), encoding="utf-8")

    assert discover_methods(tmp_path, task_id=3, suites=suites) == [
        "fixed_gap4",
        "v4_t036",
    ]
