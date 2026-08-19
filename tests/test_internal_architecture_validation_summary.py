import json

from scripts.summarize_internal_architecture_validation import (
    build_summary,
    parse_candidate,
)


def _write_train(root, method, *, params, trainable, loss):
    path = root / "train" / method / "libero_spatial" / "gpu0_task0_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "duration": 12.0,
        "internal_lora_training": {
            "updates": 100,
            "best_loss": loss,
            "best_update": 50,
            "audit": {
                "all_parameters": params,
                "trainable_parameters": trainable,
                "trainable_fraction": trainable / params,
            },
        },
    }), encoding="utf-8")


def _write_eval(root, tag, task, method, suite, successes):
    path = (
        root / tag / f"task{task}" / f"{method}_{suite}" / suite
        / f"gpu0_task{task}_results.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = []
    for index in range(5):
        episodes.append({
            "episode_index": index,
            "chunks": [{
                "internal_lora_inference": {
                    "step_modes": ["full", "internal", "internal", "full"]
                }
            }],
        })
    path.write_text(json.dumps({
        "task_suite": suite,
        "successes": successes,
        "total_episodes": 5,
        "success_episodes": list(range(successes)),
        "failure_episodes": list(range(successes, 5)),
        "timing_profile": {
            "summary": {
                "action_denoise_total_ms_mean": 100.0,
                "infer_action_total_ms_mean": 150.0,
            },
            "episodes": episodes,
        },
    }), encoding="utf-8")


def test_parse_candidate():
    assert parse_candidate("2:0") == (2, 0, "fork2_b0")
    assert parse_candidate("16:4") == (16, 4, "fork16_b4")


def test_build_summary_aggregates_and_pairs(tmp_path):
    for method, loss in (("fork2_b0", 0.2), ("fork4_b1", 0.1)):
        _write_train(tmp_path, method, params=100, trainable=10, loss=loss)
    for suite in ("libero_goal", "libero_spatial"):
        _write_eval(tmp_path, "pilot", 1, "fork2_b0", suite, 4)
        _write_eval(tmp_path, "pilot", 1, "fork4_b1", suite, 5)

    payload = build_summary(
        tmp_path,
        "pilot",
        ["2:0", "4:1"],
        "4:1",
        [1],
        ["libero_goal", "libero_spatial"],
    )
    candidate = next(
        row for row in payload["aggregates"] if row["method"] == "fork2_b0"
    )
    assert candidate["successes"] == 8
    assert candidate["total_episodes"] == 10
    assert candidate["internal_ratio"] == 0.5
    assert len(payload["paired_outcomes"]) == 2
    assert payload["paired_outcomes"][0]["reference_only"] == [4]
