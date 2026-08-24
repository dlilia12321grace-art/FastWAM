import json

from scripts.summarize_video_gap import aggregate, summarize_result


def test_video_gap_summary_counts_refreshes_and_timing(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "successes": 1,
                "success_episodes": [0],
                "failure_episodes": [1],
                "timing_profile": {
                    "episodes": [
                        {
                            "episode_index": 0,
                            "chunks": [
                                {
                                    "image_encode_ms": 10.0,
                                    "video_pre_dit_ms": 1.0,
                                    "video_kv_prefill_ms": 30.0,
                                    "action_denoise_total_ms": 100.0,
                                    "infer_action_total_ms": 150.0,
                                    "video_gap": {
                                        "video_gap": 2,
                                        "refresh_cache": True,
                                    },
                                },
                                {
                                    "image_encode_ms": 0.0,
                                    "video_pre_dit_ms": 0.0,
                                    "video_kv_prefill_ms": 0.0,
                                    "action_denoise_total_ms": 100.0,
                                    "infer_action_total_ms": 109.0,
                                    "video_gap": {
                                        "video_gap": 2,
                                        "refresh_cache": False,
                                    },
                                },
                            ],
                        }
                    ]
                },
            }
        )
    )

    row = summarize_result(path)
    summary = aggregate([row])
    assert summary["successes"] == 1
    assert summary["episodes"] == 2
    assert summary["refresh_ratio"] == 0.5
    assert summary["video_kv_prefill_ms"] == 15.0
    assert summary["infer_action_ms"] == 129.5
