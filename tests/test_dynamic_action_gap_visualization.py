from scripts.visualize_dynamic_action_gap import extract_dynamic_rows


def test_extract_dynamic_rows_preserves_episode_outcome():
    data = {
        "task_suite": "libero_goal",
        "task_id": 2,
        "failure_episodes": [1],
        "timing_profile": {
            "episodes": [
                {
                    "episode_index": 1,
                    "chunks": [
                        {
                            "dynamic_action_gap": {
                                "steps": [
                                    {
                                        "step": 3,
                                        "predicted_gap": 0.4,
                                        "threshold": 0.36,
                                        "route": "full",
                                        "reason": "predicted_risky",
                                    }
                                ]
                            }
                        }
                    ],
                }
            ]
        },
    }

    rows = extract_dynamic_rows(data)

    assert rows == [
        {
            "suite": "libero_goal",
            "task_id": 2,
            "episode_index": 1,
            "success": False,
            "chunk_index": 0,
            "step_index": 3,
            "predicted_gap": 0.4,
            "threshold": 0.36,
            "route": "full",
            "reason": "predicted_risky",
        }
    ]
