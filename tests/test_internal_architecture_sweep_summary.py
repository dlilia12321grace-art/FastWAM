import json

import torch.nn as nn

from scripts.summarize_internal_architecture_sweep import build_row, method_name
from fastwam.models.wan22.internal_action_branch import DeepCopyInternalActionBranch


def _write_result(path, *, suite, successes, total, denoise, infer):
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for _ in range(total):
        chunks.append({
            "action_denoise_total_ms": denoise,
            "infer_action_total_ms": infer,
            "internal_lora_inference": {
                "step_modes": [
                    "full", "internal", "internal", "full", "internal",
                    "internal", "internal", "full", "internal", "full",
                ]
            },
        })
    payload = {
        "task_suite": suite,
        "successes": successes,
        "total_episodes": total,
        "success_episodes": list(range(successes)),
        "timing_profile": {
            "summary": {
                "action_denoise_total_ms_mean": denoise,
                "infer_action_total_ms_mean": infer,
            },
            "episodes": [
                {"episode_index": i, "chunks": [chunks[i]]}
                for i in range(total)
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_row_combines_training_audit_and_suite_results(tmp_path):
    assert method_name(4, 1) == "fork4_b1"
    train_path = (
        tmp_path / "train/fork4_b1/libero_spatial/gpu0_task0_results.json"
    )
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text(json.dumps({
        "duration": 12.5,
        "internal_lora_training": {
            "updates": 20,
            "best_loss": 0.01,
            "best_update": 8,
            "audit": {
                "all_parameters": 1000,
                "trainable_parameters": 100,
                "trainable_fraction": 0.1,
            },
        },
    }), encoding="utf-8")
    for suite, successes in (("libero_goal", 4), ("libero_spatial", 5)):
        _write_result(
            tmp_path / f"eval/fork4_b1_{suite}/{suite}/gpu0_task0_results.json",
            suite=suite,
            successes=successes,
            total=5,
            denoise=100.0,
            infer=150.0,
        )

    row = build_row(
        tmp_path, 4, 1, ["libero_goal", "libero_spatial"], 0
    )
    assert row["successes"] == 9
    assert row["total_episodes"] == 10
    assert row["internal_ratio"] == 0.6
    assert row["denoise_ms"] == 100.0
    assert row["trainable_parameters"] == 100


def test_zero_block_branch_trains_only_copied_head():
    class ActionExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
            self.head = nn.Linear(8, 4)

    branch = DeepCopyInternalActionBranch(
        ActionExpert(),
        fork_layer=1,
        source_start_layer=2,
        source_end_layer=1,
    )
    assert len(branch.blocks) == 0
    assert branch.source_start_layer == 2
    assert branch.source_end_layer == 1
    assert branch.trainable_parameter_names() == ["head.weight", "head.bias"]
    assert "source_layers=none" in branch.extra_repr()
