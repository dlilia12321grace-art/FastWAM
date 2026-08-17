from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def compute_action_jitter_metrics(
    actions: np.ndarray,
    replan_start_indices: Iterable[int],
) -> dict[str, Any]:
    """Measure action discontinuity, excluding the discrete gripper from L2 norms."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 2:
        raise ValueError(f"Expected actions [T,D] with D >= 2, got {actions.shape}.")
    num_actions = int(actions.shape[0])
    arm = actions[:, :-1]
    translation = arm[:, : min(3, arm.shape[1])]
    rotation = arm[:, 3: min(6, arm.shape[1])]
    delta_arm = np.linalg.norm(np.diff(arm, axis=0), axis=1)
    delta_translation = np.linalg.norm(np.diff(translation, axis=0), axis=1)
    delta_rotation = (
        np.linalg.norm(np.diff(rotation, axis=0), axis=1)
        if rotation.shape[1]
        else np.zeros_like(delta_arm)
    )
    jerk_arm = np.linalg.norm(np.diff(arm, n=2, axis=0), axis=1)

    starts = sorted({int(index) for index in replan_start_indices})
    boundary_delta_indices = np.asarray(
        [index - 1 for index in starts if 0 < index < num_actions], dtype=np.int64
    )
    boundary_mask = np.zeros(delta_arm.shape[0], dtype=bool)
    if boundary_delta_indices.size:
        boundary_mask[boundary_delta_indices] = True

    gripper = actions[:, -1]
    gripper_flips = int(np.count_nonzero(np.diff(np.sign(gripper))))
    metrics: dict[str, Any] = {
        "num_actions": num_actions,
        "num_replans": len(starts),
        "gripper_flip_count": gripper_flips,
    }
    metrics.update(_stats(delta_arm, "arm_delta_l2"))
    metrics.update(_stats(delta_translation, "translation_delta_l2"))
    metrics.update(_stats(delta_rotation, "rotation_delta_l2"))
    metrics.update(_stats(jerk_arm, "arm_jerk_l2"))
    metrics.update(_stats(delta_arm[boundary_mask], "replan_boundary_delta_l2"))
    metrics.update(_stats(delta_arm[~boundary_mask], "within_chunk_delta_l2"))
    return metrics
