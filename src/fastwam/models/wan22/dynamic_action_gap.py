from __future__ import annotations

import math
import random
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


Route = Literal["full", "internal"]
InputMode = Literal["hidden_meta", "meta_only", "hidden_only"]
ComputeMatchedMode = Literal["fixed", "random"]


class DynamicActionGapGate(nn.Module):
    """Predict the log-normalized error of the cheap internal action branch."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int = 256,
        meta_dim: int = 3,
        input_mode: InputMode = "hidden_meta",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0 or meta_dim <= 0:
            raise ValueError("Gate dimensions must all be positive.")
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.meta_dim = int(meta_dim)
        if input_mode not in {"hidden_meta", "meta_only", "hidden_only"}:
            raise ValueError(f"Unsupported Dynamic ActionGap input mode: {input_mode}.")
        self.input_mode: InputMode = input_mode
        self.hidden_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.bottleneck_dim + self.meta_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, pooled_hidden: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if pooled_hidden.ndim != 2 or pooled_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                "`pooled_hidden` must have shape [B, hidden_dim], got "
                f"{tuple(pooled_hidden.shape)}."
            )
        if meta.ndim != 2 or meta.shape[0] != pooled_hidden.shape[0]:
            raise ValueError(
                "`meta` must have shape [B, meta_dim] with the same batch size, got "
                f"{tuple(meta.shape)}."
            )
        if meta.shape[-1] != self.meta_dim:
            raise ValueError(
                f"Expected {self.meta_dim} metadata features, got {meta.shape[-1]}."
            )
        if self.input_mode == "meta_only":
            hidden_features = pooled_hidden.new_zeros(
                (pooled_hidden.shape[0], self.bottleneck_dim)
            )
        else:
            hidden_features = self.hidden_proj(pooled_hidden)
        if self.input_mode == "hidden_only":
            meta = torch.zeros_like(meta)
        features = torch.cat([hidden_features, meta], dim=-1)
        # The target is log1p(normalized_gap), so a non-negative output is natural.
        return F.softplus(self.head(features)).squeeze(-1)


def build_dynamic_action_gap_meta(
    *,
    batch_size: int,
    timestep: float,
    step_index: int,
    num_steps: int,
    steps_since_full: int,
    max_internal_run: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the three cheap scalar features used by the MVP gate."""
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
    if num_steps <= 0:
        raise ValueError(f"`num_steps` must be positive, got {num_steps}.")
    if not 0 <= step_index < num_steps:
        raise ValueError(f"Invalid step index {step_index} for {num_steps} steps.")
    if max_internal_run <= 0:
        raise ValueError(
            f"`max_internal_run` must be positive, got {max_internal_run}."
        )
    values = torch.tensor(
        [
            float(timestep) / 1000.0,
            float(step_index) / float(max(num_steps - 1, 1)),
            float(steps_since_full) / float(max_internal_run),
        ],
        device=device,
        dtype=dtype,
    )
    return values.unsqueeze(0).expand(batch_size, -1)


def select_dynamic_action_gap_route(
    *,
    predicted_gap: float,
    threshold: float,
    step_index: int,
    num_steps: int,
    steps_since_full: int,
    max_internal_run: int,
    anchor_action_gap: Optional[int] = None,
) -> tuple[Route, str]:
    """Apply learned routing with deterministic safety constraints."""
    if num_steps <= 0:
        raise ValueError(f"`num_steps` must be positive, got {num_steps}.")
    if not 0 <= step_index < num_steps:
        raise ValueError(f"Invalid step index {step_index} for {num_steps} steps.")
    if max_internal_run <= 0:
        raise ValueError(
            f"`max_internal_run` must be positive, got {max_internal_run}."
        )
    if anchor_action_gap is not None and anchor_action_gap <= 0:
        raise ValueError(
            f"`anchor_action_gap` must be positive when provided, got {anchor_action_gap}."
        )
    if step_index == 0:
        return "full", "first_step"
    if step_index == num_steps - 1:
        return "full", "last_step"
    if anchor_action_gap is not None and (step_index + 1) % anchor_action_gap == 0:
        return "full", "anchor_schedule"
    if steps_since_full >= max_internal_run:
        return "full", "max_internal_run"
    if predicted_gap <= threshold:
        return "internal", "predicted_safe"
    return "full", "predicted_risky"


def build_compute_matched_action_gap_schedule(
    *,
    num_steps: int,
    chunk_index: int,
    target_internal_ratio: float = 0.65,
    mode: ComputeMatchedMode = "fixed",
    seed: int = 0,
    max_internal_run: int = 7,
    anchor_action_gap: Optional[int] = 8,
) -> list[Route]:
    """Build a state-independent schedule with a matched long-run compute budget.

    Fractional targets are error-diffused across chunks. For example, ten denoise
    steps and a 0.65 target alternate between six and seven internal steps.
    Safety boundaries are applied before the remaining full steps are selected.
    """
    if num_steps <= 0:
        raise ValueError(f"`num_steps` must be positive, got {num_steps}.")
    if chunk_index < 0:
        raise ValueError(f"`chunk_index` must be non-negative, got {chunk_index}.")
    if not 0.0 <= target_internal_ratio <= 1.0:
        raise ValueError(
            "`target_internal_ratio` must be between zero and one, got "
            f"{target_internal_ratio}."
        )
    if mode not in {"fixed", "random"}:
        raise ValueError(f"Unsupported compute-matched routing mode: {mode}.")
    if max_internal_run <= 0:
        raise ValueError(
            f"`max_internal_run` must be positive, got {max_internal_run}."
        )
    if anchor_action_gap is not None and anchor_action_gap <= 0:
        raise ValueError(
            f"`anchor_action_gap` must be positive when provided, got {anchor_action_gap}."
        )

    full_steps = {0, num_steps - 1}
    if anchor_action_gap is not None:
        full_steps.update(
            step
            for step in range(num_steps)
            if (step + 1) % anchor_action_gap == 0
        )

    # Insert the minimum number of safety steps needed to bound internal runs.
    while True:
        ordered = sorted(full_steps)
        unsafe_gap = next(
            (
                (left, right)
                for left, right in zip(ordered, ordered[1:])
                if right - left - 1 > max_internal_run
            ),
            None,
        )
        if unsafe_gap is None:
            break
        left, _ = unsafe_gap
        full_steps.add(left + max_internal_run + 1)

    raw_target = float(target_internal_ratio) * num_steps
    target_internal_steps = (
        math.floor((chunk_index + 1) * raw_target + 1e-9)
        - math.floor(chunk_index * raw_target + 1e-9)
    )
    maximum_internal_steps = num_steps - len(full_steps)
    target_internal_steps = min(
        max(int(target_internal_steps), 0), maximum_internal_steps
    )
    extra_full_steps = maximum_internal_steps - target_internal_steps

    if mode == "random":
        candidates = [step for step in range(num_steps) if step not in full_steps]
        rng = random.Random(int(seed) + 1_000_003 * int(chunk_index) + 9_176 * num_steps)
        rng.shuffle(candidates)
        full_steps.update(candidates[:extra_full_steps])
    else:
        # Repeatedly split the longest internal run, producing a deterministic
        # state-independent schedule without favoring an endpoint.
        for _ in range(extra_full_steps):
            ordered = sorted(full_steps)
            runs = [
                list(range(left + 1, right))
                for left, right in zip(ordered, ordered[1:])
                if right - left > 1
            ]
            longest = max(runs, key=lambda run: (len(run), -run[0]))
            full_steps.add(longest[(len(longest) - 1) // 2])

    return ["full" if step in full_steps else "internal" for step in range(num_steps)]


def make_dynamic_action_gap_checkpoint(
    gate: DynamicActionGapGate,
    *,
    default_threshold: float,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "hidden_dim": gate.hidden_dim,
        "bottleneck_dim": gate.bottleneck_dim,
        "meta_dim": gate.meta_dim,
        "input_mode": gate.input_mode,
        "default_threshold": float(default_threshold),
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in gate.state_dict().items()
        },
        "extra": {} if extra is None else dict(extra),
    }


def load_dynamic_action_gap_gate(
    checkpoint_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[DynamicActionGapGate, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(
            f"Unsupported Dynamic ActionGap checkpoint: {checkpoint_path}."
        )
    gate = DynamicActionGapGate(
        hidden_dim=int(payload["hidden_dim"]),
        bottleneck_dim=int(payload["bottleneck_dim"]),
        meta_dim=int(payload["meta_dim"]),
        input_mode=str(payload.get("input_mode", "hidden_meta")),
    )
    gate.load_state_dict(payload["state_dict"], strict=True)
    gate = gate.to(device=device, dtype=dtype).eval()
    return gate, payload
