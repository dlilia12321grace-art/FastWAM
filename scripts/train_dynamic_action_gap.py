#!/usr/bin/env python
"""Train the lightweight Dynamic ActionGap risk predictor from collected rollouts."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fastwam.models.wan22.dynamic_action_gap import (
    DynamicActionGapGate,
    make_dynamic_action_gap_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument(
        "--input-mode",
        choices=["hidden_meta", "meta_only", "hidden_only"],
        default="hidden_meta",
    )
    parser.add_argument("--max-internal-run", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--target-internal-rate", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_samples(
    paths: list[Path], max_internal_run: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], int]:
    hidden_rows = []
    meta_rows = []
    targets = []
    groups = []
    hidden_dim = None
    for source_index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload.get("format_version", -1)) != 1:
            raise ValueError(f"Unsupported dataset format: {path}")
        payload_hidden_dim = int(payload["hidden_dim"])
        if hidden_dim is None:
            hidden_dim = payload_hidden_dim
        elif hidden_dim != payload_hidden_dim:
            raise ValueError("All datasets must use the same Action DiT hidden size.")
        for sample in payload["samples"]:
            pooled = sample["pooled_hidden"].float().reshape(-1)
            if pooled.numel() != hidden_dim:
                raise ValueError(f"Invalid pooled hidden shape in {path}.")
            step_index = int(sample["step_index"])
            num_steps = int(sample["num_steps"])
            timestep = float(sample["timestep"])
            target = math.log1p(float(sample["gap"]))
            # Counter augmentation prevents the predictor from tying a visual state
            # to only one hand-authored routing history.
            for steps_since_full in range(max_internal_run):
                hidden_rows.append(pooled)
                meta_rows.append(torch.tensor([
                    timestep / 1000.0,
                    step_index / max(num_steps - 1, 1),
                    steps_since_full / max_internal_run,
                ]))
                targets.append(target)
                groups.append(
                    f"{source_index}:{int(sample['chunk_index'])}"
                )
    if not hidden_rows:
        raise ValueError("No Dynamic ActionGap samples were found.")
    return (
        torch.stack(hidden_rows),
        torch.stack(meta_rows),
        torch.tensor(targets, dtype=torch.float32),
        groups,
        int(hidden_dim),
    )


def split_by_chunk(
    groups: list[str], validation_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("`validation_fraction` must be between zero and one.")
    unique_groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    if len(unique_groups) >= 2:
        num_validation = max(1, round(len(unique_groups) * validation_fraction))
        validation_groups = set(unique_groups[:num_validation])
        validation_mask = torch.tensor(
            [group in validation_groups for group in groups], dtype=torch.bool
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        validation_mask = torch.rand(len(groups), generator=generator) < validation_fraction
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("Unable to create non-empty train and validation splits.")
    indices = torch.arange(len(groups))
    return indices[~validation_mask], indices[validation_mask]


@torch.no_grad()
def evaluate(
    gate: DynamicActionGapGate,
    hidden: torch.Tensor,
    meta: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    gate.eval()
    predictions = []
    loader = DataLoader(TensorDataset(hidden, meta), batch_size=batch_size)
    for hidden_batch, meta_batch in loader:
        predictions.append(
            gate(hidden_batch.to(device), meta_batch.to(device)).float().cpu()
        )
    prediction = torch.cat(predictions)
    metrics = {
        "smooth_l1": float(F.smooth_l1_loss(prediction, target)),
        "mae": float((prediction - target).abs().mean()),
    }
    return prediction, metrics


def main() -> None:
    args = parse_args()
    if args.max_internal_run <= 0:
        raise ValueError("`max_internal_run` must be positive.")
    if not 0.0 < args.target_internal_rate < 1.0:
        raise ValueError("`target_internal_rate` must be between zero and one.")
    torch.manual_seed(args.seed)
    hidden, meta, target, groups, hidden_dim = load_samples(
        args.datasets, args.max_internal_run
    )
    train_indices, validation_indices = split_by_chunk(
        groups, args.validation_fraction, args.seed
    )
    device = torch.device(args.device)
    gate = DynamicActionGapGate(
        hidden_dim=hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        meta_dim=meta.shape[1],
        input_mode=args.input_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        TensorDataset(
            hidden[train_indices], meta[train_indices], target[train_indices]
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    best_state = None
    best_validation = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        gate.train()
        train_loss_sum = 0.0
        train_count = 0
        for hidden_batch, meta_batch, target_batch in train_loader:
            hidden_batch = hidden_batch.to(device)
            meta_batch = meta_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = gate(hidden_batch, meta_batch)
            loss = F.smooth_l1_loss(prediction.float(), target_batch.float())
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * hidden_batch.shape[0]
            train_count += hidden_batch.shape[0]
        validation_prediction, validation_metrics = evaluate(
            gate,
            hidden[validation_indices],
            meta[validation_indices],
            target[validation_indices],
            args.batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "train_smooth_l1": train_loss_sum / max(train_count, 1),
            **{f"validation_{k}": v for k, v in validation_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))
        if validation_metrics["smooth_l1"] < best_validation:
            best_validation = validation_metrics["smooth_l1"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in gate.state_dict().items()
            }
    gate.load_state_dict(best_state, strict=True)
    validation_prediction, validation_metrics = evaluate(
        gate,
        hidden[validation_indices],
        meta[validation_indices],
        target[validation_indices],
        args.batch_size,
        device,
    )
    threshold = float(
        torch.quantile(validation_prediction, args.target_internal_rate)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        make_dynamic_action_gap_checkpoint(
            gate,
            default_threshold=threshold,
            extra={
                "datasets": [str(path) for path in args.datasets],
                "input_mode": args.input_mode,
                "max_internal_run": args.max_internal_run,
                "target_internal_rate": args.target_internal_rate,
                "train_samples": int(train_indices.numel()),
                "validation_samples": int(validation_indices.numel()),
                "validation_metrics": validation_metrics,
                "history": history,
            },
        ),
        args.output,
    )
    print(f"Saved Dynamic ActionGap gate to {args.output}")
    print(f"Default threshold: {threshold:.8f}")


if __name__ == "__main__":
    main()
