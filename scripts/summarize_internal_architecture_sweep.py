#!/usr/bin/env python
"""Summarize staged fork-layer / internal-block architecture screening."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_dynamic_action_gap_smoke import summarize_result
except ModuleNotFoundError:
    from summarize_dynamic_action_gap_smoke import summarize_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--fork-layers", type=int, nargs="+", required=True)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--block-counts", type=int, nargs="+", default=None)
    parser.add_argument(
        "--suites", nargs="+", default=["libero_goal", "libero_spatial"]
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def method_name(fork_layer: int, num_blocks: int) -> str:
    return f"fork{fork_layer}_b{num_blocks}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def build_row(
    output_root: Path,
    fork_layer: int,
    num_blocks: int,
    suites: list[str],
    task_id: int,
) -> dict[str, Any]:
    method = method_name(fork_layer, num_blocks)
    train_path = (
        output_root / "train" / method / "libero_spatial"
        / f"gpu0_task{task_id}_results.json"
    )
    train_payload = load_json(train_path)
    training = train_payload["internal_lora_training"]
    audit = training["audit"]

    suite_rows = []
    for suite in suites:
        eval_path = (
            output_root / "eval" / f"{method}_{suite}" / suite
            / f"gpu0_task{task_id}_results.json"
        )
        summary = summarize_result(load_json(eval_path), method)
        summary["suite"] = suite
        suite_rows.append(summary)

    total_successes = sum(int(row["successes"]) for row in suite_rows)
    total_episodes = sum(int(row["total_episodes"]) for row in suite_rows)
    full_steps = sum(int(row["full_steps"]) for row in suite_rows)
    internal_steps = sum(int(row["internal_steps"]) for row in suite_rows)
    return {
        "method": method,
        "fork_layer": int(fork_layer),
        "num_blocks": int(num_blocks),
        "all_parameters": int(audit["all_parameters"]),
        "trainable_parameters": int(audit["trainable_parameters"]),
        "trainable_fraction": float(audit["trainable_fraction"]),
        "training_duration_s": float(train_payload["duration"]),
        "training_updates": int(training["updates"]),
        "best_loss": training["best_loss"],
        "best_update": training["best_update"],
        "successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": total_successes / total_episodes,
        "denoise_ms": mean(
            [row["action_denoise_total_ms_mean"] for row in suite_rows]
        ),
        "infer_ms": mean([row["infer_action_total_ms_mean"] for row in suite_rows]),
        "internal_ratio": (
            internal_steps / (full_steps + internal_steps)
            if full_steps + internal_steps
            else None
        ),
        "suites": suite_rows,
    }


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    block_counts = (
        list(args.block_counts)
        if args.block_counts is not None
        else [int(args.num_blocks)]
    )
    rows = [
        build_row(
            args.output_root,
            fork_layer,
            num_blocks,
            list(args.suites),
            args.task_id,
        )
        for num_blocks in block_counts
        for fork_layer in args.fork_layers
    ]

    print(
        "| Method | Params | Trainable | Train s | Best loss | Success | "
        "Denoise ms | Infer ms | Internal ratio |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['method']} | {row['all_parameters']:,} | "
            f"{row['trainable_parameters']:,} | {row['training_duration_s']:.1f} | "
            f"{fmt(row['best_loss'], 6)} | "
            f"{row['successes']}/{row['total_episodes']} | "
            f"{fmt(row['denoise_ms'])} | {fmt(row['infer_ms'])} | "
            f"{fmt(None if row['internal_ratio'] is None else 100 * row['internal_ratio'], 1)}% |"
        )

    payload = {
        "fork_layers": list(args.fork_layers),
        "block_counts": block_counts,
        "task_id": int(args.task_id),
        "suites": list(args.suites),
        "rows": rows,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "method", "fork_layer", "num_blocks", "all_parameters",
            "trainable_parameters", "trainable_fraction", "training_duration_s",
            "training_updates", "best_loss", "best_update", "successes",
            "total_episodes", "success_rate", "denoise_ms", "infer_ms",
            "internal_ratio",
        ]
        with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fieldnames})


if __name__ == "__main__":
    main()
