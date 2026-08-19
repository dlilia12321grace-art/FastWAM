#!/usr/bin/env python
"""Summarize staged closed-loop validation of selected internal architectures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_dynamic_action_gap_cross_task import aggregate_rows
    from scripts.summarize_dynamic_action_gap_smoke import load_result, summarize_result
except ModuleNotFoundError:
    from summarize_dynamic_action_gap_cross_task import aggregate_rows
    from summarize_dynamic_action_gap_smoke import load_result, summarize_result


def parse_candidate(value: str) -> tuple[int, int, str]:
    try:
        fork_text, blocks_text = value.split(":", maxsplit=1)
        fork_layer, num_blocks = int(fork_text), int(blocks_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid candidate {value!r}; expected FORK:BLOCKS."
        ) from exc
    if fork_layer < 1 or num_blocks < 0:
        raise argparse.ArgumentTypeError(
            f"Invalid candidate {value!r}; values must be non-negative and fork >= 1."
        )
    return fork_layer, num_blocks, f"fork{fork_layer}_b{num_blocks}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--validation-tag", default="pilot")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--reference-candidate", default="4:1")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--train-task-id", type=int, default=0)
    parser.add_argument("--train-suite", default="libero_spatial")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def training_summary(
    output_root: Path,
    method: str,
    train_suite: str,
    train_task_id: int,
) -> dict[str, Any]:
    path = (
        output_root / "train" / method / train_suite
        / f"gpu0_task{train_task_id}_results.json"
    )
    payload = load_result(path)
    training = payload["internal_lora_training"]
    audit = training["audit"]
    checkpoint = output_root / "checkpoints" / f"{method}.best.pt"
    return {
        "training_duration_s": float(payload["duration"]),
        "training_updates": int(training["updates"]),
        "best_loss": training.get("best_loss"),
        "best_update": training.get("best_update"),
        "all_parameters": int(audit["all_parameters"]),
        "trainable_parameters": int(audit["trainable_parameters"]),
        "trainable_fraction": float(audit["trainable_fraction"]),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
    }


def validation_result_path(
    output_root: Path,
    validation_tag: str,
    task_id: int,
    method: str,
    suite: str,
) -> Path:
    return (
        output_root / validation_tag / f"task{task_id}"
        / f"{method}_{suite}" / suite / f"gpu0_task{task_id}_results.json"
    )


def build_summary(
    output_root: Path,
    validation_tag: str,
    candidates: list[str],
    reference_candidate: str,
    task_ids: list[int],
    suites: list[str],
    train_task_id: int = 0,
    train_suite: str = "libero_spatial",
) -> dict[str, Any]:
    parsed = [parse_candidate(candidate) for candidate in candidates]
    reference_method = parse_candidate(reference_candidate)[2]
    methods = [item[2] for item in parsed]
    if reference_method not in methods:
        raise ValueError(f"Reference method {reference_method} is not in candidates.")

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        for suite in suites:
            for _, _, method in parsed:
                row = summarize_result(
                    load_result(validation_result_path(
                        output_root, validation_tag, task_id, method, suite
                    )),
                    method,
                )
                row["task_id"] = int(task_id)
                rows.append(row)

    aggregates = []
    for fork_layer, num_blocks, method in parsed:
        aggregate = aggregate_rows(rows, method)
        aggregate.update({
            "fork_layer": fork_layer,
            "num_blocks": num_blocks,
            **training_summary(
                output_root, method, train_suite, train_task_id
            ),
        })
        aggregates.append(aggregate)

    paired_outcomes = []
    for task_id in task_ids:
        for suite in suites:
            reference = next(
                row for row in rows
                if row["task_id"] == task_id
                and row["suite"] == suite
                and row["method"] == reference_method
            )
            reference_success = set(reference["success_episodes"])
            all_episodes = set(range(int(reference["total_episodes"])))
            for method in methods:
                if method == reference_method:
                    continue
                candidate = next(
                    row for row in rows
                    if row["task_id"] == task_id
                    and row["suite"] == suite
                    and row["method"] == method
                )
                candidate_success = set(candidate["success_episodes"])
                paired_outcomes.append({
                    "task_id": task_id,
                    "suite": suite,
                    "method": method,
                    "reference_method": reference_method,
                    "reference_only": sorted(reference_success - candidate_success),
                    "candidate_only": sorted(candidate_success - reference_success),
                    "both_fail": sorted(
                        all_episodes - reference_success - candidate_success
                    ),
                })

    return {
        "validation_tag": validation_tag,
        "candidate_specs": candidates,
        "reference_method": reference_method,
        "task_ids": task_ids,
        "suites": suites,
        "rows": rows,
        "aggregates": aggregates,
        "paired_outcomes": paired_outcomes,
    }


def optional(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    payload = build_summary(
        args.output_root,
        args.validation_tag,
        list(args.candidates),
        args.reference_candidate,
        list(args.task_ids),
        list(args.suites),
        args.train_task_id,
        args.train_suite,
    )

    print("| Method | Params | Trainable | Train s | Loss | Success | Denoise ms | Infer ms | Internal |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["aggregates"]:
        ratio = row["internal_ratio"]
        print(
            f"| {row['method']} | {row['all_parameters']:,} | "
            f"{row['trainable_parameters']:,} | {row['training_duration_s']:.1f} | "
            f"{optional(row['best_loss'], 6)} | "
            f"{row['successes']}/{row['total_episodes']} | "
            f"{optional(row['action_denoise_total_ms_mean'])} | "
            f"{optional(row['infer_action_total_ms_mean'])} | "
            f"{optional(None if ratio is None else 100 * ratio, 1)}% |"
        )

    print(f"\nPaired disagreements versus {payload['reference_method']}:")
    for row in payload["paired_outcomes"]:
        if row["reference_only"] or row["candidate_only"] or row["both_fail"]:
            print(
                f"- task{row['task_id']} {row['suite']} {row['method']}: "
                f"reference_only={row['reference_only']}, "
                f"candidate_only={row['candidate_only']}, "
                f"both_fail={row['both_fail']}"
            )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "method", "fork_layer", "num_blocks", "all_parameters",
            "trainable_parameters", "training_duration_s", "training_updates",
            "best_loss", "successes", "total_episodes", "success_rate",
            "action_denoise_total_ms_mean", "infer_action_total_ms_mean",
            "internal_ratio", "checkpoint_bytes",
        ]
        with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in payload["aggregates"]:
                writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    main()
