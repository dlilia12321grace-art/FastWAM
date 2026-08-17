#!/usr/bin/env python
"""Aggregate held-out Dynamic ActionGap evaluations across LIBERO task IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_dynamic_action_gap_smoke import (
        format_optional,
        load_result,
        result_path,
        summarize_result,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<this_file>.py
    from summarize_dynamic_action_gap_smoke import (
        format_optional,
        load_result,
        result_path,
        summarize_result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_goal", "libero_spatial"],
    )
    parser.add_argument("--dynamic-method", default="v4_t036")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def aggregate_rows(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        raise ValueError(f"No rows found for method {method}.")
    total_episodes = sum(int(row["total_episodes"]) for row in selected)

    def episode_weighted_mean(key: str) -> float | None:
        available = [row for row in selected if row.get(key) is not None]
        if not available:
            return None
        weight = sum(int(row["total_episodes"]) for row in available)
        return sum(
            float(row[key]) * int(row["total_episodes"]) for row in available
        ) / weight

    full_steps = sum(int(row["full_steps"]) for row in selected)
    internal_steps = sum(int(row["internal_steps"]) for row in selected)
    total_steps = full_steps + internal_steps
    return {
        "method": method,
        "successes": sum(int(row["successes"]) for row in selected),
        "total_episodes": total_episodes,
        "success_rate": sum(int(row["successes"]) for row in selected)
        / total_episodes,
        "action_denoise_total_ms_mean": episode_weighted_mean(
            "action_denoise_total_ms_mean"
        ),
        "infer_action_total_ms_mean": episode_weighted_mean(
            "infer_action_total_ms_mean"
        ),
        "video_pre_dit_ms_mean": episode_weighted_mean("video_pre_dit_ms_mean"),
        "video_kv_prefill_ms_mean": episode_weighted_mean(
            "video_kv_prefill_ms_mean"
        ),
        "full_steps": full_steps,
        "internal_steps": internal_steps,
        "internal_ratio": internal_steps / total_steps if total_steps else None,
    }


def main() -> None:
    args = parse_args()
    suites = tuple(args.suites)
    has_full = all(
        result_path(
            args.output_root / f"task{task_id}", "full", suite, task_id
        ).is_file()
        for task_id in args.task_ids
        for suite in suites
    )
    dynamic_method = str(args.dynamic_method)
    methods = (
        ("full", "fixed_gap4", dynamic_method)
        if has_full
        else ("fixed_gap4", dynamic_method)
    )
    rows = []
    for task_id in args.task_ids:
        task_root = args.output_root / f"task{task_id}"
        for suite in suites:
            for method in methods:
                row = summarize_result(
                    load_result(result_path(task_root, method, suite, task_id)),
                    method,
                )
                row["task_id"] = task_id
                rows.append(row)

    print("| Task | Suite | Method | Success | Denoise ms | Infer ms | Internal ratio |")
    print("|---:|---|---|---:|---:|---:|---:|")
    for row in rows:
        ratio = row["internal_ratio"]
        print(
            f"| {row['task_id']} | {row['suite']} | {row['method']} | "
            f"{row['successes']}/{row['total_episodes']} | "
            f"{format_optional(row['action_denoise_total_ms_mean'])} | "
            f"{format_optional(row['infer_action_total_ms_mean'])} | "
            f"{format_optional(None if ratio is None else 100 * ratio, 1)}% |"
        )

    aggregates = [aggregate_rows(rows, method) for method in methods]
    aggregates_by_method = {row["method"]: row for row in aggregates}
    fixed = aggregates_by_method["fixed_gap4"]
    dynamic = aggregates_by_method[dynamic_method]
    fixed_denoise = fixed["action_denoise_total_ms_mean"]
    dynamic_denoise = dynamic["action_denoise_total_ms_mean"]
    fixed_infer = fixed["infer_action_total_ms_mean"]
    dynamic_infer = dynamic["infer_action_total_ms_mean"]
    comparison = {
        "success_rate_delta": dynamic["success_rate"] - fixed["success_rate"],
        "denoise_speedup": fixed_denoise / dynamic_denoise,
        "infer_speedup": fixed_infer / dynamic_infer,
    }
    if has_full:
        full = aggregates_by_method["full"]
        comparison.update({
            "fixed_denoise_speedup_vs_full": (
                full["action_denoise_total_ms_mean"] / fixed_denoise
            ),
            "dynamic_denoise_speedup_vs_full": (
                full["action_denoise_total_ms_mean"] / dynamic_denoise
            ),
            "fixed_infer_speedup_vs_full": (
                full["infer_action_total_ms_mean"] / fixed_infer
            ),
            "dynamic_infer_speedup_vs_full": (
                full["infer_action_total_ms_mean"] / dynamic_infer
            ),
        })
    print("\nAggregate:")
    for row in aggregates:
        ratio = row["internal_ratio"]
        print(
            f"- {row['method']}: {row['successes']}/{row['total_episodes']} success, "
            f"denoise {format_optional(row['action_denoise_total_ms_mean'])} ms, "
            f"infer {format_optional(row['infer_action_total_ms_mean'])} ms, "
            f"internal {format_optional(None if ratio is None else 100 * ratio, 1)}%"
        )
    if has_full:
        print("\nPer-action-chunk timing breakdown:")
        print("| Method | Video prefill ms | Action denoise ms | infer_action ms |")
        print("|---|---:|---:|---:|")
        for row in aggregates:
            video_parts = (
                row["video_pre_dit_ms_mean"],
                row["video_kv_prefill_ms_mean"],
            )
            video_ms = (
                sum(float(value) for value in video_parts)
                if all(value is not None for value in video_parts)
                else None
            )
            print(
                f"| {row['method']} | {format_optional(video_ms)} | "
                f"{format_optional(row['action_denoise_total_ms_mean'])} | "
                f"{format_optional(row['infer_action_total_ms_mean'])} |"
            )
    print(
        f"- {dynamic_method} vs fixed: success delta {comparison['success_rate_delta']:+.1%}, "
        f"denoise {comparison['denoise_speedup']:.3f}x, "
        f"infer {comparison['infer_speedup']:.3f}x"
    )
    if has_full:
        print(
            "- fixed vs full: denoise "
            f"{comparison['fixed_denoise_speedup_vs_full']:.3f}x, infer "
            f"{comparison['fixed_infer_speedup_vs_full']:.3f}x"
        )
        print(
            "- dynamic vs full: denoise "
            f"{comparison['dynamic_denoise_speedup_vs_full']:.3f}x, infer "
            f"{comparison['dynamic_infer_speedup_vs_full']:.3f}x"
        )

    paired = []
    for task_id in args.task_ids:
        for suite in suites:
            fixed_row = next(
                row for row in rows
                if row["task_id"] == task_id
                and row["suite"] == suite
                and row["method"] == "fixed_gap4"
            )
            dynamic_row = next(
                row for row in rows
                if row["task_id"] == task_id
                and row["suite"] == suite
                and row["method"] == dynamic_method
            )
            fixed_success = set(fixed_row["success_episodes"])
            dynamic_success = set(dynamic_row["success_episodes"])
            paired.append({
                "task_id": task_id,
                "suite": suite,
                "fixed_only": sorted(fixed_success - dynamic_success),
                "dynamic_only": sorted(dynamic_success - fixed_success),
                "both_fail": sorted(
                    set(range(int(fixed_row["total_episodes"])))
                    - fixed_success
                    - dynamic_success
                ),
            })
    print("\nPaired outcome disagreements:")
    for row in paired:
        if row["fixed_only"] or row["dynamic_only"] or row["both_fail"]:
            print(
                f"- task{row['task_id']} {row['suite']}: "
                f"fixed_only={row['fixed_only']}, "
                f"dynamic_only={row['dynamic_only']}, "
                f"both_fail={row['both_fail']}"
            )

    payload = {
        "task_ids": args.task_ids,
        "dynamic_method": dynamic_method,
        "rows": rows,
        "aggregates": aggregates,
        "comparison": comparison,
        "paired_outcomes": paired,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
