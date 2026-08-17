#!/usr/bin/env python
"""Summarize fixed-gap and Dynamic ActionGap LIBERO smoke results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_goal", "libero_spatial"],
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def result_path(root: Path, method: str, suite: str, task_id: int = 0) -> Path:
    return root / f"{method}_{suite}" / suite / f"gpu0_task{task_id}_results.json"


def discover_methods(
    root: Path,
    task_id: int = 0,
    suites: tuple[str, ...] = ("libero_goal", "libero_spatial"),
) -> list[str]:
    """Return complete two-suite methods in a stable presentation order."""
    candidates = [
        "fixed_gap4",
        "dynamic",
        "hybrid",
        "v3_safe",
        "v3_balanced",
        "v3_fast",
        "v4_t034",
        "v4_t036",
    ]
    discovery_suffix = f"_{suites[0]}"
    discovered = {
        path.name[: -len(discovery_suffix)]
        for path in root.glob(f"*{discovery_suffix}")
        if path.is_dir() and not path.name.startswith("collect_")
    }
    candidates.extend(sorted(discovered.difference(candidates)))
    methods = [
        method
        for method in candidates
        if all(
            result_path(root, method, suite, task_id).is_file()
            for suite in suites
        )
    ]
    if "fixed_gap4" not in methods:
        raise FileNotFoundError(
            "A complete fixed_gap4 baseline is required for comparison."
        )
    return methods


def load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_result(data: dict[str, Any], method: str) -> dict[str, Any]:
    timing = data.get("timing_profile", {}).get("summary", {})
    route_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    predicted_gaps = []
    for episode in data.get("timing_profile", {}).get("episodes", []):
        for chunk in episode.get("chunks", []):
            dynamic = chunk.get("dynamic_action_gap")
            if dynamic is not None:
                for step in dynamic.get("steps", []):
                    route_counter[str(step["route"])] += 1
                    reason_counter[str(step["reason"])] += 1
                    predicted_gaps.append(float(step["predicted_gap"]))
                continue
            internal = chunk.get("internal_lora_inference")
            if internal is not None:
                route_counter.update(str(mode) for mode in internal.get("step_modes", []))
    total_routes = sum(route_counter.values())
    return {
        "method": method,
        "suite": str(data["task_suite"]),
        "successes": int(data["successes"]),
        "total_episodes": int(data["total_episodes"]),
        "success_rate": float(data["successes"]) / max(int(data["total_episodes"]), 1),
        "action_denoise_total_ms_mean": timing.get("action_denoise_total_ms_mean"),
        "infer_action_total_ms_mean": timing.get("infer_action_total_ms_mean"),
        "video_pre_dit_ms_mean": timing.get("video_pre_dit_ms_mean"),
        "video_kv_prefill_ms_mean": timing.get("video_kv_prefill_ms_mean"),
        "full_steps": int(route_counter["full"]),
        "internal_steps": int(route_counter["internal"]),
        "internal_ratio": (
            float(route_counter["internal"]) / total_routes if total_routes else None
        ),
        "predicted_gap_mean": (
            sum(predicted_gaps) / len(predicted_gaps) if predicted_gaps else None
        ),
        "route_reasons": dict(reason_counter),
        "success_episodes": [int(index) for index in data.get("success_episodes", [])],
        "failure_episodes": [int(index) for index in data.get("failure_episodes", [])],
    }


def format_optional(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    rows = []
    suites = tuple(args.suites)
    methods = discover_methods(args.output_root, args.task_id, suites)
    for suite in suites:
        for method in methods:
            rows.append(
                summarize_result(
                    load_result(
                        result_path(args.output_root, method, suite, args.task_id)
                    ),
                    method,
                )
            )
            rows[-1]["task_id"] = args.task_id

    print("| Suite | Method | Success | Denoise ms | Infer ms | Internal ratio |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        internal_ratio = row["internal_ratio"]
        print(
            f"| {row['suite']} | {row['method']} | "
            f"{row['successes']}/{row['total_episodes']} | "
            f"{format_optional(row['action_denoise_total_ms_mean'])} | "
            f"{format_optional(row['infer_action_total_ms_mean'])} | "
            f"{format_optional(None if internal_ratio is None else 100 * internal_ratio, 1)}% |"
        )

    comparisons = []
    for suite in suites:
        fixed = next(row for row in rows if row["suite"] == suite and row["method"] == "fixed_gap4")
        for method in methods[1:]:
            candidate = next(row for row in rows if row["suite"] == suite and row["method"] == method)
            fixed_ms = fixed["action_denoise_total_ms_mean"]
            candidate_ms = candidate["action_denoise_total_ms_mean"]
            comparisons.append({
                "suite": suite,
                "method": method,
                "success_rate_delta": candidate["success_rate"] - fixed["success_rate"],
                "denoise_speedup_vs_fixed": (
                    float(fixed_ms) / float(candidate_ms)
                    if fixed_ms is not None and candidate_ms not in (None, 0)
                    else None
                ),
            })
    payload = {
        "task_id": args.task_id,
        "rows": rows,
        "comparisons": comparisons,
    }
    print("\nDecision summary:")
    for comparison in comparisons:
        print(
            f"- {comparison['suite']} {comparison['method']}: success delta "
            f"{comparison['success_rate_delta']:+.1%}, denoise speedup "
            f"{format_optional(comparison['denoise_speedup_vs_fixed'], 3)}x"
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
