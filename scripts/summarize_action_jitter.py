#!/usr/bin/env python
"""Summarize saved LIBERO action discontinuity diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def mean_metric(episodes: list[dict[str, Any]], key: str) -> float:
    return sum(float(episode[key]) for episode in episodes) / len(episodes)


def main() -> None:
    args = parse_args()
    rows = []
    for suite in args.suites:
        for method in ("full", "fixed_gap4", "v4_t036"):
            path = (
                args.output_root
                / f"{method}_{suite}"
                / suite
                / f"gpu0_task{args.task_id}_results.json"
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            episodes = data["action_jitter"]["episodes"]
            rows.append({
                "suite": suite,
                "method": method,
                "successes": int(data["successes"]),
                "total_episodes": int(data["total_episodes"]),
                "arm_delta_mean": mean_metric(episodes, "arm_delta_l2_mean"),
                "arm_delta_p95": mean_metric(episodes, "arm_delta_l2_p95"),
                "arm_jerk_mean": mean_metric(episodes, "arm_jerk_l2_mean"),
                "boundary_delta_mean": mean_metric(
                    episodes, "replan_boundary_delta_l2_mean"
                ),
                "gripper_flips_mean": mean_metric(
                    episodes, "gripper_flip_count"
                ),
            })

    print("| Suite | Method | Success | Delta mean | Delta p95 | Jerk mean | Replan jump | Gripper flips |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['suite']} | {row['method']} | "
            f"{row['successes']}/{row['total_episodes']} | "
            f"{row['arm_delta_mean']:.5f} | {row['arm_delta_p95']:.5f} | "
            f"{row['arm_jerk_mean']:.5f} | {row['boundary_delta_mean']:.5f} | "
            f"{row['gripper_flips_mean']:.2f} |"
        )
    payload = {"task_id": args.task_id, "rows": rows}
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
