from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_video_gap import aggregate, summarize_result
except ImportError:
    from summarize_video_gap import aggregate, summarize_result


def load_successes(path: Path) -> set[int]:
    return set(json.loads(path.read_text())["success_episodes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", type=int, required=True)
    parser.add_argument("--suites", nargs="+", required=True)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    baseline_rows: list[dict[str, Any]] = []
    dynamic_rows: list[dict[str, Any]] = []
    disagreements = []
    all_episodes = set(range(args.num_trials))

    for task_id in args.task_ids:
        for suite in args.suites:
            baseline_path = (
                args.baseline_root
                / f"task{task_id}"
                / f"fixed_gap4_{suite}"
                / suite
                / f"gpu0_task{task_id}_results.json"
            )
            dynamic_path = (
                args.root
                / f"task{task_id}"
                / f"dynamic_img_q20_{suite}"
                / suite
                / f"gpu0_task{task_id}_results.json"
            )
            if not baseline_path.is_file():
                raise FileNotFoundError(baseline_path)
            if not dynamic_path.is_file():
                raise FileNotFoundError(dynamic_path)
            baseline_rows.append(summarize_result(baseline_path))
            dynamic_rows.append(summarize_result(dynamic_path))
            baseline_success = load_successes(baseline_path)
            dynamic_success = load_successes(dynamic_path)
            if baseline_success != dynamic_success:
                disagreements.append(
                    {
                        "task_id": task_id,
                        "suite": suite,
                        "baseline_only": sorted(baseline_success - dynamic_success),
                        "dynamic_only": sorted(dynamic_success - baseline_success),
                        "both_fail": sorted(
                            all_episodes - baseline_success - dynamic_success
                        ),
                    }
                )

    baseline = {"method": "fixed_gap4", **aggregate(baseline_rows)}
    dynamic = {"method": "dynamic_video_gap_q20", **aggregate(dynamic_rows)}
    comparison = {
        "success_delta": dynamic["successes"] - baseline["successes"],
        "infer_speedup": baseline["infer_action_ms"] / dynamic["infer_action_ms"],
        "video_kv_speedup": (
            baseline["video_kv_prefill_ms"] / dynamic["video_kv_prefill_ms"]
        ),
    }
    output = {
        "task_ids": args.task_ids,
        "suites": args.suites,
        "aggregates": [baseline, dynamic],
        "comparison": comparison,
        "paired_disagreements": disagreements,
    }

    print("| Method | Success | Image ms | Video pre ms | Video KV ms | Action ms | Infer ms | Refresh |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in (baseline, dynamic):
        refresh = "n/a" if row["refresh_ratio"] is None else f"{100 * row['refresh_ratio']:.1f}%"
        print(
            f"| {row['method']} | {row['successes']}/{row['episodes']} | "
            f"{row['image_encode_ms']:.2f} | {row['video_pre_dit_ms']:.2f} | "
            f"{row['video_kv_prefill_ms']:.2f} | {row['action_denoise_ms']:.2f} | "
            f"{row['infer_action_ms']:.2f} | {refresh} |"
        )
    print(
        f"- q20 vs fixed: success delta {comparison['success_delta']:+d}/"
        f"{dynamic['episodes']}, infer {comparison['infer_speedup']:.3f}x"
    )
    for row in disagreements:
        print(
            f"- task{row['task_id']} {row['suite']}: "
            f"fixed_only={row['baseline_only']}, "
            f"dynamic_only={row['dynamic_only']}, both_fail={row['both_fail']}"
        )

    output_path = args.json_output or args.root / "dynamic_video_gap_cross_summary.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved Dynamic VideoGap cross summary: {output_path}")


if __name__ == "__main__":
    main()
