from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.summarize_video_gap import aggregate, summarize_result
except ImportError:
    from summarize_video_gap import aggregate, summarize_result


def threshold_tag(value: float) -> str:
    return f"{value:.8g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--suites", nargs="+", default=["libero_goal", "libero_spatial"]
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    methods = []
    print("| Method | Success | Image ms | Video pre ms | Video KV ms | Action ms | Infer ms | Refresh |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for threshold in args.thresholds:
        tag = threshold_tag(threshold)
        rows = []
        for suite in args.suites:
            path = (
                args.root
                / f"dynamic_img_t{tag}_{suite}"
                / suite
                / f"gpu0_task{args.task_id}_results.json"
            )
            rows.append(summarize_result(path))
        row = {
            "method": f"dynamic_img_t{tag}",
            "threshold": threshold,
            **aggregate(rows),
        }
        methods.append(row)
        print(
            f"| {row['method']} | {row['successes']}/{row['episodes']} | "
            f"{row['image_encode_ms']:.2f} | {row['video_pre_dit_ms']:.2f} | "
            f"{row['video_kv_prefill_ms']:.2f} | {row['action_denoise_ms']:.2f} | "
            f"{row['infer_action_ms']:.2f} | {100 * row['refresh_ratio']:.1f}% |"
        )

    output = {"task_id": args.task_id, "methods": methods}
    output_path = args.json_output or args.root / "dynamic_video_gap_summary.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved Dynamic VideoGap summary: {output_path}")


if __name__ == "__main__":
    main()
