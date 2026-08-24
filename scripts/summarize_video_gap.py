from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    episodes = data.get("timing_profile", {}).get("episodes", [])
    chunks = [chunk for episode in episodes for chunk in episode.get("chunks", [])]

    def timing_mean(key: str) -> float | None:
        return _mean([float(chunk[key]) for chunk in chunks if key in chunk])

    video_gap_rows = [chunk["video_gap"] for chunk in chunks if "video_gap" in chunk]
    refreshes = sum(bool(row["refresh_cache"]) for row in video_gap_rows)
    return {
        "successes": int(data["successes"]),
        "episodes": int(len(data["success_episodes"]) + len(data["failure_episodes"])),
        "image_encode_ms": timing_mean("image_encode_ms"),
        "video_pre_dit_ms": timing_mean("video_pre_dit_ms"),
        "video_kv_prefill_ms": timing_mean("video_kv_prefill_ms"),
        "action_denoise_ms": timing_mean("action_denoise_total_ms"),
        "infer_action_ms": timing_mean("infer_action_total_ms"),
        "chunks": len(chunks),
        "refreshes": int(refreshes),
        "video_gap_chunks": len(video_gap_rows),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = sum(row["chunks"] for row in rows)

    def weighted(key: str) -> float | None:
        valid = [row for row in rows if row[key] is not None and row["chunks"] > 0]
        if not valid:
            return None
        return sum(float(row[key]) * row["chunks"] for row in valid) / sum(
            row["chunks"] for row in valid
        )

    refreshes = sum(row["refreshes"] for row in rows)
    video_gap_chunks = sum(row["video_gap_chunks"] for row in rows)
    return {
        "successes": sum(row["successes"] for row in rows),
        "episodes": sum(row["episodes"] for row in rows),
        "image_encode_ms": weighted("image_encode_ms"),
        "video_pre_dit_ms": weighted("video_pre_dit_ms"),
        "video_kv_prefill_ms": weighted("video_kv_prefill_ms"),
        "action_denoise_ms": weighted("action_denoise_ms"),
        "infer_action_ms": weighted("infer_action_ms"),
        "chunks": chunks,
        "refreshes": refreshes,
        "video_gap_chunks": video_gap_chunks,
        "refresh_ratio": (
            refreshes / video_gap_chunks if video_gap_chunks else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--gaps", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_goal", "libero_spatial"],
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    output: dict[str, Any] = {"task_id": args.task_id, "methods": []}
    print("| Method | Success | Image encode ms | Video pre ms | Video KV ms | Action ms | Infer ms | Refresh |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for gap in args.gaps:
        rows = []
        for suite in args.suites:
            path = (
                args.root
                / f"video_gap{gap}_{suite}"
                / suite
                / f"gpu0_task{args.task_id}_results.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(summarize_result(path))
        summary = {"method": f"video_gap{gap}", "video_gap": gap, **aggregate(rows)}
        output["methods"].append(summary)
        print(
            f"| video_gap{gap} | {summary['successes']}/{summary['episodes']} | "
            f"{summary['image_encode_ms']:.2f} | {summary['video_pre_dit_ms']:.2f} | "
            f"{summary['video_kv_prefill_ms']:.2f} | {summary['action_denoise_ms']:.2f} | "
            f"{summary['infer_action_ms']:.2f} | {100 * summary['refresh_ratio']:.1f}% |"
        )

    baseline = output["methods"][0]
    for row in output["methods"][1:]:
        row["infer_speedup_vs_gap1"] = (
            baseline["infer_action_ms"] / row["infer_action_ms"]
        )
        print(
            f"- {row['method']} vs video_gap1: infer "
            f"{row['infer_speedup_vs_gap1']:.3f}x, success delta "
            f"{row['successes'] - baseline['successes']:+d}/{row['episodes']}"
        )

    output_path = args.json_output or args.root / "video_gap_summary.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved VideoGap summary: {output_path}")


if __name__ == "__main__":
    main()
