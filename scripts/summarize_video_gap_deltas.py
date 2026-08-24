from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_goal", "libero_spatial"],
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    scores: list[float] = []
    per_suite: dict[str, dict[str, float | int]] = {}
    for suite in args.suites:
        path = (
            args.root
            / f"collect_delta_{suite}"
            / suite
            / f"gpu0_task{args.task_id}_results.json"
        )
        data = json.loads(path.read_text())
        suite_scores = [
            float(chunk["video_gap"]["image_mse"])
            for episode in data.get("timing_profile", {}).get("episodes", [])
            for chunk in episode.get("chunks", [])
            if chunk.get("video_gap", {}).get("image_mse") is not None
        ]
        scores.extend(suite_scores)
        per_suite[suite] = {
            "num_scores": len(suite_scores),
            "mean": float(np.mean(suite_scores)),
            "median": float(np.median(suite_scores)),
        }

    if not scores:
        raise RuntimeError("No VideoGap image_mse values were found.")
    quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(scores, q))
        for q in (0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.75, 0.90)
    }
    output = {
        "task_id": args.task_id,
        "num_scores": len(scores),
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "quantiles": quantiles,
        "per_suite": per_suite,
        "recommended_initial_thresholds": [quantiles["q20"], quantiles["q35"]],
    }
    print(json.dumps(output, indent=2))
    output_path = args.json_output or args.root / "video_gap_delta_summary.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved VideoGap delta summary: {output_path}")


if __name__ == "__main__":
    main()
