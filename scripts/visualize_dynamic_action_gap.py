#!/usr/bin/env python
"""Visualize Dynamic ActionGap importance predictions and routing behavior."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-failure-heatmaps", type=int, default=20)
    return parser.parse_args()


def find_dynamic_results(roots: list[Path]) -> list[Path]:
    paths = {
        path.resolve()
        for root in roots
        for pattern in (
            "task*/v4_t036_*/*/gpu0_task*_results.json",
            "v4_t036_*/*/gpu0_task*_results.json",
        )
        for path in root.glob(pattern)
    }
    return sorted(paths)


def extract_dynamic_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    suite = str(data["task_suite"])
    task_id = int(data["task_id"])
    failures = {int(index) for index in data.get("failure_episodes", [])}
    episodes = data.get("timing_profile", {}).get("episodes", [])
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        for chunk_index, chunk in enumerate(episode.get("chunks", [])):
            dynamic = chunk.get("dynamic_action_gap")
            if dynamic is None:
                continue
            for step_position, step in enumerate(dynamic.get("steps", [])):
                rows.append({
                    "suite": suite,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "success": episode_index not in failures,
                    "chunk_index": chunk_index,
                    "step_index": int(
                        step.get("step_index", step.get("step", step_position))
                    ),
                    "predicted_gap": float(step["predicted_gap"]),
                    "threshold": float(step["threshold"]),
                    "route": str(step["route"]),
                    "reason": str(step["reason"]),
                })
    return rows


def save_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_step_metric(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[int, list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        value = (
            float(row[key])
            if key != "internal"
            else float(row["route"] == "internal")
        )
        grouped[row["suite"]][row["step_index"]].append(value)
    return grouped


def plot_step_curves(rows: list[dict[str, Any]], output_dir: Path) -> None:
    gap_groups = group_step_metric(rows, "predicted_gap")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for suite, by_step in sorted(gap_groups.items()):
        steps = sorted(by_step)
        means = [np.mean(by_step[step]) for step in steps]
        lower = [np.percentile(by_step[step], 25) for step in steps]
        upper = [np.percentile(by_step[step], 75) for step in steps]
        ax.plot(steps, means, marker="o", label=suite)
        ax.fill_between(steps, lower, upper, alpha=0.15)
    threshold = float(np.median([row["threshold"] for row in rows]))
    ax.axhline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    ax.set(xlabel="Denoising step", ylabel="Predicted importance (log gap)")
    ax.set_title("Predicted step importance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "importance_by_step.png", dpi=180)
    plt.close(fig)

    internal_groups = group_step_metric(rows, "internal")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for suite, by_step in sorted(internal_groups.items()):
        steps = sorted(by_step)
        rates = [100.0 * np.mean(by_step[step]) for step in steps]
        ax.plot(steps, rates, marker="o", label=suite)
    ax.set(
        xlabel="Denoising step",
        ylabel="Internal route (%)",
        ylim=(-5, 105),
    )
    ax.set_title("Internal routing rate by denoising step")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "internal_rate_by_step.png", dpi=180)
    plt.close(fig)


def plot_reason_distribution(rows: list[dict[str, Any]], output_dir: Path) -> None:
    suites = sorted({row["suite"] for row in rows})
    reasons = sorted({row["reason"] for row in rows})
    counts = {
        suite: Counter(row["reason"] for row in rows if row["suite"] == suite)
        for suite in suites
    }
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(suites))
    for reason in reasons:
        values = np.asarray([
            100.0 * counts[suite][reason] / max(sum(counts[suite].values()), 1)
            for suite in suites
        ])
        ax.bar(suites, values, bottom=bottom, label=reason)
        bottom += values
    ax.set(ylabel="Route decisions (%)", ylim=(0, 100))
    ax.set_title("Routing reason distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "route_reason_distribution.png", dpi=180)
    plt.close(fig)


def plot_task_heatmap(rows: list[dict[str, Any]], output_dir: Path) -> None:
    suites = sorted({row["suite"] for row in rows})
    tasks = sorted({int(row["task_id"]) for row in rows})
    matrix = np.full((len(suites), len(tasks)), np.nan)
    for suite_index, suite in enumerate(suites):
        for task_index, task_id in enumerate(tasks):
            selected = [
                row for row in rows
                if row["suite"] == suite and row["task_id"] == task_id
            ]
            if selected:
                matrix[suite_index, task_index] = 100.0 * np.mean([
                    row["route"] == "internal" for row in selected
                ])
    fig, ax = plt.subplots(figsize=(max(7, len(tasks)), 1.5 + len(suites)))
    image = ax.imshow(matrix, vmin=0, vmax=80, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_yticks(range(len(suites)), suites)
    ax.set(xlabel="Task ID", title="Internal routing ratio (%)")
    for i in range(len(suites)):
        for j in range(len(tasks)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Internal (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "task_internal_ratio_heatmap.png", dpi=180)
    plt.close(fig)


def plot_episode_heatmap(
    episode_rows: list[dict[str, Any]], path: Path
) -> None:
    chunks = sorted({row["chunk_index"] for row in episode_rows})
    steps = sorted({row["step_index"] for row in episode_rows})
    gap = np.full((len(chunks), len(steps)), np.nan)
    route = np.full((len(chunks), len(steps)), np.nan)
    chunk_pos = {value: index for index, value in enumerate(chunks)}
    step_pos = {value: index for index, value in enumerate(steps)}
    for row in episode_rows:
        i = chunk_pos[row["chunk_index"]]
        j = step_pos[row["step_index"]]
        gap[i, j] = row["predicted_gap"]
        route[i, j] = float(row["route"] == "internal")
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.2, len(chunks) * 0.16)))
    first = axes[0].imshow(gap, aspect="auto", cmap="magma")
    axes[0].set(title="Predicted importance", xlabel="Denoising step", ylabel="Replan chunk")
    fig.colorbar(first, ax=axes[0])
    second = axes[1].imshow(route, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    axes[1].set(title="Route (blue=full, red=internal)", xlabel="Denoising step")
    fig.colorbar(second, ax=axes[1], ticks=[0, 1])
    fig.suptitle(
        f"{episode_rows[0]['suite']} task{episode_rows[0]['task_id']} "
        f"episode{episode_rows[0]['episode_index']} success={episode_rows[0]['success']}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for suite in sorted({row["suite"] for row in rows}):
        selected = [row for row in rows if row["suite"] == suite]
        reasons = Counter(row["reason"] for row in selected)
        summary[suite] = {
            "num_steps": len(selected),
            "predicted_gap_mean": float(np.mean([row["predicted_gap"] for row in selected])),
            "internal_ratio": float(np.mean([row["route"] == "internal" for row in selected])),
            "route_reasons": dict(reasons),
        }
    return summary


def main() -> None:
    args = parse_args()
    paths = find_dynamic_results(args.roots)
    if not paths:
        raise FileNotFoundError("No v4_t036 result JSON files were found.")
    rows = [row for path in paths for row in extract_dynamic_rows(json.loads(path.read_text()))]
    if not rows:
        raise ValueError("The result files contain no Dynamic ActionGap step logs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_rows_csv(rows, args.output_dir / "dynamic_route_steps.csv")
    plot_step_curves(rows, args.output_dir)
    plot_reason_distribution(rows, args.output_dir)
    plot_task_heatmap(rows, args.output_dir)

    failures: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["success"]:
            failures[(row["suite"], row["task_id"], row["episode_index"])].append(row)
    for (suite, task_id, episode_index), episode_rows in list(sorted(failures.items()))[
        : args.max_failure_heatmaps
    ]:
        plot_episode_heatmap(
            episode_rows,
            args.output_dir / "failure_heatmaps" / f"{suite}_task{task_id}_episode{episode_index}.png",
        )

    payload = {
        "source_files": [str(path) for path in paths],
        "num_step_rows": len(rows),
        "num_failure_episodes": len(failures),
        "suite_summary": build_summary(rows),
    }
    (args.output_dir / "dynamic_route_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload["suite_summary"], indent=2, ensure_ascii=False))
    print(f"Saved Dynamic ActionGap visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
