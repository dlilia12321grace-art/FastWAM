#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-collect}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-5}"
SUITES="${SUITES:-libero_goal libero_spatial}"
CROSS_TASK_IDS="${CROSS_TASK_IDS:-1 2 3 4}"
CROSS_NUM_TRIALS="${CROSS_NUM_TRIALS:-10}"
CROSS_SUITES="${CROSS_SUITES:-libero_goal libero_spatial}"
FROZEN_THRESHOLD="${FROZEN_THRESHOLD:-0.07011688053607941}"
DYNAMIC_THRESHOLDS="${DYNAMIC_THRESHOLDS:-}"
MAX_CACHE_AGE="${MAX_CACHE_AGE:-1}"
ACTION_GAP="${ACTION_GAP:-4}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/root/autodl-tmp/envs/fastwam/bin/python}"
FASTWAM_CKPT="${FASTWAM_CKPT:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
FASTWAM_STATS="${FASTWAM_STATS:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
INTERNAL_LORA_CKPT="${INTERNAL_LORA_CKPT:-/root/autodl-tmp/evaluate_results/internal_architecture_pareto/checkpoints/fork2_b0.best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/evaluate_results/dynamic_video_gap_smoke}"
CROSS_OUTPUT_ROOT="${CROSS_OUTPUT_ROOT:-/root/autodl-tmp/evaluate_results/dynamic_video_gap_cross_v1}"
BASELINE_ROOT="${BASELINE_ROOT:-/root/autodl-tmp/evaluate_results/dynamic_action_gap_fork2_b0}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

for required in "$FASTWAM_CKPT" "$FASTWAM_STATS" "$INTERNAL_LORA_CKPT"; do
  [[ -f "$required" ]] || { echo "Required file not found: $required" >&2; exit 2; }
done
mkdir -p "$OUTPUT_ROOT"

common_args=(
  task=libero_uncond_2cam224_1e-4
  "ckpt=${FASTWAM_CKPT}"
  gpu_id=0
  "EVALUATION.dataset_stats_path=${FASTWAM_STATS}"
  EVALUATION.save_rollout_video=false
  EVALUATION.save_action_trace=false
  EVALUATION.profile_timing=true
  EVALUATION.enable_internal_lora_branch=true
  "EVALUATION.internal_lora_checkpoint=${INTERNAL_LORA_CKPT}"
  EVALUATION.internal_lora_fork_layer=2
  EVALUATION.internal_lora_source_start_layer=30
  EVALUATION.internal_lora_source_end_layer=29
  EVALUATION.internal_lora_rank=8
  EVALUATION.internal_lora_alpha=16.0
  EVALUATION.enable_action_gap_schedule=true
  "EVALUATION.action_gap=${ACTION_GAP}"
  EVALUATION.enable_dynamic_action_gap=false
  EVALUATION.collect_dynamic_action_gap_data=false
)

run_collect() {
  local suite output_dir result
  for suite in $SUITES; do
    output_dir="$OUTPUT_ROOT/collect_delta_${suite}"
    result="$output_dir/$suite/gpu0_task${TASK_ID}_results.json"
    if [[ -f "$result" ]]; then
      echo "[skip collect] task${TASK_ID} ${suite}"
      continue
    fi
    "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
      "${common_args[@]}" \
      "EVALUATION.task_id=${TASK_ID}" \
      "EVALUATION.num_trials=${NUM_TRIALS}" \
      "EVALUATION.task_suite_name=${suite}" \
      "EVALUATION.output_dir=${output_dir}" \
      EVALUATION.enable_video_gap_schedule=true \
      EVALUATION.video_gap=1 \
      EVALUATION.enable_dynamic_video_gap=false
  done
  "$FASTWAM_PYTHON" scripts/summarize_video_gap_deltas.py "$OUTPUT_ROOT" \
    --task-id "$TASK_ID" --suites $SUITES \
    --json-output "$OUTPUT_ROOT/video_gap_delta_summary.json"
}

load_default_thresholds() {
  if [[ -n "$DYNAMIC_THRESHOLDS" ]]; then
    return
  fi
  local summary="$OUTPUT_ROOT/video_gap_delta_summary.json"
  [[ -f "$summary" ]] || { echo "Run collect before dynamic." >&2; exit 2; }
  DYNAMIC_THRESHOLDS="$($FASTWAM_PYTHON -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["recommended_initial_thresholds"])))' "$summary")"
}

run_dynamic() {
  load_default_thresholds
  local threshold tag suite output_dir result
  for threshold in $DYNAMIC_THRESHOLDS; do
    tag="$($FASTWAM_PYTHON -c 'import sys; print(f"{float(sys.argv[1]):.8g}".replace(".","p"))' "$threshold")"
    for suite in $SUITES; do
      output_dir="$OUTPUT_ROOT/dynamic_img_t${tag}_${suite}"
      result="$output_dir/$suite/gpu0_task${TASK_ID}_results.json"
      if [[ -f "$result" ]]; then
        echo "[skip dynamic] t${threshold} task${TASK_ID} ${suite}"
        continue
      fi
      "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
        "${common_args[@]}" \
        "EVALUATION.task_id=${TASK_ID}" \
        "EVALUATION.num_trials=${NUM_TRIALS}" \
        "EVALUATION.task_suite_name=${suite}" \
        "EVALUATION.output_dir=${output_dir}" \
        EVALUATION.enable_video_gap_schedule=false \
        EVALUATION.enable_dynamic_video_gap=true \
        "EVALUATION.dynamic_video_gap_image_mse_threshold=${threshold}" \
        "EVALUATION.dynamic_video_gap_max_cache_age=${MAX_CACHE_AGE}"
    done
  done
  "$FASTWAM_PYTHON" scripts/summarize_dynamic_video_gap.py "$OUTPUT_ROOT" \
    --thresholds $DYNAMIC_THRESHOLDS --task-id "$TASK_ID" --suites $SUITES \
    --json-output "$OUTPUT_ROOT/dynamic_video_gap_summary.json"
}

run_cross() {
  local task_id suite output_dir result
  mkdir -p "$CROSS_OUTPUT_ROOT"
  for task_id in $CROSS_TASK_IDS; do
    for suite in $CROSS_SUITES; do
      output_dir="$CROSS_OUTPUT_ROOT/task${task_id}/dynamic_img_q20_${suite}"
      result="$output_dir/$suite/gpu0_task${task_id}_results.json"
      if [[ -f "$result" ]]; then
        echo "[skip cross] task${task_id} ${suite}"
        continue
      fi
      echo "=== Dynamic VideoGap q20: task${task_id} ${suite} ==="
      "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
        "${common_args[@]}" \
        "EVALUATION.task_id=${task_id}" \
        "EVALUATION.num_trials=${CROSS_NUM_TRIALS}" \
        "EVALUATION.task_suite_name=${suite}" \
        "EVALUATION.output_dir=${output_dir}" \
        EVALUATION.enable_video_gap_schedule=false \
        EVALUATION.enable_dynamic_video_gap=true \
        "EVALUATION.dynamic_video_gap_image_mse_threshold=${FROZEN_THRESHOLD}" \
        "EVALUATION.dynamic_video_gap_max_cache_age=${MAX_CACHE_AGE}"
    done
  done
  "$FASTWAM_PYTHON" scripts/summarize_dynamic_video_gap_cross.py \
    "$CROSS_OUTPUT_ROOT" \
    --baseline-root "$BASELINE_ROOT" \
    --task-ids $CROSS_TASK_IDS \
    --suites $CROSS_SUITES \
    --num-trials "$CROSS_NUM_TRIALS" \
    --json-output "$CROSS_OUTPUT_ROOT/dynamic_video_gap_cross_summary.json"
}

case "$STAGE" in
  collect) run_collect ;;
  dynamic) run_dynamic ;;
  cross) run_cross ;;
  all) run_collect; run_dynamic ;;
  *) echo "Unknown stage '$STAGE'; expected collect, dynamic, cross, or all." >&2; exit 2 ;;
esac
