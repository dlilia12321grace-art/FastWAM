#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-5}"
SUITES="${SUITES:-libero_goal libero_spatial}"
VIDEO_GAPS="${VIDEO_GAPS:-1 2 4}"
ACTION_GAP="${ACTION_GAP:-4}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/root/autodl-tmp/envs/fastwam/bin/python}"
FASTWAM_CKPT="${FASTWAM_CKPT:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
FASTWAM_STATS="${FASTWAM_STATS:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
INTERNAL_LORA_CKPT="${INTERNAL_LORA_CKPT:-/root/autodl-tmp/evaluate_results/internal_architecture_pareto/checkpoints/fork2_b0.best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/evaluate_results/video_gap_smoke}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

for required in "$FASTWAM_CKPT" "$FASTWAM_STATS" "$INTERNAL_LORA_CKPT"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local gap="$1"
  local suite="$2"
  local output_dir="$OUTPUT_ROOT/video_gap${gap}_${suite}"
  local result="$output_dir/$suite/gpu0_task${TASK_ID}_results.json"
  if [[ -f "$result" ]]; then
    echo "[skip] video_gap${gap} task${TASK_ID} ${suite}"
    return
  fi
  echo "=== VideoGap=${gap}: task${TASK_ID} ${suite} ==="
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    task=libero_uncond_2cam224_1e-4 \
    "ckpt=${FASTWAM_CKPT}" \
    gpu_id=0 \
    "EVALUATION.task_id=${TASK_ID}" \
    "EVALUATION.num_trials=${NUM_TRIALS}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.dataset_stats_path=${FASTWAM_STATS}" \
    "EVALUATION.output_dir=${output_dir}" \
    EVALUATION.save_rollout_video=false \
    EVALUATION.save_action_trace=false \
    EVALUATION.profile_timing=true \
    EVALUATION.enable_video_gap_schedule=true \
    "EVALUATION.video_gap=${gap}" \
    EVALUATION.enable_internal_lora_branch=true \
    "EVALUATION.internal_lora_checkpoint=${INTERNAL_LORA_CKPT}" \
    EVALUATION.internal_lora_fork_layer=2 \
    EVALUATION.internal_lora_source_start_layer=30 \
    EVALUATION.internal_lora_source_end_layer=29 \
    EVALUATION.internal_lora_rank=8 \
    EVALUATION.internal_lora_alpha=16.0 \
    EVALUATION.enable_action_gap_schedule=true \
    "EVALUATION.action_gap=${ACTION_GAP}" \
    EVALUATION.enable_dynamic_action_gap=false \
    EVALUATION.collect_dynamic_action_gap_data=false
}

run_smoke() {
  local gap suite
  for gap in $VIDEO_GAPS; do
    for suite in $SUITES; do
      run_one "$gap" "$suite"
    done
  done
}

run_summary() {
  "$FASTWAM_PYTHON" scripts/summarize_video_gap.py "$OUTPUT_ROOT" \
    --gaps $VIDEO_GAPS \
    --suites $SUITES \
    --task-id "$TASK_ID" \
    --json-output "$OUTPUT_ROOT/video_gap_summary.json"
}

case "$STAGE" in
  smoke) run_smoke ;;
  summarize) run_summary ;;
  all) run_smoke; run_summary ;;
  *) echo "Unknown stage '$STAGE'; expected smoke, summarize, or all." >&2; exit 2 ;;
esac
