#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-fork-screen}"
FORK_LAYERS="${FORK_LAYERS:-2 4 8 12 16}"
NUM_BLOCKS="${NUM_BLOCKS:-1}"
BLOCK_COUNTS="${BLOCK_COUNTS:-0 1 2 4}"
SOURCE_END_LAYER="${SOURCE_END_LAYER:-30}"
TRAIN_MAX_UPDATES="${TRAIN_MAX_UPDATES:-20}"
TRAIN_NUM_TRIALS="${TRAIN_NUM_TRIALS:-1}"
EVAL_NUM_TRIALS="${EVAL_NUM_TRIALS:-5}"
TASK_ID="${TASK_ID:-0}"
TRAIN_SUITE="${TRAIN_SUITE:-libero_spatial}"
EVAL_SUITES="${EVAL_SUITES:-libero_goal libero_spatial}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/root/autodl-tmp/envs/fastwam/bin/python}"
FASTWAM_CKPT="${FASTWAM_CKPT:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
FASTWAM_STATS="${FASTWAM_STATS:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/evaluate_results/internal_architecture_sweep}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

for required in "$FASTWAM_CKPT" "$FASTWAM_STATS"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 2
  fi
done
if (( NUM_BLOCKS < 0 )); then
  echo "NUM_BLOCKS must be non-negative." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/checkpoints" "$OUTPUT_ROOT/train" "$OUTPUT_ROOT/eval"
if (( NUM_BLOCKS == 0 )); then
  SOURCE_START_LAYER="$SOURCE_END_LAYER"
  BRANCH_SOURCE_END_LAYER=$((SOURCE_END_LAYER - 1))
else
  SOURCE_START_LAYER=$((SOURCE_END_LAYER - NUM_BLOCKS + 1))
  BRANCH_SOURCE_END_LAYER="$SOURCE_END_LAYER"
fi
if (( SOURCE_START_LAYER < 1 )); then
  echo "NUM_BLOCKS=$NUM_BLOCKS exceeds SOURCE_END_LAYER=$SOURCE_END_LAYER." >&2
  exit 2
fi

base_args=(
  task=libero_uncond_2cam224_1e-4
  "ckpt=${FASTWAM_CKPT}"
  gpu_id=0
  "EVALUATION.task_id=${TASK_ID}"
  "EVALUATION.dataset_stats_path=${FASTWAM_STATS}"
  EVALUATION.save_rollout_video=false
  EVALUATION.save_action_trace=false
  EVALUATION.profile_timing=true
)

method_name() {
  local fork_layer="$1"
  echo "fork${fork_layer}_b${NUM_BLOCKS}"
}

train_one() {
  local fork_layer="$1"
  local method
  method="$(method_name "$fork_layer")"
  local checkpoint="$OUTPUT_ROOT/checkpoints/${method}.pt"
  local best_checkpoint="$OUTPUT_ROOT/checkpoints/${method}.best.pt"
  if [[ -f "$best_checkpoint" ]]; then
    echo "[skip train] $method: $best_checkpoint already exists"
    return
  fi
  echo "=== Train $method ==="
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${base_args[@]}" \
    "EVALUATION.task_suite_name=${TRAIN_SUITE}" \
    "EVALUATION.num_trials=${TRAIN_NUM_TRIALS}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/train/${method}" \
    EVALUATION.enable_internal_lora_branch=false \
    "+EVALUATION.train_internal_lora=true" \
    "+EVALUATION.internal_lora_train_steps=[0,1,2,3,4,5,6]" \
    "+EVALUATION.internal_lora_max_updates=${TRAIN_MAX_UPDATES}" \
    "+EVALUATION.internal_lora_learning_rate=0.0001" \
    "+EVALUATION.internal_lora_weight_decay=0.0" \
    "+EVALUATION.internal_lora_output_checkpoint=${checkpoint}" \
    "EVALUATION.internal_lora_fork_layer=${fork_layer}" \
    "EVALUATION.internal_lora_source_start_layer=${SOURCE_START_LAYER}" \
    "EVALUATION.internal_lora_source_end_layer=${BRANCH_SOURCE_END_LAYER}" \
    EVALUATION.internal_lora_rank=8 \
    EVALUATION.internal_lora_alpha=16.0
}

eval_one() {
  local fork_layer="$1"
  local suite="$2"
  local method
  method="$(method_name "$fork_layer")"
  local checkpoint="$OUTPUT_ROOT/checkpoints/${method}.best.pt"
  local result="$OUTPUT_ROOT/eval/${method}_${suite}/${suite}/gpu0_task${TASK_ID}_results.json"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing trained checkpoint: $checkpoint" >&2
    exit 2
  fi
  if [[ -f "$result" ]]; then
    echo "[skip eval] $method $suite: result already exists"
    return
  fi
  echo "=== Evaluate $method on $suite ==="
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${base_args[@]}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.num_trials=${EVAL_NUM_TRIALS}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/eval/${method}_${suite}" \
    EVALUATION.enable_internal_lora_branch=true \
    "EVALUATION.internal_lora_checkpoint=${checkpoint}" \
    "EVALUATION.internal_lora_fork_layer=${fork_layer}" \
    "EVALUATION.internal_lora_source_start_layer=${SOURCE_START_LAYER}" \
    "EVALUATION.internal_lora_source_end_layer=${BRANCH_SOURCE_END_LAYER}" \
    EVALUATION.internal_lora_rank=8 \
    EVALUATION.internal_lora_alpha=16.0 \
    EVALUATION.enable_action_gap_schedule=true \
    EVALUATION.action_gap=4 \
    EVALUATION.enable_dynamic_action_gap=false \
    EVALUATION.collect_dynamic_action_gap_data=false
}

run_train() {
  local fork_layer
  for fork_layer in $FORK_LAYERS; do
    train_one "$fork_layer"
  done
}

run_eval() {
  local fork_layer suite
  for fork_layer in $FORK_LAYERS; do
    for suite in $EVAL_SUITES; do
      eval_one "$fork_layer" "$suite"
    done
  done
}

run_summarize() {
  "$FASTWAM_PYTHON" scripts/summarize_internal_architecture_sweep.py \
    "$OUTPUT_ROOT" \
    --fork-layers $FORK_LAYERS \
    --num-blocks "$NUM_BLOCKS" \
    --suites $EVAL_SUITES \
    --task-id "$TASK_ID" \
    --json-output "$OUTPUT_ROOT/fork_screen_summary.json" \
    --csv-output "$OUTPUT_ROOT/fork_screen_summary.csv"
}

run_block_screen() {
  local blocks
  for blocks in $BLOCK_COUNTS; do
    NUM_BLOCKS="$blocks" bash "$0" train
    NUM_BLOCKS="$blocks" bash "$0" eval
  done
  "$FASTWAM_PYTHON" scripts/summarize_internal_architecture_sweep.py \
    "$OUTPUT_ROOT" \
    --fork-layers $FORK_LAYERS \
    --block-counts $BLOCK_COUNTS \
    --suites $EVAL_SUITES \
    --task-id "$TASK_ID" \
    --json-output "$OUTPUT_ROOT/block_screen_summary.json" \
    --csv-output "$OUTPUT_ROOT/block_screen_summary.csv"
}

case "$STAGE" in
  train) run_train ;;
  eval) run_eval ;;
  summarize) run_summarize ;;
  block-screen) run_block_screen ;;
  fork-screen)
    run_train
    run_eval
    run_summarize
    ;;
  *)
    echo "Usage: $0 [train|eval|summarize|fork-screen|block-screen]" >&2
    exit 2
    ;;
esac

echo "Internal architecture sweep stage '$STAGE' completed: $OUTPUT_ROOT"
