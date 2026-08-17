#!/usr/bin/env bash
set -euo pipefail

# End-to-end AutoDL smoke run: collect teacher data, train the gate, then
# compare fixed ActionGap=4 against learned routing on two LIBERO tasks.

STAGE="${1:-all}"
TASK_ID="${TASK_ID:-0}"
CROSS_TASK_IDS="${CROSS_TASK_IDS:-1 2 3 4}"
CROSS_NUM_TRIALS="${CROSS_NUM_TRIALS:-10}"
CROSS_SUITES="${CROSS_SUITES:-libero_goal libero_spatial}"
VIS_SUITES="${VIS_SUITES:-libero_goal}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-false}"
SAVE_ACTION_TRACE="${SAVE_ACTION_TRACE:-false}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/root/autodl-tmp/envs/fastwam/bin/python}"
FASTWAM_CKPT="${FASTWAM_CKPT:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
FASTWAM_STATS="${FASTWAM_STATS:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
INTERNAL_LORA_CKPT="${INTERNAL_LORA_CKPT:-/root/autodl-tmp/checkpoints/shallow_fork4_copy30_train20.best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/evaluate_results/dynamic_action_gap_smoke}"
NUM_TRIALS="${NUM_TRIALS:-5}"
GATE_EPOCHS="${GATE_EPOCHS:-30}"
GATE_CKPT="${GATE_CKPT:-${OUTPUT_ROOT}/dynamic_action_gap_gate.pt}"
CANDIDATE_METHOD="${CANDIDATE_METHOD:-candidate}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.36}"
DYNAMIC_MAX_INTERNAL_RUN="${DYNAMIC_MAX_INTERNAL_RUN:-7}"
DYNAMIC_ANCHOR_ACTION_GAP="${DYNAMIC_ANCHOR_ACTION_GAP:-8}"
MATCHED_TARGET_INTERNAL_RATIO="${MATCHED_TARGET_INTERNAL_RATIO:-0.65}"
MATCHED_RANDOM_SEED="${MATCHED_RANDOM_SEED:-20260815}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 2
  fi
}

require_file "$FASTWAM_CKPT"
require_file "$FASTWAM_STATS"
require_file "$INTERNAL_LORA_CKPT"
mkdir -p "$OUTPUT_ROOT"

base_args=(
  task=libero_uncond_2cam224_1e-4
  "ckpt=${FASTWAM_CKPT}"
  gpu_id=0
  "EVALUATION.task_id=${TASK_ID}"
  "EVALUATION.num_trials=${NUM_TRIALS}"
  "EVALUATION.dataset_stats_path=${FASTWAM_STATS}"
  "EVALUATION.save_rollout_video=${SAVE_ROLLOUT_VIDEO}"
  "EVALUATION.save_action_trace=${SAVE_ACTION_TRACE}"
  EVALUATION.profile_timing=true
)

internal_args=(
  EVALUATION.enable_internal_lora_branch=true
  "EVALUATION.internal_lora_checkpoint=${INTERNAL_LORA_CKPT}"
  EVALUATION.internal_lora_fork_layer=4
  EVALUATION.internal_lora_source_start_layer=30
  EVALUATION.internal_lora_source_end_layer=30
  EVALUATION.internal_lora_rank=8
  EVALUATION.internal_lora_alpha=16.0
)

common_args=("${base_args[@]}" "${internal_args[@]}")

run_collect() {
  local suite="$1"
  local output_dir="${OUTPUT_ROOT}/collect_${suite}"
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${common_args[@]}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.output_dir=${output_dir}" \
    EVALUATION.collect_dynamic_action_gap_data=true \
    EVALUATION.enable_action_gap_schedule=false \
    EVALUATION.enable_dynamic_action_gap=false
}

run_train() {
  local goal_data="${OUTPUT_ROOT}/collect_libero_goal/libero_goal/gpu0_task0_dynamic_action_gap.pt"
  local spatial_data="${OUTPUT_ROOT}/collect_libero_spatial/libero_spatial/gpu0_task0_dynamic_action_gap.pt"
  require_file "$goal_data"
  require_file "$spatial_data"
  "$FASTWAM_PYTHON" scripts/train_dynamic_action_gap.py \
    "$goal_data" "$spatial_data" \
    --output "$GATE_CKPT" \
    --epochs "$GATE_EPOCHS" \
    --max-internal-run 3 \
    --target-internal-rate 0.6 \
    --device cuda
}

run_train_meta_only() {
  local data_root="${GATE_DATA_ROOT:-/root/autodl-tmp/evaluate_results/dynamic_action_gap_smoke}"
  local goal_data="${data_root}/collect_libero_goal/libero_goal/gpu0_task0_dynamic_action_gap.pt"
  local spatial_data="${data_root}/collect_libero_spatial/libero_spatial/gpu0_task0_dynamic_action_gap.pt"
  require_file "$goal_data"
  require_file "$spatial_data"
  "$FASTWAM_PYTHON" scripts/train_dynamic_action_gap.py \
    "$goal_data" "$spatial_data" \
    --output "$GATE_CKPT" \
    --input-mode meta_only \
    --epochs "$GATE_EPOCHS" \
    --max-internal-run 3 \
    --target-internal-rate 0.6 \
    --device cuda
}

run_fixed_eval() {
  local suite="$1"
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${common_args[@]}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/fixed_gap4_${suite}" \
    EVALUATION.collect_dynamic_action_gap_data=false \
    EVALUATION.enable_dynamic_action_gap=false \
    EVALUATION.enable_action_gap_schedule=true \
    EVALUATION.action_gap=4
}

run_full_eval() {
  local suite="$1"
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${base_args[@]}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/full_${suite}" \
    EVALUATION.collect_dynamic_action_gap_data=false \
    EVALUATION.enable_internal_lora_branch=false \
    EVALUATION.enable_dynamic_action_gap=false \
    EVALUATION.enable_action_gap_schedule=false
}

run_dynamic_eval() {
  local suite="$1"
  require_file "$GATE_CKPT"
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${common_args[@]}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/dynamic_${suite}" \
    EVALUATION.collect_dynamic_action_gap_data=false \
    EVALUATION.enable_action_gap_schedule=false \
    EVALUATION.enable_dynamic_action_gap=true \
    "EVALUATION.dynamic_action_gap_checkpoint=${GATE_CKPT}" \
    EVALUATION.dynamic_action_gap_threshold=null \
    EVALUATION.dynamic_action_gap_max_internal_run=3
}

run_hybrid_eval() {
  local suite="$1"
  require_file "$GATE_CKPT"
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${common_args[@]}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/hybrid_${suite}" \
    EVALUATION.collect_dynamic_action_gap_data=false \
    EVALUATION.enable_action_gap_schedule=false \
    EVALUATION.enable_dynamic_action_gap=true \
    "EVALUATION.dynamic_action_gap_checkpoint=${GATE_CKPT}" \
    EVALUATION.dynamic_action_gap_threshold=null \
    EVALUATION.dynamic_action_gap_max_internal_run=3 \
    EVALUATION.dynamic_action_gap_anchor_action_gap=4
}

run_v3_eval() {
  local suite="$1"
  local method="$2"
  local max_internal_run="$3"
  local anchor_action_gap="$4"
  local threshold="$5"
  require_file "$GATE_CKPT"
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${common_args[@]}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/${method}_${suite}" \
    EVALUATION.collect_dynamic_action_gap_data=false \
    EVALUATION.enable_action_gap_schedule=false \
    EVALUATION.enable_dynamic_action_gap=true \
    "EVALUATION.dynamic_action_gap_checkpoint=${GATE_CKPT}" \
    "EVALUATION.dynamic_action_gap_threshold=${threshold}" \
    "EVALUATION.dynamic_action_gap_max_internal_run=${max_internal_run}" \
    "EVALUATION.dynamic_action_gap_anchor_action_gap=${anchor_action_gap}"
}

run_v3_sweep() {
  local suite
  for suite in libero_goal libero_spatial; do
    run_v3_eval "$suite" v3_safe 5 6 0.28
    run_v3_eval "$suite" v3_balanced 5 6 0.30
    run_v3_eval "$suite" v3_fast 7 8 0.32
  done
}

run_v4_sweep() {
  local suite
  for suite in libero_goal libero_spatial; do
    run_v3_eval "$suite" v4_t034 7 8 0.34
    run_v3_eval "$suite" v4_t036 7 8 0.36
  done
}

run_candidate() {
  local suite
  for suite in $CROSS_SUITES; do
    run_v3_eval \
      "$suite" \
      "$CANDIDATE_METHOD" \
      "$DYNAMIC_MAX_INTERNAL_RUN" \
      "$DYNAMIC_ANCHOR_ACTION_GAP" \
      "$DYNAMIC_THRESHOLD"
  done
}

run_compute_matched_eval() {
  local suite="$1"
  local method="$2"
  local mode="$3"
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    "${common_args[@]}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/${method}_${suite}" \
    EVALUATION.collect_dynamic_action_gap_data=false \
    EVALUATION.enable_action_gap_schedule=false \
    EVALUATION.enable_dynamic_action_gap=false \
    "EVALUATION.compute_matched_action_gap_mode=${mode}" \
    "EVALUATION.compute_matched_action_gap_target_internal_ratio=${MATCHED_TARGET_INTERNAL_RATIO}" \
    "EVALUATION.compute_matched_action_gap_seed=${MATCHED_RANDOM_SEED}" \
    "EVALUATION.dynamic_action_gap_max_internal_run=${DYNAMIC_MAX_INTERNAL_RUN}" \
    "EVALUATION.dynamic_action_gap_anchor_action_gap=${DYNAMIC_ANCHOR_ACTION_GAP}"
}

run_compute_matched() {
  local suite
  for suite in $CROSS_SUITES; do
    run_compute_matched_eval "$suite" fixed65 fixed
    run_compute_matched_eval "$suite" random65 random
  done
}

run_compute_matched_cross_task_validation() {
  local task_id
  for task_id in $CROSS_TASK_IDS; do
    echo "=== Compute-matched baselines: task ${task_id} ==="
    TASK_ID="$task_id" \
    NUM_TRIALS="$CROSS_NUM_TRIALS" \
    CROSS_SUITES="$CROSS_SUITES" \
    OUTPUT_ROOT="${OUTPUT_ROOT}/task${task_id}" \
    MATCHED_TARGET_INTERNAL_RATIO="$MATCHED_TARGET_INTERNAL_RATIO" \
    MATCHED_RANDOM_SEED="$MATCHED_RANDOM_SEED" \
    DYNAMIC_MAX_INTERNAL_RUN="$DYNAMIC_MAX_INTERNAL_RUN" \
    DYNAMIC_ANCHOR_ACTION_GAP="$DYNAMIC_ANCHOR_ACTION_GAP" \
      bash "$0" matched
  done
  for method in fixed65 random65; do
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_cross_task.py \
      "$OUTPUT_ROOT" \
      --task-ids $CROSS_TASK_IDS \
      --suites $CROSS_SUITES \
      --dynamic-method "$method" \
      --json-output "${OUTPUT_ROOT}/cross_task_summary_${method}.json"
  done
}

run_candidate_cross_task_validation() {
  local task_id
  for task_id in $CROSS_TASK_IDS; do
    echo "=== Candidate ${CANDIDATE_METHOD}: task ${task_id} ==="
    TASK_ID="$task_id" \
    NUM_TRIALS="$CROSS_NUM_TRIALS" \
    CROSS_SUITES="$CROSS_SUITES" \
    OUTPUT_ROOT="${OUTPUT_ROOT}/task${task_id}" \
    GATE_CKPT="$GATE_CKPT" \
    CANDIDATE_METHOD="$CANDIDATE_METHOD" \
    DYNAMIC_THRESHOLD="$DYNAMIC_THRESHOLD" \
    DYNAMIC_MAX_INTERNAL_RUN="$DYNAMIC_MAX_INTERNAL_RUN" \
    DYNAMIC_ANCHOR_ACTION_GAP="$DYNAMIC_ANCHOR_ACTION_GAP" \
      bash "$0" candidate
  done
  "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_cross_task.py \
    "$OUTPUT_ROOT" \
    --task-ids $CROSS_TASK_IDS \
    --suites $CROSS_SUITES \
    --dynamic-method "$CANDIDATE_METHOD" \
    --json-output "${OUTPUT_ROOT}/cross_task_summary_${CANDIDATE_METHOD}.json"
}

run_validation() {
  local suite
  for suite in $CROSS_SUITES; do
    run_fixed_eval "$suite"
    run_v3_eval "$suite" v4_t036 7 8 0.36
  done
}

run_full_suites() {
  local suite
  for suite in $CROSS_SUITES; do
    run_full_eval "$suite"
  done
}

run_cross_task_validation() {
  local task_id
  for task_id in $CROSS_TASK_IDS; do
    echo "=== Cross-task validation: task ${task_id} ==="
    TASK_ID="$task_id" \
    NUM_TRIALS="$CROSS_NUM_TRIALS" \
    CROSS_SUITES="$CROSS_SUITES" \
    OUTPUT_ROOT="${OUTPUT_ROOT}/task${task_id}" \
    GATE_CKPT="$GATE_CKPT" \
      bash "$0" validate
  done
  "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_cross_task.py \
    "$OUTPUT_ROOT" \
    --task-ids $CROSS_TASK_IDS \
    --suites $CROSS_SUITES \
    --json-output "${OUTPUT_ROOT}/cross_task_summary.json"
}

run_full_cross_task_validation() {
  local task_id
  for task_id in $CROSS_TASK_IDS; do
    echo "=== Full baseline: task ${task_id} ==="
    TASK_ID="$task_id" \
    NUM_TRIALS="$CROSS_NUM_TRIALS" \
    CROSS_SUITES="$CROSS_SUITES" \
    OUTPUT_ROOT="${OUTPUT_ROOT}/task${task_id}" \
      bash "$0" full
  done
  "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_cross_task.py \
    "$OUTPUT_ROOT" \
    --task-ids $CROSS_TASK_IDS \
    --suites $CROSS_SUITES \
    --json-output "${OUTPUT_ROOT}/cross_task_summary_with_full.json"
}

run_visual_diagnostics() {
  local suite
  for suite in $VIS_SUITES; do
    run_full_eval "$suite"
    run_fixed_eval "$suite"
    run_v3_eval "$suite" v4_t036 7 8 0.36
  done
  "$FASTWAM_PYTHON" scripts/summarize_action_jitter.py \
    "$OUTPUT_ROOT" \
    --task-id "$TASK_ID" \
    --suites $VIS_SUITES \
    --json-output "${OUTPUT_ROOT}/action_jitter_summary.json"
}

case "$STAGE" in
  full)
    run_full_suites
    ;;
  collect)
    run_collect libero_goal
    run_collect libero_spatial
    ;;
  train)
    run_train
    ;;
  train-meta)
    run_train_meta_only
    ;;
  eval)
    run_fixed_eval libero_goal
    run_dynamic_eval libero_goal
    run_fixed_eval libero_spatial
    run_dynamic_eval libero_spatial
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_smoke.py "$OUTPUT_ROOT" \
      --task-id "$TASK_ID" \
      --json-output "${OUTPUT_ROOT}/comparison_summary.json"
    ;;
  hybrid)
    run_hybrid_eval libero_goal
    run_hybrid_eval libero_spatial
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_smoke.py "$OUTPUT_ROOT" \
      --task-id "$TASK_ID" \
      --json-output "${OUTPUT_ROOT}/comparison_summary.json"
    ;;
  v3)
    run_v3_sweep
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_smoke.py "$OUTPUT_ROOT" \
      --task-id "$TASK_ID" \
      --json-output "${OUTPUT_ROOT}/comparison_summary.json"
    ;;
  v4)
    run_v4_sweep
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_smoke.py "$OUTPUT_ROOT" \
      --task-id "$TASK_ID" \
      --json-output "${OUTPUT_ROOT}/comparison_summary.json"
    ;;
  candidate)
    run_candidate
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_smoke.py "$OUTPUT_ROOT" \
      --task-id "$TASK_ID" \
      --json-output "${OUTPUT_ROOT}/comparison_summary.json"
    ;;
  candidate-cross)
    run_candidate_cross_task_validation
    ;;
  matched)
    run_compute_matched
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_smoke.py "$OUTPUT_ROOT" \
      --task-id "$TASK_ID" \
      --suites $CROSS_SUITES \
      --json-output "${OUTPUT_ROOT}/comparison_summary.json"
    ;;
  matched-cross)
    run_compute_matched_cross_task_validation
    ;;
  cross)
    run_cross_task_validation
    ;;
  full-cross)
    run_full_cross_task_validation
    ;;
  visualize)
    run_visual_diagnostics
    ;;
  validate)
    run_validation
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_smoke.py "$OUTPUT_ROOT" \
      --task-id "$TASK_ID" \
      --suites $CROSS_SUITES \
      --json-output "${OUTPUT_ROOT}/comparison_summary.json"
    ;;
  all)
    run_collect libero_goal
    run_collect libero_spatial
    run_train
    run_fixed_eval libero_goal
    run_dynamic_eval libero_goal
    run_fixed_eval libero_spatial
    run_dynamic_eval libero_spatial
    run_hybrid_eval libero_goal
    run_hybrid_eval libero_spatial
    run_v3_sweep
    "$FASTWAM_PYTHON" scripts/summarize_dynamic_action_gap_smoke.py "$OUTPUT_ROOT" \
      --task-id "$TASK_ID" \
      --json-output "${OUTPUT_ROOT}/comparison_summary.json"
    ;;
  *)
    echo "Usage: $0 [full|collect|train|train-meta|eval|hybrid|v3|v4|candidate|candidate-cross|matched|matched-cross|validate|cross|full-cross|visualize|all]" >&2
    exit 2
    ;;
esac

echo "Dynamic ActionGap smoke stage '${STAGE}' completed. Results: ${OUTPUT_ROOT}"
