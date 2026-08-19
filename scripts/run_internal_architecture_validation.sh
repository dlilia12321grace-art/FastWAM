#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-pilot}"
CANDIDATES="${CANDIDATES:-2:0 2:4 12:0 16:4 4:1}"
REFERENCE_CANDIDATE="${REFERENCE_CANDIDATE:-4:1}"
SOURCE_END_LAYER="${SOURCE_END_LAYER:-30}"
TRAIN_TASK_ID="${TRAIN_TASK_ID:-0}"
TRAIN_SUITE="${TRAIN_SUITE:-libero_spatial}"
TRAIN_MAX_UPDATES="${TRAIN_MAX_UPDATES:-100}"
TRAIN_NUM_TRIALS="${TRAIN_NUM_TRIALS:-5}"
TRAIN_STEPS="${TRAIN_STEPS:-[0,1,2,3,4,5,6]}"
VALIDATION_TAG="${VALIDATION_TAG:-pilot}"
VALIDATION_TASK_IDS="${VALIDATION_TASK_IDS:-1 2 3 4}"
VALIDATION_SUITES="${VALIDATION_SUITES:-libero_goal libero_spatial}"
VALIDATION_NUM_TRIALS="${VALIDATION_NUM_TRIALS:-5}"
ACTION_GAP="${ACTION_GAP:-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/root/autodl-tmp/envs/fastwam/bin/python}"
FASTWAM_CKPT="${FASTWAM_CKPT:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
FASTWAM_STATS="${FASTWAM_STATS:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/evaluate_results/internal_architecture_pareto}"

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

mkdir -p "$OUTPUT_ROOT/checkpoints" "$OUTPUT_ROOT/train" "$OUTPUT_ROOT/$VALIDATION_TAG"

parse_candidate() {
  local candidate="$1"
  if [[ ! "$candidate" =~ ^[0-9]+:[0-9]+$ ]]; then
    echo "Invalid candidate '$candidate'; expected FORK:BLOCKS (for example 2:0)." >&2
    exit 2
  fi
  CANDIDATE_FORK="${candidate%%:*}"
  CANDIDATE_BLOCKS="${candidate##*:}"
  if (( CANDIDATE_FORK < 1 || CANDIDATE_FORK > SOURCE_END_LAYER )); then
    echo "Invalid fork layer $CANDIDATE_FORK for source end $SOURCE_END_LAYER." >&2
    exit 2
  fi
  if (( CANDIDATE_BLOCKS == 0 )); then
    CANDIDATE_SOURCE_START="$SOURCE_END_LAYER"
    CANDIDATE_SOURCE_END=$((SOURCE_END_LAYER - 1))
  else
    CANDIDATE_SOURCE_START=$((SOURCE_END_LAYER - CANDIDATE_BLOCKS + 1))
    CANDIDATE_SOURCE_END="$SOURCE_END_LAYER"
  fi
  if (( CANDIDATE_SOURCE_START < 1 )); then
    echo "Candidate '$candidate' requests too many internal blocks." >&2
    exit 2
  fi
  CANDIDATE_METHOD="fork${CANDIDATE_FORK}_b${CANDIDATE_BLOCKS}"
}

train_one() {
  local candidate="$1"
  parse_candidate "$candidate"
  local checkpoint="$OUTPUT_ROOT/checkpoints/${CANDIDATE_METHOD}.pt"
  local best_checkpoint="$OUTPUT_ROOT/checkpoints/${CANDIDATE_METHOD}.best.pt"
  if [[ -f "$best_checkpoint" ]]; then
    echo "[skip train] $CANDIDATE_METHOD: $best_checkpoint already exists"
    return
  fi
  echo "=== Formal train $CANDIDATE_METHOD ($candidate) ==="
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    task=libero_uncond_2cam224_1e-4 \
    "ckpt=${FASTWAM_CKPT}" \
    gpu_id=0 \
    "EVALUATION.task_id=${TRAIN_TASK_ID}" \
    "EVALUATION.num_trials=${TRAIN_NUM_TRIALS}" \
    "EVALUATION.task_suite_name=${TRAIN_SUITE}" \
    "EVALUATION.dataset_stats_path=${FASTWAM_STATS}" \
    "EVALUATION.output_dir=${OUTPUT_ROOT}/train/${CANDIDATE_METHOD}" \
    EVALUATION.save_rollout_video=false \
    EVALUATION.save_action_trace=false \
    EVALUATION.profile_timing=true \
    EVALUATION.enable_internal_lora_branch=false \
    +EVALUATION.train_internal_lora=true \
    "+EVALUATION.internal_lora_train_steps=${TRAIN_STEPS}" \
    "+EVALUATION.internal_lora_max_updates=${TRAIN_MAX_UPDATES}" \
    +EVALUATION.internal_lora_learning_rate=0.0001 \
    +EVALUATION.internal_lora_weight_decay=0.0 \
    "+EVALUATION.internal_lora_output_checkpoint=${checkpoint}" \
    "EVALUATION.internal_lora_fork_layer=${CANDIDATE_FORK}" \
    "EVALUATION.internal_lora_source_start_layer=${CANDIDATE_SOURCE_START}" \
    "EVALUATION.internal_lora_source_end_layer=${CANDIDATE_SOURCE_END}" \
    "EVALUATION.internal_lora_rank=${LORA_RANK}" \
    "EVALUATION.internal_lora_alpha=${LORA_ALPHA}"
}

validate_one() {
  local candidate="$1"
  local task_id="$2"
  local suite="$3"
  parse_candidate "$candidate"
  local checkpoint="$OUTPUT_ROOT/checkpoints/${CANDIDATE_METHOD}.best.pt"
  local task_root="$OUTPUT_ROOT/$VALIDATION_TAG/task${task_id}"
  local result="$task_root/${CANDIDATE_METHOD}_${suite}/${suite}/gpu0_task${task_id}_results.json"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing formal checkpoint: $checkpoint" >&2
    exit 2
  fi
  if [[ -f "$result" ]]; then
    echo "[skip validation] $CANDIDATE_METHOD task${task_id} $suite"
    return
  fi
  echo "=== Validate $CANDIDATE_METHOD: task${task_id} $suite ==="
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    task=libero_uncond_2cam224_1e-4 \
    "ckpt=${FASTWAM_CKPT}" \
    gpu_id=0 \
    "EVALUATION.task_id=${task_id}" \
    "EVALUATION.num_trials=${VALIDATION_NUM_TRIALS}" \
    "EVALUATION.task_suite_name=${suite}" \
    "EVALUATION.dataset_stats_path=${FASTWAM_STATS}" \
    "EVALUATION.output_dir=${task_root}/${CANDIDATE_METHOD}_${suite}" \
    EVALUATION.save_rollout_video=false \
    EVALUATION.save_action_trace=false \
    EVALUATION.profile_timing=true \
    EVALUATION.enable_internal_lora_branch=true \
    "EVALUATION.internal_lora_checkpoint=${checkpoint}" \
    "EVALUATION.internal_lora_fork_layer=${CANDIDATE_FORK}" \
    "EVALUATION.internal_lora_source_start_layer=${CANDIDATE_SOURCE_START}" \
    "EVALUATION.internal_lora_source_end_layer=${CANDIDATE_SOURCE_END}" \
    "EVALUATION.internal_lora_rank=${LORA_RANK}" \
    "EVALUATION.internal_lora_alpha=${LORA_ALPHA}" \
    EVALUATION.enable_action_gap_schedule=true \
    "EVALUATION.action_gap=${ACTION_GAP}" \
    EVALUATION.enable_dynamic_action_gap=false \
    EVALUATION.collect_dynamic_action_gap_data=false
}

run_train() {
  local candidate
  for candidate in $CANDIDATES; do
    train_one "$candidate"
  done
}

run_validate() {
  local candidate task_id suite
  for candidate in $CANDIDATES; do
    for task_id in $VALIDATION_TASK_IDS; do
      for suite in $VALIDATION_SUITES; do
        validate_one "$candidate" "$task_id" "$suite"
      done
    done
  done
}

run_summarize() {
  "$FASTWAM_PYTHON" scripts/summarize_internal_architecture_validation.py \
    "$OUTPUT_ROOT" \
    --validation-tag "$VALIDATION_TAG" \
    --candidates $CANDIDATES \
    --reference-candidate "$REFERENCE_CANDIDATE" \
    --task-ids $VALIDATION_TASK_IDS \
    --suites $VALIDATION_SUITES \
    --train-task-id "$TRAIN_TASK_ID" \
    --train-suite "$TRAIN_SUITE" \
    --json-output "$OUTPUT_ROOT/$VALIDATION_TAG/summary.json" \
    --csv-output "$OUTPUT_ROOT/$VALIDATION_TAG/summary.csv"
}

case "$STAGE" in
  train) run_train ;;
  validate) run_validate ;;
  summarize) run_summarize ;;
  pilot|all)
    run_train
    run_validate
    run_summarize
    ;;
  *)
    echo "Usage: $0 [train|validate|summarize|pilot|all]" >&2
    exit 2
    ;;
esac

echo "Internal architecture validation stage '$STAGE' completed: $OUTPUT_ROOT/$VALIDATION_TAG"
