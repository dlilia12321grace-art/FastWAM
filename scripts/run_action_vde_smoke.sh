#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-both}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-1}"
VDE_WARMUP="${VDE_WARMUP:-4}"
VDE_INTERVAL="${VDE_INTERVAL:-2}"
FASTWAM_PYTHON="${FASTWAM_PYTHON:-/root/autodl-tmp/envs/fastwam/bin/python}"
FASTWAM_CKPT="${FASTWAM_CKPT:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
FASTWAM_STATS="${FASTWAM_STATS:-/root/autodl-tmp/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/evaluate_results/fastwam_action_vde_smoke}"

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

run_one() {
  local name="$1"
  local enabled="$2"
  local output_dir="$OUTPUT_ROOT/$name"
  echo "=== $name: $TASK_SUITE task$TASK_ID, trials=$NUM_TRIALS ==="
  "$FASTWAM_PYTHON" experiments/libero/eval_libero_single.py \
    task=libero_uncond_2cam224_1e-4 \
    "ckpt=${FASTWAM_CKPT}" \
    gpu_id=0 \
    "EVALUATION.task_suite_name=${TASK_SUITE}" \
    "EVALUATION.task_id=${TASK_ID}" \
    "EVALUATION.num_trials=${NUM_TRIALS}" \
    "EVALUATION.dataset_stats_path=${FASTWAM_STATS}" \
    "EVALUATION.output_dir=${output_dir}" \
    EVALUATION.save_rollout_video=true \
    EVALUATION.save_action_trace=false \
    EVALUATION.profile_timing=true \
    EVALUATION.enable_video_gap_schedule=false \
    EVALUATION.enable_dynamic_video_gap=false \
    EVALUATION.enable_c3cache=false \
    EVALUATION.enable_internal_lora_branch=false \
    EVALUATION.enable_action_gap_schedule=false \
    EVALUATION.enable_dynamic_action_gap=false \
    EVALUATION.enable_oracle_action_gap=false \
    EVALUATION.collect_dynamic_action_gap_data=false \
    "EVALUATION.enable_action_vde=${enabled}" \
    "EVALUATION.action_vde_warmup_steps=${VDE_WARMUP}" \
    "EVALUATION.action_vde_anchor_interval=${VDE_INTERVAL}"
}

case "$MODE" in
  baseline) run_one full false ;;
  vde) run_one action_vde true ;;
  both) run_one full false; run_one action_vde true ;;
  *) echo "Unknown mode '$MODE'; expected baseline, vde, or both." >&2; exit 2 ;;
esac
