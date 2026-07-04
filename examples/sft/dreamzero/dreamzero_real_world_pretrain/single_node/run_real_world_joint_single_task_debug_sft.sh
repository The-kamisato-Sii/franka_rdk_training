#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to the problematic Robocoin task, while keeping the script reusable.
DEBUG_TASK_TAG=${DEBUG_TASK_TAG:-robocoin_ruantong_a2d_34d}
DEBUG_TASK_NAME=${DEBUG_TASK_NAME:-AgiBot-g1_battery_storage_b}
DEBUG_MAX_TASKS=${DEBUG_MAX_TASKS:-1}

export REAL_WORLD_DATA_ROOT=${REAL_WORLD_DATA_ROOT:-/inspire/qb-ilm2/project/robot-body/public/real_world_data_dreamzero_motion}
export CONFIG=${CONFIG:-real_world_joint_sft_dreamzero_motion_5b}

detect_num_gpus() {
  local detected
  detected=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  if [ -z "${detected}" ] || [ "${detected}" = "0" ]; then
    detected=8
  fi
  echo "${detected}"
}

NUM_GPUS=${NUM_GPUS:-$(detect_num_gpus)}
export NUM_GPUS

# Small defaults make loss spikes easier to localize. Override these env vars
# or append Hydra overrides after the script when you want production settings.
DEBUG_MICRO_BATCH_SIZE=${DEBUG_MICRO_BATCH_SIZE:-1}
DEBUG_GLOBAL_BATCH_SIZE=${DEBUG_GLOBAL_BATCH_SIZE:-${NUM_GPUS}}
DEBUG_MAX_STEPS=${DEBUG_MAX_STEPS:-10000}
DEBUG_LOG_INTERVAL=${DEBUG_LOG_INTERVAL:-1}
DEBUG_SAVE_INTERVAL=${DEBUG_SAVE_INTERVAL:-1000000}
DEBUG_NUM_WORKERS=${DEBUG_NUM_WORKERS:-0}

echo "[RLinf][debug] task_tag=${DEBUG_TASK_TAG}"
echo "[RLinf][debug] task_name_contains=${DEBUG_TASK_NAME}"
echo "[RLinf][debug] config=${CONFIG} num_gpus=${NUM_GPUS} micro_batch=${DEBUG_MICRO_BATCH_SIZE} global_batch=${DEBUG_GLOBAL_BATCH_SIZE}"

exec bash "${SCRIPT_DIR}/run_real_world_joint_dreamzero_motion_sft.sh" \
  data.real_world_joint_tags="${DEBUG_TASK_TAG}" \
  data.real_world_joint_task_name_contains="${DEBUG_TASK_NAME}" \
  data.real_world_joint_max_tasks="${DEBUG_MAX_TASKS}" \
  data.num_workers="${DEBUG_NUM_WORKERS}" \
  runner.log_interval="${DEBUG_LOG_INTERVAL}" \
  runner.save_interval="${DEBUG_SAVE_INTERVAL}" \
  runner.max_steps="${DEBUG_MAX_STEPS}" \
  actor.micro_batch_size="${DEBUG_MICRO_BATCH_SIZE}" \
  actor.global_batch_size="${DEBUG_GLOBAL_BATCH_SIZE}" \
  "$@"
