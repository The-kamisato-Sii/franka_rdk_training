#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export DREAMZERO_PATH=${DREAMZERO_PATH:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/dreamzero}
export PYTHONPATH="${REPO_PATH}:${DREAMZERO_PATH}:${PYTHONPATH:-}"
export EMBODIED_PATH=${EMBODIED_PATH:-"${REPO_PATH}"}

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

detect_num_gpus() {
  local detected
  detected=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d " ")
  if [ -z "${detected}" ] || [ "${detected}" = "0" ]; then
    detected=8
  fi
  echo "${detected}"
}

DETECTED_GPUS=$(detect_num_gpus)
NUM_GPUS=${NUM_GPUS:-8}
if [ "${NUM_GPUS}" -gt "${DETECTED_GPUS}" ]; then
  echo "[RLinf] requested NUM_GPUS=${NUM_GPUS}, but only detected ${DETECTED_GPUS}; using detected value."
  NUM_GPUS="${DETECTED_GPUS}"
fi
NNODES=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-${RAY_HEAD_PORT:-6379}}
export NUM_GPUS NNODES NODE_RANK MASTER_ADDR MASTER_PORT
CONFIG=${CONFIG:-real_world_franka_dual_wmam_motion_sft}
export REAL_WORLD_FRANKA_DUAL_MOTION_ROOT=${REAL_WORLD_FRANKA_DUAL_MOTION_ROOT:-/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_motion}
export REAL_WORLD_FRANKA_DUAL_MOTION_DIR_NAME=${REAL_WORLD_FRANKA_DUAL_MOTION_DIR_NAME:-motions_sam}
export REAL_WORLD_JOINT_TASK_NAME_CONTAINS=${REAL_WORLD_JOINT_TASK_NAME_CONTAINS:-arrange_vegetables}
export REAL_WORLD_JOINT_MAX_TASKS=${REAL_WORLD_JOINT_MAX_TASKS:-1}
export RUN_LOG_PATH=${RUN_LOG_PATH:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/results_franka_dual_wmam_debug}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-real_world_franka_dual_wmam_motion_sft_debug}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
export ACTOR_MODEL_PRECISION=${ACTOR_MODEL_PRECISION:-fp32}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
  export CUDA_VISIBLE_DEVICES
fi

TOTAL_GPUS=${NUM_GPUS}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / TOTAL_GPUS))}
RAY_PORT=${RAY_PORT:-${RAY_HEAD_PORT:-29500}}
RAY_HEAD_ADDR=${RAY_HEAD_ADDR:-${MASTER_ADDR}}
RAY_ADDRESS="${RAY_HEAD_ADDR}:${RAY_PORT}"
RLINF_RUN_ID=${RLINF_RUN_ID:-${CONFIG}_${RAY_HEAD_ADDR}_${RAY_PORT}_${NNODES}_${NUM_GPUS}_debug}
RLINF_RUN_ID_SAFE=$(printf "%s" "${RLINF_RUN_ID}" | tr -c "[:alnum:]_.-" "_")
RAY_DONE_FILE=${RAY_DONE_FILE:-"${REPO_PATH}/ray_utils/ray_done_${RLINF_RUN_ID_SAFE}"}
export RLINF_NODE_RANK=0

start_single_node_ray() {
  if [ "${RLINF_START_RAY:-1}" != "1" ]; then
    echo "[RLinf] RLINF_START_RAY=0: assuming local Ray is already started."
    return
  fi
  ray stop --force >/dev/null 2>&1 || true
  local ray_args=(--num-gpus="${NUM_GPUS}")
  if [ -n "${RAY_MEMORY_BYTES:-}" ]; then
    ray_args+=(--memory="${RAY_MEMORY_BYTES}")
  fi
  rm -f "${RAY_DONE_FILE}"
  local node_ip
  node_ip=${RAY_NODE_IP:-$(hostname -I 2>/dev/null | cut -d " " -f 1)}
  node_ip=${node_ip:-127.0.0.1}
  RAY_HEAD_ADDR="${node_ip}"
  RAY_ADDRESS="${RAY_HEAD_ADDR}:${RAY_PORT}"
  echo "[RLinf] Starting single-node Ray: node_ip=${node_ip} port=${RAY_PORT} num_gpus=${NUM_GPUS}"
  ray start --head --node-ip-address="${node_ip}" --port="${RAY_PORT}" --include-dashboard=false "${ray_args[@]}"
}

finish_head() {
  local status=$?
  if [ "${RLINF_START_RAY:-1}" = "1" ]; then
    touch "${RAY_DONE_FILE}" 2>/dev/null || true
    if [ "${RLINF_STOP_RAY_ON_EXIT:-1}" = "1" ]; then
      ray stop --force >/dev/null 2>&1 || true
    fi
  fi
  return "${status}"
}
trap finish_head EXIT

start_single_node_ray

if [ "${TOTAL_GPUS}" = "1" ]; then
  ACTOR_PLACEMENT=${ACTOR_PLACEMENT:-0}
else
  ACTOR_PLACEMENT=${ACTOR_PLACEMENT:-0-$((TOTAL_GPUS - 1))}
fi

echo "[RLinf] single-node NUM_GPUS=${NUM_GPUS} RAY_ADDRESS=${RAY_ADDRESS} ACTOR_PLACEMENT=${ACTOR_PLACEMENT} CONFIG=${CONFIG}"
echo "[RLinf] REAL_WORLD_FRANKA_DUAL_MOTION_ROOT=${REAL_WORLD_FRANKA_DUAL_MOTION_ROOT}"
echo "[RLinf] REAL_WORLD_FRANKA_DUAL_MOTION_DIR_NAME=${REAL_WORLD_FRANKA_DUAL_MOTION_DIR_NAME}"
echo "[RLinf] REAL_WORLD_JOINT_TASK_NAME_CONTAINS=${REAL_WORLD_JOINT_TASK_NAME_CONTAINS} REAL_WORLD_JOINT_MAX_TASKS=${REAL_WORLD_JOINT_MAX_TASKS}"

if [ "${RLINF_WAIT_FOR_RAY_GPUS:-1}" = "1" ]; then
  bash "${REPO_PATH}/ray_utils/check_ray.sh" "${TOTAL_GPUS}"
fi

extra_args=(
  runner.logger.log_path="${RUN_LOG_PATH}"
  runner.logger.experiment_name="${EXPERIMENT_NAME}"
  actor.global_batch_size="${GLOBAL_BATCH_SIZE}"
  actor.micro_batch_size="${MICRO_BATCH_SIZE}"
  actor.model.precision="${ACTOR_MODEL_PRECISION}"
)
if [ -n "${RESUME_DIR:-}" ]; then
  extra_args+=(runner.resume_dir="${RESUME_DIR}")
fi

cd "${REPO_PATH}"
python examples/sft/train_vla_sft.py --config-name "${CONFIG}" cluster.num_nodes="${NNODES}" cluster.component_placement.actor="${ACTOR_PLACEMENT}" "${extra_args[@]}" "$@"
