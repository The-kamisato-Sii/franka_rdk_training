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
  detected=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  if [ -z "${detected}" ] || [ "${detected}" = "0" ]; then
    detected=8
  fi
  echo "${detected}"
}

PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-${NUM_GPUS:-$(detect_num_gpus)}}
PET_NNODES=${PET_NNODES:-${NNODES:-1}}
PET_NODE_RANK=${PET_NODE_RANK:-${NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-${RAY_HEAD_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}}
MASTER_PORT=${MASTER_PORT:-${RAY_HEAD_PORT:-${PET_MASTER_PORT:-6379}}}
export PET_NPROC_PER_NODE PET_NNODES PET_NODE_RANK MASTER_ADDR MASTER_PORT

NUM_GPUS=${NUM_GPUS:-${PET_NPROC_PER_NODE}}
NNODES=${NNODES:-${PET_NNODES}}
NODE_RANK=${NODE_RANK:-${PET_NODE_RANK}}
export NUM_GPUS NNODES NODE_RANK
CONFIG=${CONFIG:-real_world_franka_dual_dreamzero_sft_debug}
export REAL_WORLD_FRANKA_DUAL_ROOT=${REAL_WORLD_FRANKA_DUAL_ROOT:-/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_test}
export RUN_LOG_PATH=${RUN_LOG_PATH:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/results_franka_dual}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-real_world_franka_dual_dreamzero_sft_arrange_vegetables}
export MAX_STEPS=${MAX_STEPS:-20000}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${MAX_STEPS}}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
export LR=${LR:-1e-5}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-1000}

ACTOR_MODEL_PRECISION=${ACTOR_MODEL_PRECISION:-fp32}
export ACTOR_MODEL_PRECISION

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
  export CUDA_VISIBLE_DEVICES
fi

TOTAL_GPUS=$((NNODES * NUM_GPUS))
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / TOTAL_GPUS))}
RAY_PORT=${RAY_PORT:-${RAY_HEAD_PORT:-29500}}
RAY_HEAD_ADDR=${RAY_HEAD_ADDR:-${MASTER_ADDR}}
RAY_ADDRESS="${RAY_HEAD_ADDR}:${RAY_PORT}"
RLINF_RUN_ID=${RLINF_RUN_ID:-${SLURM_JOB_ID:-${PBS_JOBID:-${LSB_JOBID:-${JOB_ID:-${CONFIG}_${RAY_HEAD_ADDR}_${RAY_PORT}_${NNODES}_${NUM_GPUS}}}}}}
RLINF_RUN_ID_SAFE=$(printf "%s" "${RLINF_RUN_ID}" | tr -c "[:alnum:]_.-" "_")
RAY_DONE_FILE=${RAY_DONE_FILE:-"${REPO_PATH}/ray_utils/ray_done_${RLINF_RUN_ID_SAFE}"}
export RLINF_NODE_RANK=${RLINF_NODE_RANK:-${NODE_RANK}}

start_ray_for_rank() {
  if [ "${RLINF_START_RAY:-1}" != "1" ]; then
    echo "[RLinf] RLINF_START_RAY=0: assuming Ray cluster is already started."
    if [ "${NODE_RANK}" != "0" ]; then
      echo "[RLinf] external Ray mode: driver runs only on node rank 0; exiting worker shell."
      exit 0
    fi
    return
  fi

  ray stop --force >/dev/null 2>&1 || true
  local ray_args=(--num-gpus="${NUM_GPUS}")
  if [ -n "${RAY_MEMORY_BYTES:-}" ]; then
    ray_args+=(--memory="${RAY_MEMORY_BYTES}")
  fi

  if [ "${NODE_RANK}" = "0" ]; then
    rm -f "${RAY_DONE_FILE}"
    local node_ip
    node_ip=${RAY_NODE_IP:-$(hostname -I 2>/dev/null | cut -d " " -f 1)}
    node_ip=${node_ip:-${RAY_HEAD_ADDR}}
    echo "[RLinf] Starting Ray head: node_rank=${NODE_RANK} node_ip=${node_ip} port=${RAY_PORT}"
    ray start --head --node-ip-address="${node_ip}" --port="${RAY_PORT}" --include-dashboard=false "${ray_args[@]}"
  else
    echo "[RLinf] Starting Ray worker: node_rank=${NODE_RANK} address=${RAY_ADDRESS}"
    local started=0
    for _ in $(seq 1 360); do
      if ray start --address="${RAY_ADDRESS}" "${ray_args[@]}"; then
        started=1
        break
      fi
      sleep 2
    done
    if [ "${started}" != "1" ]; then
      echo "[RLinf] ERROR: Ray worker node_rank=${NODE_RANK} failed to connect to ${RAY_ADDRESS}" >&2
      exit 1
    fi
    while [ ! -f "${RAY_DONE_FILE}" ]; do
      sleep 30
    done
    ray stop --force >/dev/null 2>&1 || true
    exit 0
  fi
}

finish_head() {
  local status=$?
  if [ "${NODE_RANK}" = "0" ] && [ "${RLINF_START_RAY:-1}" = "1" ]; then
    touch "${RAY_DONE_FILE}" 2>/dev/null || true
    if [ "${RLINF_STOP_RAY_ON_EXIT:-1}" = "1" ]; then
      ray stop --force >/dev/null 2>&1 || true
    fi
  fi
  return "${status}"
}
trap finish_head EXIT

start_ray_for_rank

if [ "${TOTAL_GPUS}" = "1" ]; then
  ACTOR_PLACEMENT=${ACTOR_PLACEMENT:-0}
else
  ACTOR_PLACEMENT=${ACTOR_PLACEMENT:-0-$((TOTAL_GPUS - 1))}
fi

echo "[RLinf] NUM_GPUS=${NUM_GPUS} NNODES=${NNODES} NODE_RANK=${NODE_RANK} RAY_ADDRESS=${RAY_ADDRESS} ACTOR_PLACEMENT=${ACTOR_PLACEMENT} CONFIG=${CONFIG}"
echo "[RLinf] REAL_WORLD_FRANKA_DUAL_ROOT=${REAL_WORLD_FRANKA_DUAL_ROOT}"
echo "[RLinf] precision override: actor.model.precision=${ACTOR_MODEL_PRECISION}"
echo "[RLinf] outputs: ${RUN_LOG_PATH}/${EXPERIMENT_NAME}/{tensorboard,checkpoints}"
echo "[RLinf] max_steps=${MAX_STEPS} global_batch_size=${GLOBAL_BATCH_SIZE} micro_batch_size=${MICRO_BATCH_SIZE} total_gpus=${TOTAL_GPUS} lr=${LR} warmup=${LR_WARMUP_STEPS}"

if [ ! -f "${REAL_WORLD_FRANKA_DUAL_ROOT}/stats.json" ] || [ ! -f "${REAL_WORLD_FRANKA_DUAL_ROOT}/norm_stats.json" ]; then
  echo "[RLinf] ERROR: ${REAL_WORLD_FRANKA_DUAL_ROOT} is missing stats.json or norm_stats.json." >&2
  exit 1
fi

if [ "${RLINF_WAIT_FOR_RAY_GPUS:-1}" = "1" ]; then
  bash "${REPO_PATH}/ray_utils/check_ray.sh" "${TOTAL_GPUS}"
fi

extra_args=(
  runner.logger.log_path="${RUN_LOG_PATH}"
  runner.logger.experiment_name="${EXPERIMENT_NAME}"
  runner.max_steps="${MAX_STEPS}"
  actor.model.precision="${ACTOR_MODEL_PRECISION}"
  actor.optim.total_training_steps="${TOTAL_TRAINING_STEPS}"
  actor.optim.lr="${LR}"
  actor.optim.lr_scheduler=cosine
  actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}"
  actor.global_batch_size="${GLOBAL_BATCH_SIZE}"
  actor.micro_batch_size="${MICRO_BATCH_SIZE}"
)
if [ -n "${RESUME_DIR:-}" ]; then
  extra_args+=(runner.resume_dir="${RESUME_DIR}")
fi

cd "${REPO_PATH}"
python examples/sft/train_vla_sft.py --config-name "${CONFIG}" cluster.num_nodes="${NNODES}" cluster.component_placement.actor="${ACTOR_PLACEMENT}" "${extra_args[@]}" "$@"
