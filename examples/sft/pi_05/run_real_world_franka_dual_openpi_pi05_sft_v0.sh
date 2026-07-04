#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export EMBODIED_PATH=${EMBODIED_PATH:-"${REPO_PATH}/examples/embodiment"}
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"${REPO_PATH}/checkpoints/openpi_cache"}
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
# Avoid the NCCL NVLS path that fails on this cluster during OpenPI FSDP init
# with "transport/nvls.cc: Cuda failure 1 'invalid argument'".
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

CONFIG=${CONFIG:-real_world_franka_dual_openpi_pi05_sft_v0}
detect_num_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d " "
  else
    echo 1
  fi
}

NUM_GPUS=${NUM_GPUS:-$(detect_num_gpus)}
NNODES=${NNODES:-1}
TOTAL_GPUS=$((NNODES * NUM_GPUS))
export NUM_GPUS NNODES
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / TOTAL_GPUS))}


if [ -z "${RANK:-}" ]; then
  export RANK=0
fi

if [ "${RLINF_RESET_RAY:-0}" = "1" ]; then
  echo "[RLinf] RLINF_RESET_RAY=1: stopping stale local Ray processes first."
  ray stop --force || true
fi

unset RAY_ADDRESS

if [ "${RLINF_START_RAY:-0}" = "1" ]; then
  bash ray_utils/start_ray.sh
else
  echo "[RLinf] RLINF_START_RAY=${RLINF_START_RAY:-0}: letting RLinf create/connect Ray."
fi

echo "[RLinf] REPO_PATH=${REPO_PATH}"
echo "[RLinf] EMBODIED_PATH=${EMBODIED_PATH}"
echo "[RLinf] OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "[RLinf] RLINF_RESET_RAY=${RLINF_RESET_RAY:-0}"
echo "[RLinf] CONFIG=${CONFIG}"
echo "[RLinf] GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} TOTAL_GPUS=${TOTAL_GPUS}"
echo "[RLinf] NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE}"
echo "[RLinf] NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "[RLinf] NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE}"
python examples/sft/train_vla_sft.py --config-name "${CONFIG}" \
  actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  actor.micro_batch_size="${MICRO_BATCH_SIZE}" \
  "$@"
