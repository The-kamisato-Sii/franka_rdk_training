#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
export EMBODIED_PATH=${EMBODIED_PATH:-"${REPO_PATH}/examples/embodiment"}
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"${REPO_PATH}/checkpoints/openpi_cache"}
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

CONFIG=${CONFIG:-openpi_v2}

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
echo "[RLinf] REAL_WORLD_FRANKA_DUAL_ROOT=${REAL_WORLD_FRANKA_DUAL_ROOT:-/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_test}"
echo "[RLinf] CONFIG=${CONFIG}"
python examples/sft/train_vla_sft.py --config-name "${CONFIG}" "$@"
