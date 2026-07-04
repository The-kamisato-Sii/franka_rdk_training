#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export DREAMZERO_PATH=${DREAMZERO_PATH:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/dreamzero}
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH:-}"
export EMBODIED_PATH=${EMBODIED_PATH:-$(pwd)}

# DreamZero/RLinf training is PyTorch-only here. Prevent Transformers from
# importing TensorFlow/JAX in every Ray worker, which slows startup and emits
# noisy cuDNN/cuBLAS factory warnings.
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ "${NUM_GPUS}" = "0" ]; then
  NUM_GPUS=8
fi
CONFIG=${CONFIG:-real_world_joint_sft_dreamzero_motion_5b}

# RLinf launches one Ray driver process; the driver creates FSDP workers as Ray
# actors and sets MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE for them internally.
# Do not wrap this entrypoint with torchrun, otherwise each torchrun process
# starts its own Ray worker group and c10d rendezvous can hang.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
  export CUDA_VISIBLE_DEVICES
fi

if [ "${NUM_GPUS}" = "1" ]; then
  ACTOR_PLACEMENT=${ACTOR_PLACEMENT:-0}
else
  ACTOR_PLACEMENT=${ACTOR_PLACEMENT:-0-$((NUM_GPUS - 1))}
fi

extra_args=()
if [ -n "${RESUME_DIR:-}" ]; then
  extra_args+=(runner.resume_dir="${RESUME_DIR}")
fi

python examples/sft/train_vla_sft.py --config-name "${CONFIG}" cluster.component_placement.actor="${ACTOR_PLACEMENT}" "${extra_args[@]}" "$@"
