#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CKPT=${CKPT:-}
export INIT_MODEL_STATE_DICT_PATH=${INIT_MODEL_STATE_DICT_PATH:-${CKPT}}

usage() {
  cat >&2 <<USAGE
Usage:
  $0 [hydra overrides...]
  $0 /path/to/full_weights.pt [hydra overrides...]
  CKPT=/path/to/full_weights.pt $0 [hydra overrides...]
  INIT_MODEL_STATE_DICT_PATH=/path/to/full_weights.pt $0 [hydra overrides...]

Without an explicit checkpoint path, actor.model.init_model_state_dict_path stays null
and training starts from the configured base Wan2.2/DreamZero weights.

The path may also be a checkpoint directory containing one of:
  full_weights.pt
  actor/model_state_dict/full_weights.pt
  model_state_dict/full_weights.pt
USAGE
}

abs_file() {
  local path="$1"
  local dir
  dir="$(cd "$(dirname "${path}")" && pwd -P)"
  printf "%s/%s\n" "${dir}" "$(basename "${path}")"
}

resolve_pretrained_path() {
  local input="$1"
  if [ -f "${input}" ]; then
    abs_file "${input}"
    return 0
  fi
  if [ -d "${input}" ]; then
    local candidate
    for candidate in \
      "${input}/full_weights.pt" \
      "${input}/actor/model_state_dict/full_weights.pt" \
      "${input}/model_state_dict/full_weights.pt"; do
      if [ -f "${candidate}" ]; then
        abs_file "${candidate}"
        return 0
      fi
    done
  fi
  return 1
}

pretrained="${INIT_MODEL_STATE_DICT_PATH:-${WMAM_INIT_MODEL_STATE_DICT_PATH:-${PRETRAINED_MODEL_STATE_DICT_PATH:-${PRETRAINED_MODEL_PATH:-${PRETRAINED_DIR:-}}}}}"
if [ -z "${pretrained}" ] && [ "$#" -gt 0 ] && [[ "$1" != *=* ]]; then
  pretrained="$1"
  shift
fi

RESOLVED_PRETRAINED_MODEL_STATE_DICT_PATH=""
INIT_MODEL_STATE_DICT_OVERRIDE="actor.model.init_model_state_dict_path=null"
if [ -n "${pretrained}" ]; then
  if ! RESOLVED_PRETRAINED_MODEL_STATE_DICT_PATH="$(resolve_pretrained_path "${pretrained}")"; then
    echo "[RLinf] ERROR: could not resolve pretrained model state_dict from: ${pretrained}" >&2
    usage
    exit 2
  fi
  export PRETRAINED_MODEL_STATE_DICT_PATH="${RESOLVED_PRETRAINED_MODEL_STATE_DICT_PATH}"
  export WMAM_INIT_MODEL_STATE_DICT_PATH="${RESOLVED_PRETRAINED_MODEL_STATE_DICT_PATH}"
  INIT_MODEL_STATE_DICT_OVERRIDE="actor.model.init_model_state_dict_path=${RESOLVED_PRETRAINED_MODEL_STATE_DICT_PATH}"
else
  unset PRETRAINED_MODEL_STATE_DICT_PATH
  unset WMAM_INIT_MODEL_STATE_DICT_PATH
fi

# This is intentionally not runner.resume_dir. We only initialize model weights;
# optimizer, scheduler, dataloader state, and global_step start fresh.
unset RESUME_DIR

export CONFIG=${CONFIG:-real_world_franka_dual_wmam_motion_sft}
export NUM_GPUS=${NUM_GPUS:-8}
export NNODES=${NNODES:-1}
export PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-${NUM_GPUS}}
export PET_NNODES=${PET_NNODES:-${NNODES}}
export RUN_LOG_PATH=${RUN_LOG_PATH:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/results_franka_dual_wmam_continue}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-real_world_franka_dual_wmam_motion_continue_sft}
export ACTOR_MODEL_PRECISION=${ACTOR_MODEL_PRECISION:-fp32}
export REAL_WORLD_FRANKA_DUAL_MOTION_DIR_NAME=${REAL_WORLD_FRANKA_DUAL_MOTION_DIR_NAME:-motions_sam}
USER_GLOBAL_BATCH_SIZE_SET=${GLOBAL_BATCH_SIZE+x}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
TOTAL_GPUS=$((NNODES * NUM_GPUS))
DEFAULT_MICRO_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / TOTAL_GPUS))
if [ "${DEFAULT_MICRO_BATCH_SIZE}" -lt 1 ]; then
  DEFAULT_MICRO_BATCH_SIZE=1
fi
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-${DEFAULT_MICRO_BATCH_SIZE}}

if [ -n "${RESOLVED_PRETRAINED_MODEL_STATE_DICT_PATH}" ]; then
  echo "[RLinf] Continue WMAM from model weights: ${RESOLVED_PRETRAINED_MODEL_STATE_DICT_PATH}"
else
  echo "[RLinf] WMAM starts from configured base Wan2.2/DreamZero weights; init_model_state_dict_path=null."
fi
echo "[RLinf] Fresh optimizer/scheduler/dataloader state; runner.resume_dir is forced to null."

extra_args=(
  runner.resume_dir=null
  actor.micro_batch_size="${MICRO_BATCH_SIZE}"
  "${INIT_MODEL_STATE_DICT_OVERRIDE}"
  actor.model.init_model_state_dict_strict=false
  actor.model.init_model_state_dict_broadcast_from_rank0=true
)
if [ -n "${USER_GLOBAL_BATCH_SIZE_SET}" ]; then
  extra_args+=(actor.global_batch_size="${GLOBAL_BATCH_SIZE}")
fi

exec "${SCRIPT_DIR}/run_real_world_franka_dual_wmam_motion_sft_multi_node.sh" \
  "${extra_args[@]}" \
  "$@"
