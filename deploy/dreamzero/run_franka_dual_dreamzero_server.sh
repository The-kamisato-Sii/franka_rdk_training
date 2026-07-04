#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLINF_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
DREAMZERO_ROOT=${DREAMZERO_ROOT:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/dreamzero}

export PYTHONPATH="${RLINF_ROOT}:${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export EMBODIED_PATH=${EMBODIED_PATH:-"${RLINF_ROOT}"}
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_FLAX=0
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=1
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-TE}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

MODEL_PATH=${MODEL_PATH:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/results_franka_dual/real_world_franka_dual_dreamzero_sft_arrange_vegetables_filtered/hf_global_step_20000}
TOKENIZER_PATH=${TOKENIZER_PATH:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/dreamzero/checkpoints/umt5-xxl}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29511}
OUTPUT_ROOT=${OUTPUT_ROOT:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/RLinf/deploy/dreamzero/log}
N_ACTION_STEPS=${N_ACTION_STEPS:-48}
RETURN_ACTION_DIM=${RETURN_ACTION_DIM:-16}
MODEL_STATE_DIM=${MODEL_STATE_DIM:-64}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-4}
NUM_DIT_STEPS=${NUM_DIT_STEPS:-$NUM_INFERENCE_STEPS}
DEFAULT_PROMPT=${DEFAULT_PROMPT:-"Arrange the vegetables"}
ACTION_INFERENCE_MODE=${ACTION_INFERENCE_MODE:-lazy}

if [[ -z "$MODEL_PATH" ]]; then
  echo "[dreamzero-server] MODEL_PATH is empty. Set it to the converted DreamZero checkpoint path." >&2
  exit 2
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[dreamzero-server] MODEL_PATH must be a converted DreamZero model directory, not a .pt weight file: $MODEL_PATH" >&2
  echo "[dreamzero-server] Convert full_weights.pt with rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf first." >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/experiment_cfg/conf.yaml" ]]; then
  echo "[dreamzero-server] MODEL_PATH is missing experiment_cfg/conf.yaml: $MODEL_PATH" >&2
  echo "[dreamzero-server] This means the checkpoint has not been converted to Groot/DreamZero HF format yet." >&2
  exit 2
fi

MODEL_OVERRIDE_ARGS=()
if [[ -n "${MODEL_CONFIG_OVERRIDES:-}" ]]; then
  read -r -a _overrides <<< "$MODEL_CONFIG_OVERRIDES"
  for override in "${_overrides[@]}"; do
    MODEL_OVERRIDE_ARGS+=(--model-config-override "$override")
  done
fi

cd "$RLINF_ROOT"
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  deploy/dreamzero/serve_franka_dual_dreamzero.py \
  --model-kind dreamzero \
  --payload-key dreamzero \
  --model-path "$MODEL_PATH" \
  --dreamzero-root "$DREAMZERO_ROOT" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --n-action-steps "$N_ACTION_STEPS" \
  --return-action-dim "$RETURN_ACTION_DIM" \
  --model-state-dim "$MODEL_STATE_DIM" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --num-dit-steps "$NUM_DIT_STEPS" \
  --default-prompt "$DEFAULT_PROMPT" \
  --action-inference-mode "$ACTION_INFERENCE_MODE" \
  --output-root "$OUTPUT_ROOT" \
  "${MODEL_OVERRIDE_ARGS[@]}"
