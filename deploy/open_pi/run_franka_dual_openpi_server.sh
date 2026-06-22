#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="/inspire/hdd/project/robot-body/linbokai-CZXS24250037"
RLINF_OPENPI_ENV="${RLINF_OPENPI_ENV:-${PROJECT_ROOT}/miniconda/envs/rlinf_openpi}"
PYTHON_BIN="${PYTHON_BIN:-${RLINF_OPENPI_ENV}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[open_pi] ERROR: cannot execute PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "[open_pi] Activate the rlinf_openpi environment or set PYTHON_BIN manually." >&2
  echo "[open_pi] Example: source ${PROJECT_ROOT}/miniconda/etc/profile.d/conda.sh && conda activate rlinf_openpi" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"${REPO_ROOT}/checkpoints/openpi_cache"}
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-"${REPO_ROOT}/checkpoints/pi05_base_pytorch_real_world_joint"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"${PROJECT_ROOT}/results/real_world_franka_dual_openpi_pi05_sft/checkpoints/global_step_20000"}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-cuda:0}
PRECISION=${PRECISION:-bfloat16}
IMAGE_PAYLOAD_FORMAT=${IMAGE_PAYLOAD_FORMAT:-raw_hwc}
N_ACTION_STEPS=${N_ACTION_STEPS:-48}
RETURN_ACTION_DIM=${RETURN_ACTION_DIM:-16}
DEFAULT_PROMPT=${DEFAULT_PROMPT:-""}
ALLOW_LEGACY_224_CHW=${ALLOW_LEGACY_224_CHW:-0}

EXTRA_ARGS=()
if [[ "${ALLOW_LEGACY_224_CHW}" == "1" || "${ALLOW_LEGACY_224_CHW}" == "true" ]]; then
  EXTRA_ARGS+=(--allow-legacy-224-chw)
fi

echo "[open_pi] REPO_ROOT=${REPO_ROOT}"
echo "[open_pi] PYTHON_BIN=${PYTHON_BIN}"
echo "[open_pi] OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "[open_pi] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[open_pi] CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "[open_pi] HOST=${HOST}"
echo "[open_pi] PORT=${PORT}"
echo "[open_pi] DEVICE=${DEVICE}"
echo "[open_pi] PRECISION=${PRECISION}"
echo "[open_pi] IMAGE_PAYLOAD_FORMAT=${IMAGE_PAYLOAD_FORMAT}"
echo "[open_pi] N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "[open_pi] RETURN_ACTION_DIM=${RETURN_ACTION_DIM}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/serve_franka_dual_openpi.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --precision "${PRECISION}" \
  --base-model-path "${BASE_MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --image-payload-format "${IMAGE_PAYLOAD_FORMAT}" \
  --n-action-steps "${N_ACTION_STEPS}" \
  --return-action-dim "${RETURN_ACTION_DIM}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
