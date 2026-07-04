#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/inspire/hdd/project/robot-body/linbokai-CZXS24250037/miniconda/envs/rlinf_openpi/bin/python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_franka_dual_sessions.py" \
  --src-root /inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual \
  --dst-root /inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_v2 \
  --overwrite \
  --video-mode hardlink \
  "$@"
