#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export PET_NNODES=${PET_NNODES:-${NNODES}}
export PET_NODE_RANK=${PET_NODE_RANK:-${NODE_RANK}}
export CONFIG=${CONFIG:-robotwin2_motion_sft_dreamzero_5b}
exec bash "${SCRIPT_DIR}/run_robotwin2_dreamzero_motion_sft_multi_node.sh" "$@"
