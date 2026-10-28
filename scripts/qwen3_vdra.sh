#!/usr/bin/env bash
# Run the Qwen3-1.7B thinking-mode SPO + VDRA arm.
# Usage: SEED=42 STOP_STEP=220 GPUS=1 bash scripts/qwen3_vdra.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/qwen3_setup.sh"

SEED="${SEED:-42}"
TARGET="${STOP_STEP:-220}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export IGPO_R2_CONFIG="${IGPO_DIR}/configs/polIter_qwen3_1_7b_vdra_MATH.jsonnet"

exec bash "${HERE}/20_train.sh" R2 "${SEED}" "${TARGET}" "Qwen3_VDRA_s${SEED}"
