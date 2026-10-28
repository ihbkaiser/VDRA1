#!/usr/bin/env bash
# Run the Qwen3-1.7B thinking-mode SPO baseline.
# Usage: SEED=42 STOP_STEP=220 GPUS=0 bash scripts/qwen3_spo.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/qwen3_setup.sh"

SEED="${SEED:-42}"
TARGET="${STOP_STEP:-220}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export IGPO_R1_CONFIG="${IGPO_DIR}/configs/polIter_qwen3_1_7b_spo_logged_MATH.jsonnet"

exec bash "${HERE}/20_train.sh" R1 "${SEED}" "${TARGET}" "Qwen3_SPO_s${SEED}"
