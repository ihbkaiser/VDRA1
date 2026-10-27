#!/usr/bin/env bash
# Experiment 2: measured edge-averaged advantage error (CPU only, ~8 min per bank at 500 draws).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run python3 "${VDRA_HOME}/diagnostics/exp2_advantage_error.py" --banks "${BANKS_DIR}/*.jsonl.gz" \
  --out "${RESULTS_DIR}/exp2" --draws "${EXP2_DRAWS:-500}"
