#!/usr/bin/env bash
# Experiment 1: proxy vs cheap heuristics (CPU only).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
run python3 "${VDRA_HOME}/diagnostics/exp1_proxy_quality.py" --banks "${BANKS_DIR}/*.jsonl.gz" \
  --out "${RESULTS_DIR}/exp1" --reps "${EXP1_REPS:-2000}"
