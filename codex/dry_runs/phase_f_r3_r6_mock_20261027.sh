#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="$ROOT/codex/dry_runs/phase_f_current_20261027/bin:$PATH"
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
python3 -c 'import json; print("PYTHON3_SHIM_OK")'
export VDRA_HOME="$ROOT" WORK="$ROOT/work" IGPO_DIR="$ROOT/work/igpo" TREEPO_DIR="$ROOT/work/TreePO" TREERL_DIR="$ROOT/work/TreeRL"
export DRY_RUN=1 STOP_STEP=2 RESUME_EVERY=1 KEEP_EVERY=1 MAX_RESTARTS=0
export EVAL_EVERY_N=0 EVAL_AT_END=0 EVAL_STEP0=0 SEEDS=42 GPUS=0
export TREEPO_RAY_WAIT_REGISTER_CENTER_TIMEOUT=600 TREERL_VLLM_MEM=0.25 TREERL_VLLM_BATCHED_TOKENS=16384
export RUNS_DIR="$ROOT/codex/dry_runs/phase_f_r3_r6_mock_20261027/runs"
export RESULTS_DIR="$ROOT/codex/dry_runs/phase_f_r3_r6_mock_20261027/results"
export BANKS_DIR="$ROOT/codex/dry_runs/phase_f_r3_r6_mock_20261027/banks"
for run in R3 R4 R5 R6; do
  bash scripts/20_train.sh "$run" 42 2 "phase_f_mock_${run}"
  echo "PREFLIGHT_OK ${run}"
done