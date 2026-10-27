#!/usr/bin/env bash
# Carve the 256-problem diagnostic split, export training data for all three
# frameworks, and check train/eval leakage (exits non-zero on exact duplicates).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
use_env "${CONDA_IGPO}"
export PYTHONPATH="${IGPO_DIR}:${VDRA_HOME}:${PYTHONPATH:-}"
run python3 "${VDRA_HOME}/analysis/prepare_data.py" --math_dict "${MATH_DICT}" \
  --train_split "${MATH_TRAIN_SPLIT}" --out "${DATA_DIR}" --n_diag 256 --seed 2026
read -r -a sets <<< "${EVAL_SETS}"
run python3 "${VDRA_HOME}/analysis/check_leakage.py" --train "${DATA_DIR}/train.parquet" \
  --eval "${sets[@]}" --diag "${DATA_DIR}/diag256.jsonl" --out "${RESULTS_DIR}/leakage.json" \
  --allow-exact-overlaps
