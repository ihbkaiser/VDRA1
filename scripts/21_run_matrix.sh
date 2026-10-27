#!/usr/bin/env bash
# Phase 2/3: the 18-run matrix.  Seed 42 first (PHASE=seed42), then the rest.
#   PHASE=seed42 RUNS="R1 R2 R3 R4" GPU_GROUPS="0,1 2,3 4,5 6,7" bash scripts/21_run_matrix.sh
#   PHASE=seed42 RUNS="R5 R6"       GPU_GROUPS="0,1,2 3,4,5"     bash scripts/21_run_matrix.sh
#   PHASE=rest   ... (seeds 123 2026)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUNS="${RUNS:-R1 R2 R3 R4}"
case "${PHASE:-all}" in
  seed42) seeds="42" ;;
  rest)   seeds="$(tr ' ' '\n' <<< "${SEEDS}" | grep -vx 42 | tr '\n' ' ')" ;;
  *)      seeds="${SEEDS}" ;;
esac
jobs="${RUNS_DIR}/_cfg/matrix_${PHASE:-all}_$(echo ${RUNS} | tr ' ' '-').txt"
mkdir -p "$(dirname "${jobs}")"; : > "${jobs}"
for s in ${seeds}; do
  for r in ${RUNS}; do
    echo "bash ${VDRA_HOME}/scripts/20_train.sh ${r} ${s}" >> "${jobs}"
  done
done
log "jobs: ${jobs}"
run_queue "${jobs}"
