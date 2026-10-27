#!/usr/bin/env bash
# Phase 1 pilot: PILOT_STEPS updates of all six configurations (seed 42), then
# per-iteration cost, budget audit and allocation headroom (gate G1).
#   RUNS="R1 R2 R3 R4" GPU_GROUPS="0,1 2,3 4,5 6,7" bash scripts/11_pilot.sh
#   RUNS="R5 R6" GPU_GROUPS="0,1,2 3,4,5" bash scripts/11_pilot.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUNS="${RUNS:-R1 R2 R3 R4 R5 R6}"
jobs="${RUNS_DIR}/_cfg/pilot_jobs.txt"; mkdir -p "$(dirname "${jobs}")"; : > "${jobs}"
for r in ${RUNS}; do
  echo "bash ${VDRA_HOME}/scripts/20_train.sh ${r} 42 ${PILOT_STEPS} pilot_${r}" >> "${jobs}"
done
run_queue "${jobs}"
for r in ${RUNS}; do
  case "${r}" in R1|R2) fw=treetune; g=${GPUS_PER_RUN};; R3|R4) fw=verl; g=${GPUS_PER_RUN};;
                 *) fw=openrlhf; g=$((TREERL_ACTOR_GPUS + TREERL_REF_GPUS + TREERL_VLLM_ENGINES));; esac
  case "${r}" in R2|R4|R6) budget_args=(--require-budget-evidence);; *) budget_args=();; esac
  run python3 "${VDRA_HOME}/analysis/gpu_hours.py" --run_dir "${RUNS_DIR}/pilot_${r}" --framework "${fw}" \
    --gpus "${g}" --out "${RESULTS_DIR}/pilot/gpu_hours_${r}.json" || true
  run python3 "${VDRA_HOME}/analysis/budget_audit.py" "${budget_args[@]}" --logs "${RUNS_DIR}/pilot_${r}" \
    --out "${RESULTS_DIR}/pilot/audit_${r}.json" || true
done
log "gate G1: cv_n_v >= 0.3 for R2/R4/R6, budget_violations == 0; update R3-R6 cost estimates"
