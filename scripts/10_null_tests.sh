#!/usr/bin/env bash
# Gate G1, part 1: integration checks before any real run.
#  igpo:   VDRA with a constant proxy must reproduce SPO's branching exactly;
#  all:    VDRA hybrids must spend exactly their budget in every queue.
#   GPUS=0,1 bash scripts/10_null_tests.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
S="${VDRA_HOME}/scripts"
N="${NULL_STEPS:-2}"
export RESUME_EVERY=1 KEEP_EVERY=1 MAX_RESTARTS=0
VDRA_VARIANT_JSON='{"proxy":"constant"}' run bash "${S}/20_train.sh" R2 42 "${N}" null_R2_constant
run bash "${S}/20_train.sh" R1 42 "${N}" null_R1
run bash "${S}/20_train.sh" R2 42 "${N}" null_R2_tv
run bash "${S}/20_train.sh" R4 42 "${N}" null_R4_tv
run bash "${S}/20_train.sh" R6 42 "${N}" null_R6_tv
A="${VDRA_HOME}/analysis/budget_audit.py"
run python3 "${A}" --null "${RUNS_DIR}/null_R2_constant" "${RUNS_DIR}/null_R1" --out "${RESULTS_DIR}/audit/null_igpo.json"
for r in null_R2_tv null_R4_tv null_R6_tv; do
  run python3 "${A}" --logs "${RUNS_DIR}/${r}" --require-budget-evidence --out "${RESULTS_DIR}/audit/${r}.json"
done
log "check results/audit/*.json: null_igpo.pass must be true and budget_violations 0"
log "next: GPUS=... bash scripts/12_resume_test.sh  (kill-and-resume check for every framework)"
