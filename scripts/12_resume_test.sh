#!/usr/bin/env bash
# Kill-and-resume check for each framework before the real runs.
# For every RUN in RUNS: train to KILL_AT updates, kill the whole process group,
# resume to TEST_STOP, and verify that
#   * the second launch started from a saved update >= KILL_AT,
#   * the framework reported loading optimizer state (log grep),
#   * the run finished at TEST_STOP with evaluation weights for TEST_STOP.
# Run it alone on a node: it ends with `ray stop --force` (CLEANUP_RAY=0 to skip).
#   GPUS=0,1 RUNS="R2 R4" bash scripts/12_resume_test.sh
#   GPUS=0,1,2 RUNS="R6" bash scripts/12_resume_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUNS="${RUNS:-R2 R4 R6}"
KILL_AT="${KILL_AT:-4}"
TEST_STOP="${TEST_STOP:-8}"
export RESUME_EVERY=2 KEEP_EVERY=2 MAX_RESTARTS=0
GPUS="${GPUS:?GPUS}"
mkdir -p "${RESULTS_DIR}/audit"
report="${RESULTS_DIR}/audit/resume_test.txt"; : > "${report}"
for r in ${RUNS}; do
  name="resumetest_${r}"
  dir="${RUNS_DIR}/${name}"
  [[ "${DRY_RUN}" == "1" ]] || rm -rf "${dir}"
  log "[${r}] phase 1: train until update ${KILL_AT}, then kill"
  if [[ "${DRY_RUN}" == "1" ]]; then
    run bash "${VDRA_HOME}/scripts/20_train.sh" "${r}" 42 "${TEST_STOP}" "${name}"
    continue
  fi
  GPUS="${GPUS}" setsid bash "${VDRA_HOME}/scripts/20_train.sh" "${r}" 42 "${TEST_STOP}" "${name}" \
    > "${dir}.phase1.log" 2>&1 &
  pid=$!
  until (( $(python3 "${VDRA_HOME}/analysis/run_status.py" "${dir}" 2>/dev/null || echo 0) >= KILL_AT )); do
    kill -0 "${pid}" 2>/dev/null || { log "[${r}] phase 1 exited early; see ${dir}.phase1.log"; break; }
    sleep 30
  done
  kill -TERM -- "-${pid}" 2>/dev/null || true; sleep 20; kill -KILL -- "-${pid}" 2>/dev/null || true
  if [[ "${CLEANUP_RAY:-1}" == "1" ]]; then ray stop --force >/dev/null 2>&1 || true; fi
  sleep 10
  killed_at="$(python3 "${VDRA_HOME}/analysis/run_status.py" "${dir}")"
  log "[${r}] killed with resumable update ${killed_at}; phase 2: resume to ${TEST_STOP}"
  GPUS="${GPUS}" bash "${VDRA_HOME}/scripts/20_train.sh" "${r}" 42 "${TEST_STOP}" "${name}" > "${dir}.phase2.log" 2>&1 || true
  if [[ "${CLEANUP_RAY:-1}" == "1" ]]; then ray stop --force >/dev/null 2>&1 || true; fi
  final="$(python3 "${VDRA_HOME}/analysis/run_status.py" "${dir}")"
  loaded="no"
  grep -qE "Resuming from|vdra-resume\] loaded|Load from checkpoint folder" "${dir}.phase2.log" "${dir}"/*.log 2>/dev/null && loaded="yes"
  evalw="$(python3 "${VDRA_HOME}/analysis/list_ckpts.py" "${dir}" | awk -v s="${TEST_STOP}" '$1==s' | wc -l)"
  verdict="PASS"
  (( killed_at >= KILL_AT )) || verdict="FAIL(no checkpoint before kill)"
  [[ "${loaded}" == "yes" ]] || verdict="FAIL(no resume message)"
  (( final >= TEST_STOP )) || verdict="FAIL(did not reach ${TEST_STOP})"
  (( evalw >= 1 )) || verdict="FAIL(no eval weights at ${TEST_STOP})"
  echo -e "${r}\tkilled_at=${killed_at}\tresumed_loaded=${loaded}\tfinal=${final}\teval_weights_at_stop=${evalw}\t${verdict}" | tee -a "${report}"
done
log "report: ${report}"
