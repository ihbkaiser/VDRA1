#!/usr/bin/env bash
# Shared body of the six one-GPU experiment scripts (R1_spo.sh ... R6_treerl_vdra.sh).
# The caller sets EXP (R1..R6) and GPU_DEFAULT before sourcing this file.
#
# What one experiment script does, all on its single GPU:
#   1. for every seed in SEEDS: train EXP to STOP_STEP updates with automatic restarts;
#      every EVAL_EVERY_N updates training pauses, the new checkpoints are evaluated on
#      the same GPU, then training resumes;
#   2. baselines only (R1, R3, R5), COMPUTE_MATCHED=1: wait until the matching VDRA run
#      (R2, R4, R6, same seed) has finished, then continue the baseline until it has
#      used the same GPU-hours, and evaluate the continuation checkpoints.
# Re-running the script resumes where it stopped; finished parts are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/env.sh"
source "${HERE}/lib.sh"

GPU="${GPU:-${GPU_DEFAULT}}"
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "GPU must be one GPU id, got '${GPU}'" >&2; exit 2; }
export GPUS="${GPU}" GPUS_PER_RUN=1 TREERL_ACTOR_GPUS=1 TREERL_REF_GPUS=1 TREERL_VLLM_ENGINES=1
COMPUTE_MATCHED="${COMPUTE_MATCHED:-1}"
PARTNER_POLL_S="${PARTNER_POLL_S:-600}"

case "${EXP}" in
  R1) FW=treetune; PARTNER=R2; PAIR=spo;    CODE="${IGPO_DIR}";   DATA="${DATA_DIR}/math-nodiag" ;;
  R2) FW=treetune; PARTNER="";  PAIR="";    CODE="${IGPO_DIR}";   DATA="${DATA_DIR}/math-nodiag" ;;
  R3) FW=verl;     PARTNER=R4; PAIR=treepo; CODE="${TREEPO_DIR}"; DATA="${DATA_DIR}/train.parquet" ;;
  R4) FW=verl;     PARTNER="";  PAIR="";    CODE="${TREEPO_DIR}"; DATA="${DATA_DIR}/train.parquet" ;;
  R5) FW=openrlhf; PARTNER=R6; PAIR=treerl; CODE="${TREERL_DIR}"; DATA="${DATA_DIR}/train_treerl.jsonl" ;;
  R6) FW=openrlhf; PARTNER="";  PAIR="";    CODE="${TREERL_DIR}"; DATA="${DATA_DIR}/train_treerl.jsonl" ;;
  *) echo "unknown EXP ${EXP}" >&2; exit 2 ;;
esac

LOGDIR="${RUNS_DIR}/_logs"; mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/${EXP}_gpu${GPU}.log"
exec > >(tee -a "${LOG}") 2>&1
log "experiment ${EXP} (${FW}) on GPU ${GPU}; seeds: ${SEEDS}; stop ${STOP_STEP}; eval every ${EVAL_EVERY_N}"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -d "${CODE}" ]] || { log "missing ${CODE}: run scripts/00_setup.sh first"; exit 2; }
  [[ -e "${DATA}" ]] || { log "missing ${DATA}: run scripts/01_prepare_data.sh first"; exit 2; }
  if command -v nvidia-smi >/dev/null; then
    used="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" || used=0
    (( used == 0 )) || log "warning: GPU ${GPU} already has ${used} process(es); memory may not fit"
  fi
fi

done_at() { python3 "${VDRA_HOME}/analysis/run_status.py" "$1" --framework "${FW}" 2>/dev/null || echo 0; }

failed=()
for seed in ${SEEDS}; do
  log "=== ${EXP} seed ${seed}: training ==="
  if ! bash "${HERE}/20_train.sh" "${EXP}" "${seed}"; then
    log "${EXP} seed ${seed} did not finish; continuing with the next seed (re-run this script to resume it)"
    failed+=("train:${seed}")
  fi
done

if [[ -n "${PARTNER}" && "${COMPUTE_MATCHED}" == "1" ]]; then
  for seed in ${SEEDS}; do
    [[ " ${failed[*]:-} " == *" train:${seed} "* ]] && continue
    pdir="${RUNS_DIR}/${PARTNER}_s${seed}"
    log "=== ${EXP} seed ${seed}: compute-matched continuation (waits for ${PARTNER}_s${seed}) ==="
    if [[ "${DRY_RUN}" != "1" ]]; then
      while (( $(done_at "${pdir}") < STOP_STEP )); do
        log "waiting for ${PARTNER}_s${seed} (at update $(done_at "${pdir}") of ${STOP_STEP}); next check in ${PARTNER_POLL_S}s"
        sleep "${PARTNER_POLL_S}"
      done
    fi
    if ! PAIR="${PAIR}" SEED="${seed}" bash "${HERE}/30_compute_matched.sh"; then
      log "continuation of ${EXP} seed ${seed} failed; re-run this script to resume it"
      failed+=("cont:${seed}")
    fi
  done
fi

if (( ${#failed[@]} )); then log "${EXP} finished with unfinished parts: ${failed[*]}"; exit 1; fi
log "${EXP} complete on GPU ${GPU}. Results: ${RESULTS_DIR}/eval"
