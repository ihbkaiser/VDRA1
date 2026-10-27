#!/usr/bin/env bash
# Phase 3b: continue each baseline until it has used the GPU-hours its hybrid
# used for STOP_STEP updates (same node type and GPU count).
#   PAIR=spo|treepo  SEED=42  GPUS=0,1 bash scripts/30_compute_matched.sh
# All three frameworks resume from their STOP_STEP state (weights, optimizer, LR
# schedule, data position); the continuation saves evaluation weights every
# SAVE_EVERY_CONT updates.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
PAIR="${PAIR:?PAIR}"; SEED="${SEED:?SEED}"
case "${PAIR}" in
  spo)    B=R1; H=R2; FW=treetune; G=${GPUS_PER_RUN} ;;
  treepo) B=R3; H=R4; FW=verl;     G=${GPUS_PER_RUN} ;;
  treerl) B=R5; H=R6; FW=openrlhf; G=$((TREERL_ACTOR_GPUS + TREERL_REF_GPUS + TREERL_VLLM_ENGINES)) ;;
esac
# both runs must have used the same number of GPUs; take it from their launch records
G="$(run_gpus "${RUNS_DIR}/${H}_s${SEED}" "${G}")"
GB="$(run_gpus "${RUNS_DIR}/${B}_s${SEED}" "${G}")"
if [[ "${GB}" != "${G}" ]]; then echo "${B} used ${GB} GPUs but ${H} used ${G}; GPU-hours are not comparable" >&2; exit 2; fi
GH="${RESULTS_DIR}/gpu_hours"; mkdir -p "${GH}"
for r in "${B}" "${H}"; do
  run python3 "${VDRA_HOME}/analysis/gpu_hours.py" --run_dir "${RUNS_DIR}/${r}_s${SEED}" --framework "${FW}" \
    --gpus "${G}" --out "${GH}/${r}_s${SEED}.json"
done
[[ "${DRY_RUN}" == "1" ]] && { log "dry run: skipping target computation"; exit 0; }
TARGET="$(python3 "${VDRA_HOME}/analysis/compute_matched_target.py" \
  --baseline "${GH}/${B}_s${SEED}.json" --hybrid "${GH}/${H}_s${SEED}.json" \
  --stop "${STOP_STEP}" --checkpoint-spacing "${SAVE_EVERY_CONT}")"
log "${PAIR} seed ${SEED}: hybrid used $(python3 -c "import json;print(round(json.load(open('${GH}/${H}_s${SEED}.json'))['${STOP_STEP}'],2))") GPU-h; baseline target step ${TARGET}"
# keep the STOP_STEP state: treetune/verl prune checkpoints as training continues
for d in "${RUNS_DIR}/${B}_s${SEED}"/checkpoints/ckpt--iter_${STOP_STEP}--*/hf_pretrained "${RUNS_DIR}/${B}_s${SEED}/hf/global_step_${STOP_STEP}"; do
  [[ -d "${d}" ]] && run cp -r "${d}" "${RUNS_DIR}/${B}_s${SEED}/archive_step${STOP_STEP}_hf"
done
# continuation checkpoints are evaluated once at the end (no mid-continuation pauses)
KEEP_EVERY="${SAVE_EVERY_CONT}" RESUME_EVERY="${SAVE_EVERY_CONT}" GPUS="${GPUS:?GPUS}" \
  EVAL_EVERY_N=0 EVAL_AT_END="${EVAL_AT_END_CONT:-1}" EVAL_STEP0=0 \
  run bash "${VDRA_HOME}/scripts/20_train.sh" "${B}" "${SEED}" "${TARGET}"
run python3 "${VDRA_HOME}/analysis/gpu_hours.py" --run_dir "${RUNS_DIR}/${B}_s${SEED}" --framework "${FW}" \
  --gpus "${G}" --out "${GH}/${B}_s${SEED}.json"
