#!/usr/bin/env bash
# Evaluate every checkpoint that has no results yet, for every run.  Already
# evaluated (run, step, benchmark, mode) files are skipped, so this is cheap
# after in-training evaluation (EVAL_EVERY_N) and fills in whatever is missing
# (continuation checkpoints, failed evaluations, runs trained with EVAL_EVERY_N=0).
#   RUN_IDS="R1_s42 R2_s42" bash scripts/50_eval.sh
# One job per run; each job splits its benchmarks over the GPUs of its group.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
GPU_GROUPS="${EVAL_GPU_GROUPS:-${GPU_GROUPS}}"
RUN_IDS="${RUN_IDS:-$(cd "${RUNS_DIR}" && ls -d R?_s* 2>/dev/null | tr '\n' ' ')}"
mkdir -p "${RESULTS_DIR}/eval" "${RESULTS_DIR}/gpu_hours" "${RUNS_DIR}/_cfg"
limit_arg=""
if [[ -n "${VDRA_EVAL_LIMIT:-}" ]]; then
  [[ "${VDRA_EVAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || { echo "VDRA_EVAL_LIMIT must be a positive integer" >&2; exit 2; }
  limit_arg="--limit ${VDRA_EVAL_LIMIT}"
fi
steps_arg=""
if [[ -n "${VDRA_EVAL_STEPS:-}" ]]; then
  [[ "${VDRA_EVAL_STEPS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "VDRA_EVAL_STEPS must be a comma list of step numbers" >&2; exit 2; }
  steps_arg="--steps ${VDRA_EVAL_STEPS}"
fi
jobs="${RUNS_DIR}/_cfg/eval_jobs.txt"; : > "${jobs}"
for rid in ${RUN_IDS}; do
  case "${rid}" in R1*|R2*) fw=treetune; g=${GPUS_PER_RUN};; R3*|R4*) fw=verl; g=${GPUS_PER_RUN};;
                   *) fw=openrlhf; g=$((TREERL_ACTOR_GPUS + TREERL_REF_GPUS + TREERL_VLLM_ENGINES));; esac
  g="$(run_gpus "${RUNS_DIR}/${rid}" "${g}")"
  gh="${RESULTS_DIR}/gpu_hours/${rid}.json"
  run python3 "${VDRA_HOME}/analysis/gpu_hours.py" --run_dir "${RUNS_DIR}/${rid}" --framework "${fw}" \
    --gpus "${g}" --out "${gh}" || true
  step0=""; [[ "${EVAL_STEP0}" == "1" ]] && step0="--include_step0 --base_model '${MODEL_PATH}'"
  tokenizer_arg="--tokenizer '${MODEL_PATH}'"
  echo "PYTHONPATH=${IGPO_DIR}:${VDRA_HOME} python3 ${VDRA_HOME}/analysis/eval_pending.py --run_dir ${RUNS_DIR}/${rid} --run_id ${rid} --gpus \$GPUS --dataset ${EVAL_SETS} --out_dir ${RESULTS_DIR}/eval --gpu_hours_json ${gh} --sample_k ${SAMPLE_K} ${step0} ${tokenizer_arg} ${limit_arg} ${steps_arg}" >> "${jobs}"
done
log "eval jobs: $(wc -l < "${jobs}")"
run_queue "${jobs}"
