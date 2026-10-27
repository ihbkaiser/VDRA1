#!/usr/bin/env bash
# Node banks for Experiments 1 and 2 from frozen R1 / R2 checkpoints (seed 42).
# Step 0 is the shared initial model and is built once.
#   GPU_GROUPS="0 1 2 3 4 5 6 7" bash scripts/40_build_banks.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
GPU_GROUPS="${BANK_GPU_GROUPS:-$(tr ',' ' ' <<< "${GPU_GROUPS}")}"  # one GPU per job
mkdir -p "${BANKS_DIR}" "${RUNS_DIR}/_cfg"
jobs="${RUNS_DIR}/_cfg/bank_jobs.txt"; : > "${jobs}"
common="--prompts ${DATA_DIR}/diag256.jsonl --n_prompts ${BANK_PROMPTS:-64} --seed 42 --temperature ${TEMPERATURE} --vdra_config '${VDRA_CONFIG_JSON}'"
py="PYTHONPATH=${IGPO_DIR}:${VDRA_HOME} CUDA_VISIBLE_DEVICES=\$GPUS python3 ${VDRA_HOME}/diagnostics/build_node_bank.py"
echo "${py} --model ${MODEL_PATH} --label base_step0 --out ${BANKS_DIR}/R0_base_step0.jsonl.gz ${common}" >> "${jobs}"
for r in R1 R2; do
  while IFS=$'\t' read -r step path; do
    for want in ${BANK_STEPS}; do
      if [[ "${step}" == "${want}" && "${step}" != "0" ]]; then
        echo "${py} --model ${path} --label ${r}_s42_step${step} --out ${BANKS_DIR}/${r}_s42_step${step}.jsonl.gz ${common}" >> "${jobs}"
      fi
    done
  done < <(python3 "${VDRA_HOME}/analysis/list_ckpts.py" "${RUNS_DIR}/${r}_s42")
done
log "bank jobs: $(wc -l < "${jobs}")"
run_queue "${jobs}"
