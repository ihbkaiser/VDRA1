#!/usr/bin/env bash
# Central configuration for every script.  Override any variable in the
# environment or in scripts/env.local.sh (sourced last, git-ignored).

export VDRA_HOME="${VDRA_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ "${VDRA_B200:-0}" == "1" ]]; then
  source "${VDRA_HOME}/scripts/env.b200.sh"
fi
export WORK="${WORK:-${VDRA_HOME}/work}"

# --- code checkouts (created by 00_setup.sh) ---------------------------------
export IGPO_DIR="${IGPO_DIR:-${WORK}/igpo}"              # your igpo repo (VDRA = InGPO budget mode)
export IGPO_REMOTE="${IGPO_REMOTE:-https://github.com/Thanh124pav/igpo.git}"
export TREEPO_DIR="${TREEPO_DIR:-${WORK}/TreePO}"
export TREERL_DIR="${TREERL_DIR:-${WORK}/TreeRL}"

# --- python environments (conda env names; empty = use the current env) -----
export CONDA_IGPO="${CONDA_IGPO:-}"
export CONDA_TREEPO="${CONDA_TREEPO:-}"
export CONDA_TREERL="${CONDA_TREERL:-}"

# --- model and data ------------------------------------------------------------
export MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
export MATH_DICT="${MATH_DICT:-${IGPO_DIR}/data/math}"   # HF DatasetDict used by treetune
export MATH_TRAIN_SPLIT="${MATH_TRAIN_SPLIT:-train}"
export DATA_DIR="${DATA_DIR:-${WORK}/data_vdra}"
# evaluation sets: name=path[:split]  (download with igpo/scripts/download_eval_datasets.py)
export EVAL_SETS="${EVAL_SETS:-math500=${IGPO_DIR}/data/math:test aime24=${IGPO_DIR}/data/aime24 aime25=${IGPO_DIR}/data/aime25 amc23=${IGPO_DIR}/data/amc23 olympiadbench=${IGPO_DIR}/data/olympiadbench_hf}"

# --- outputs ---------------------------------------------------------------------
export RUNS_DIR="${RUNS_DIR:-${WORK}/runs}"
export RESULTS_DIR="${RESULTS_DIR:-${WORK}/results}"
export BANKS_DIR="${BANKS_DIR:-${WORK}/banks}"

# --- experiment matrix ---------------------------------------------------------
export SEEDS="${SEEDS:-42 123 2026}"
export STOP_STEP="${STOP_STEP:-220}"                     # policy updates per run
export SAVE_EVERY="${SAVE_EVERY:-25}"                    # evaluation weights kept every N updates
export SAVE_EVERY_CONT="${SAVE_EVERY_CONT:-5}"           # ... during compute-matched continuation
export RESUME_EVERY="${RESUME_EVERY:-5}"                 # resumable checkpoint every N updates (all frameworks)
export MAX_RESTARTS="${MAX_RESTARTS:-3}"                 # automatic restarts after a crash, per launch
export RESTART_DELAY="${RESTART_DELAY:-60}"              # seconds between restarts
# in-training evaluation: every EVAL_EVERY_N updates training pauses, the new checkpoints
# are evaluated on the run's own GPUs, then training resumes (0 = evaluate separately with 50_eval.sh)
export EVAL_EVERY_N="${EVAL_EVERY_N:-25}"                # must be a multiple of RESUME_EVERY for R1/R2
export EVAL_STEP0="${EVAL_STEP0:-1}"                     # evaluate the base model before the first update
export EVAL_AT_END="${EVAL_AT_END:-1}"                   # evaluate pending checkpoints when STOP is reached
export SAMPLE_K="${SAMPLE_K:-16}"                        # avg@k samples on aime24/aime25/amc23
export BANK_STEPS="${BANK_STEPS:-0 50 100 150 200 220}"
export PILOT_STEPS="${PILOT_STEPS:-20}"

# --- hardware --------------------------------------------------------------------
export GPUS_PER_RUN="${GPUS_PER_RUN:-2}"                  # treetune and verl runs
export GPU_GROUPS="${GPU_GROUPS:-0,1 2,3 4,5 6,7}"       # one concurrent run per group
export TREERL_ACTOR_GPUS="${TREERL_ACTOR_GPUS:-1}"
export TREERL_REF_GPUS="${TREERL_REF_GPUS:-1}"
export TREERL_VLLM_ENGINES="${TREERL_VLLM_ENGINES:-1}"

# --- method settings ---------------------------------------------------------------
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TREEPO_WIDTH="${TREEPO_WIDTH:-16}"
export TREEPO_FIXED_DIV="${TREEPO_FIXED_DIV:-2}"
export TREEPO_MAX_DEPTH="${TREEPO_MAX_DEPTH:-3}"
export TREEPO_SEG_TOKENS="${TREEPO_SEG_TOKENS:-512}"
export TREERL_M="${TREERL_M:-6}" TREERL_N="${TREERL_N:-2}" TREERL_L="${TREERL_L:-1}" TREERL_T="${TREERL_T:-2}"
# VDRA hyper-parameters as JSON (merged over vdra_core defaults)
if [[ -z "${VDRA_CONFIG_JSON:-}" ]]; then VDRA_CONFIG_JSON='{}'; fi
export VDRA_CONFIG_JSON

export DRY_RUN="${DRY_RUN:-0}"
[[ -f "${VDRA_HOME}/scripts/env.local.sh" ]] && source "${VDRA_HOME}/scripts/env.local.sh"
true
