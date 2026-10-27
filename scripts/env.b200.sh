#!/usr/bin/env bash
# Shared environment for all six B200 runs. Method-specific algorithm settings
# remain in the existing launchers and VDRA integration configs.

export WORK="${WORK:-/work}"
export RUNS_DIR="${RUNS_DIR:-${WORK}/runs}"
export RESULTS_DIR="${RESULTS_DIR:-${WORK}/results}"
export BANKS_DIR="${BANKS_DIR:-${WORK}/banks}"
export DATA_DIR="${DATA_DIR:-${WORK}/data_vdra}"
export MATH_DICT="${MATH_DICT:-${WORK}/source_data/math}"
export MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
export EVAL_SETS="${EVAL_SETS:-math500=${MATH_DICT}:test aime24=${WORK}/source_data/aime24 aime25=${WORK}/source_data/aime25 amc23=${WORK}/source_data/amc23 olympiadbench=${WORK}/source_data/olympiadbench_hf}"

# Framework sources are baked into the common per-framework Modal images.
export IGPO_DIR="${IGPO_DIR:-/opt/work/igpo}"
export TREEPO_DIR="${TREEPO_DIR:-/opt/work/TreePO}"
export TREERL_DIR="${TREERL_DIR:-/opt/work/TreeRL}"
export CONDA_IGPO="" CONDA_TREEPO="" CONDA_TREERL=""

# One B200 is allocated per run. Each framework uses physical CUDA device 0.
export GPUS_PER_RUN="1"
export GPU_GROUPS="0"
export GPUS="${GPUS:-0}"
export TREERL_ACTOR_GPUS="1" TREERL_REF_GPUS="1" TREERL_VLLM_ENGINES="1"

# Shared model caches live on the persistent volume; scratch remains local NVMe.
export HF_HOME="${HF_HOME:-${WORK}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_DISABLE_TELEMETRY="1"
export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM="false"
export VLLM_USE_V1="0"
export VDRA_B200="1"
export PYTHONUNBUFFERED="1"
export TMPDIR="${TMPDIR:-/tmp}"
