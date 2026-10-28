#!/usr/bin/env bash
# Prepare the shared Qwen3 SPO/VDRA environment from the user's B200 bundle.
# All paths remain overrideable through the environment.
set -euo pipefail

QWEN3_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VDRA_HOME="${QWEN3_HOME:-$(cd "${QWEN3_SCRIPTS_DIR}/.." && pwd)}"
export VDRA_BUNDLE_ROOT="${VDRA_BUNDLE_ROOT:-/workspace/storage-shared/nlp/thanhpv43/vdra-new}"
export WORK="${QWEN3_WORK:-${VDRA_BUNDLE_ROOT}}"
export IGPO_DIR="${QWEN3_IGPO_DIR:-${VDRA_BUNDLE_ROOT}/work/igpo}"
export VDRA_QWEN3_MODE=1

export DATA_DIR="${QWEN3_DATA_DIR:-${VDRA_BUNDLE_ROOT}/data_vdra}"
export MATH_DICT="${QWEN3_MATH_DICT:-${VDRA_BUNDLE_ROOT}/source_data/math}"
QWEN3_MODEL_PATH_EFFECTIVE="${QWEN3_MODEL_PATH:-/workspace/storage-shared/models/Qwen3-1.7B}"
export MODEL_PATH="${QWEN3_MODEL_PATH_EFFECTIVE}"
export EVAL_SETS="${QWEN3_EVAL_SETS:-math500=${MATH_DICT}:test aime24=${VDRA_BUNDLE_ROOT}/source_data/aime24 aime25=${VDRA_BUNDLE_ROOT}/source_data/aime25 amc23=${VDRA_BUNDLE_ROOT}/source_data/amc23 olympiadbench=${VDRA_BUNDLE_ROOT}/source_data/olympiadbench_hf}"

export RUNS_DIR="${QWEN3_RUNS_DIR:-${VDRA_HOME}/runs_qwen3}"
export RESULTS_DIR="${QWEN3_RESULTS_DIR:-${VDRA_HOME}/results_qwen3}"
export BANKS_DIR="${QWEN3_BANKS_DIR:-${VDRA_HOME}/banks_qwen3}"
export VDRA_B200=0
export HF_HOME="${VDRA_BUNDLE_ROOT}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export TORCH_CUDA_ARCH_LIST=10.0
export TORCH_EXTENSIONS_DIR="${VDRA_BUNDLE_ROOT}/torch_extensions"

# Reuse the IGPO source and venv already present in the offline bundle.
QWEN3_VENV="${QWEN3_VENV:-${VDRA_BUNDLE_ROOT}/project/.venv-igpo-2}"
if [[ ! -f "${QWEN3_VENV}/bin/activate" ]]; then
  echo "[qwen3] missing IGPO venv: ${QWEN3_VENV}/bin/activate" >&2
  exit 2
fi
if [[ "${VIRTUAL_ENV:-}" != "${QWEN3_VENV}" ]]; then
  # shellcheck disable=SC1090
  source "${QWEN3_VENV}/bin/activate"
fi

# Load standard settings. Qwen3 mode skips env.local.sh so stale paths from an
# older checkout cannot redirect these two runs; QWEN3_* variables are overrides.
source "${VDRA_HOME}/scripts/env.sh"
export CONDA_IGPO=""
export MODEL_PATH="${QWEN3_MODEL_PATH_EFFECTIVE}"

for required in \
  "${DATA_DIR}/train.parquet" \
  "${DATA_DIR}/math-nodiag" \
  "${MATH_DICT}" \
  "${MODEL_PATH}/config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[qwen3] required file or directory not found: ${required}" >&2
    exit 2
  fi
done

# The bundle contains an unpacked IGPO checkout, so prepare it entirely offline.
if [[ ! -d "${IGPO_DIR}/treetune" || ! -d "${IGPO_DIR}/vdra_core" ]]; then
  echo "[qwen3] expected bundled IGPO source at ${IGPO_DIR}" >&2
  exit 2
fi

# Serialize preparation because SPO and VDRA may start at nearly the same time.
exec 9>"${TMPDIR:-/tmp}/vdra-qwen3-igpo-install.lock"
flock 9
python "${VDRA_HOME}/scripts/qwen3_prepare_offline_igpo.py" "${IGPO_DIR}" "${VDRA_HOME}"
flock -u 9
exec 9>&-

mkdir -p "${RUNS_DIR}" "${RESULTS_DIR}" "${BANKS_DIR}"
echo "[qwen3] model: ${MODEL_PATH}"
echo "[qwen3] IGPO source: ${IGPO_DIR}"
echo "[qwen3] run outputs: ${RUNS_DIR}"
