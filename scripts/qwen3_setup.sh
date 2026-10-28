#!/usr/bin/env bash
# Prepare the shared Qwen3 SPO/VDRA environment from the user's B200 bundle.
# All paths remain overrideable through the environment.
set -euo pipefail

QWEN3_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VDRA_HOME="${VDRA_HOME:-$(cd "${QWEN3_SCRIPTS_DIR}/.." && pwd)}"
export VDRA_BUNDLE_ROOT="${VDRA_BUNDLE_ROOT:-/workspace/storage-shared/nlp/thanhpv43/vdra-new}"
export WORK="${WORK:-${VDRA_HOME}/work}"
export IGPO_DIR="${IGPO_DIR:-${WORK}/igpo}"
export IGPO_REMOTE="${IGPO_REMOTE:-https://github.com/Thanh124pav/igpo.git}"

export DATA_DIR="${DATA_DIR:-${VDRA_BUNDLE_ROOT}/data_vdra}"
export MATH_DICT="${MATH_DICT:-${VDRA_BUNDLE_ROOT}/source_data/math}"
export MODEL_PATH="${MODEL_PATH:-/workspace/storage-shared/models/Qwen3-1.7B}"
QWEN3_MODEL_PATH="${MODEL_PATH}"
export EVAL_SETS="${EVAL_SETS:-math500=${MATH_DICT}:test aime24=${VDRA_BUNDLE_ROOT}/source_data/aime24 aime25=${VDRA_BUNDLE_ROOT}/source_data/aime25 amc23=${VDRA_BUNDLE_ROOT}/source_data/amc23 olympiadbench=${VDRA_BUNDLE_ROOT}/source_data/olympiadbench_hf}"

export RUNS_DIR="${RUNS_DIR:-${VDRA_HOME}/runs_qwen3}"
export RESULTS_DIR="${RESULTS_DIR:-${VDRA_HOME}/results_qwen3}"
export BANKS_DIR="${BANKS_DIR:-${VDRA_HOME}/banks_qwen3}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false

# The IGPO source is in the code checkout; reuse the separately installed B200 venv.
QWEN3_VENV="${QWEN3_VENV:-${VDRA_BUNDLE_ROOT}/project/.venv-igpo2}"
if [[ ! -f "${QWEN3_VENV}/bin/activate" ]]; then
  echo "[qwen3] missing IGPO venv: ${QWEN3_VENV}/bin/activate" >&2
  exit 2
fi
if [[ "${VIRTUAL_ENV:-}" != "${QWEN3_VENV}" ]]; then
  # shellcheck disable=SC1090
  source "${QWEN3_VENV}/bin/activate"
fi

# Load the project's standard defaults, then keep this runner on the venv above.
# An env.local.sh can still override data/model/run paths when needed.
source "${VDRA_HOME}/scripts/env.sh"
export CONDA_IGPO=""
export MODEL_PATH="${QWEN3_MODEL_PATH}"

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

# Serialize the first-time IGPO clone/patch because SPO and VDRA may start together.
mkdir -p "$(dirname "${IGPO_DIR}")"
exec 9>"${TMPDIR:-/tmp}/vdra-qwen3-igpo-install.lock"
flock 9
if [[ ! -d "${IGPO_DIR}/.git" ]]; then
  if [[ -e "${IGPO_DIR}" ]]; then
    echo "[qwen3] ${IGPO_DIR} exists but is not a git checkout; set IGPO_DIR to a fresh path" >&2
    exit 2
  fi
  git clone "${IGPO_REMOTE}" "${IGPO_DIR}"
fi
QWEN3_CODE_REV="$(git -C "${VDRA_HOME}" rev-parse HEAD)"
QWEN3_INSTALL_STAMP="${IGPO_DIR}/.vdra_qwen3_source_revision"
INSTALLED_REV="$(cat "${QWEN3_INSTALL_STAMP}" 2>/dev/null || true)"
if [[ "${INSTALLED_REV}" != "${QWEN3_CODE_REV}" ]]; then
  bash "${VDRA_HOME}/integrations/igpo/install.sh" "${IGPO_DIR}"
  printf '%s\n' "${QWEN3_CODE_REV}" > "${QWEN3_INSTALL_STAMP}"
fi
flock -u 9
exec 9>&-

mkdir -p "${RUNS_DIR}" "${RESULTS_DIR}" "${BANKS_DIR}"
echo "[qwen3] model: ${MODEL_PATH}"
echo "[qwen3] IGPO source: ${IGPO_DIR}"
echo "[qwen3] run outputs: ${RUNS_DIR}"
