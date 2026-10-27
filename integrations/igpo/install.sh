#!/usr/bin/env bash
# Install the paper-faithful VDRA strategy into an igpo (treetune) checkout.
#
#   bash integrations/igpo/install.sh /path/to/igpo
#
# * creates branch `vdra-paper` from `origin/fix/budget-allocation-theory`
#   (unless VDRA_IGPO_BASE is set, e.g. VDRA_IGPO_BASE=main)
# * copies vdra_core/ into the repo root (importable through INGPO_ROOT on PYTHONPATH)
# * adds treetune/inference_strategies/vdra_inference_strategy.py and registers it
# * adds configs/vdra/*, polIter_deepseekR1Qwen_{vdra,spo_logged}_MATH.jsonnet
# Idempotent: re-running overwrites the copied files.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
IGPO="${1:?usage: install.sh /path/to/igpo}"
BASE="${VDRA_IGPO_BASE:-87e4061c998ebb3a348074a6f70b4599fb9f373e}"

cd "${IGPO}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "[install] ${IGPO} has uncommitted changes; commit or stash them first" >&2
  exit 1
fi
git fetch -q origin '+refs/heads/*:refs/remotes/origin/*' || true
if git rev-parse --verify -q vdra-paper >/dev/null; then
  git checkout -q vdra-paper
else
  git checkout -q -b vdra-paper "${BASE}"
fi

if git apply --check "${HERE}/deepspeed_local_rank.patch" 2>/dev/null; then
  git apply "${HERE}/deepspeed_local_rank.patch"
elif git apply -R --check "${HERE}/deepspeed_local_rank.patch" 2>/dev/null; then
  echo "[install] DeepSpeed local_rank CLI compatibility already applied"
else
  echo "[install] DeepSpeed local_rank compatibility patch does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/restem_metrics.patch" 2>/dev/null; then
  git apply "${HERE}/restem_metrics.patch"
elif git apply -R --check "${HERE}/restem_metrics.patch" 2>/dev/null; then
  echo "[install] RestEM Phase D metric hook already applied"
else
  echo "[install] RestEM metric hook does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/single_rank_vllm_ports.patch" 2>/dev/null; then
  git apply "${HERE}/single_rank_vllm_ports.patch"
elif git apply -R --check "${HERE}/single_rank_vllm_ports.patch" 2>/dev/null; then
  echo "[install] single-rank vLLM port fix already applied"
else
  echo "[install] single-rank vLLM port fix does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/episode_generation_timing.patch" 2>/dev/null; then
  git apply "${HERE}/episode_generation_timing.patch"
elif git apply -R --check "${HERE}/episode_generation_timing.patch" 2>/dev/null; then
  echo "[install] episode-generation timing instrumentation already applied"
else
  echo "[install] episode-generation timing instrumentation does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/vllm_cleanup_and_warning_noise.patch" 2>/dev/null; then
  git apply "${HERE}/vllm_cleanup_and_warning_noise.patch"
elif git apply -R --check "${HERE}/vllm_cleanup_and_warning_noise.patch" 2>/dev/null; then
  echo "[install] vLLM process cleanup and expected warning reduction already applied"
else
  echo "[install] vLLM process cleanup patch does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/logprobs_memory.patch" 2>/dev/null; then
  git apply "${HERE}/logprobs_memory.patch"
elif git apply -R --check "${HERE}/logprobs_memory.patch" 2>/dev/null; then
  echo "[install] conditional all-token logprob allocation already applied"
else
  echo "[install] logprob memory optimization does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/metric_sync.patch" 2>/dev/null; then
  git apply "${HERE}/metric_sync.patch"
elif git apply -R --check "${HERE}/metric_sync.patch" 2>/dev/null; then
  echo "[install] packed metric synchronization already applied"
else
  echo "[install] metric synchronization optimization does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/logprob_profile.patch" 2>/dev/null; then
  git apply "${HERE}/logprob_profile.patch"
elif git apply -R --check "${HERE}/logprob_profile.patch" 2>/dev/null; then
  echo "[install] split logprob profile instrumentation already applied"
else
  echo "[install] logprob profile instrumentation does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/logprob_forward_profile.patch" 2>/dev/null; then
  git apply "${HERE}/logprob_forward_profile.patch"
elif git apply -R --check "${HERE}/logprob_forward_profile.patch" 2>/dev/null; then
  echo "[install] detailed logprob forward profiling already applied"
else
  echo "[install] detailed logprob forward profiling does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/logprob_single_rank_gather.patch" 2>/dev/null; then
  git apply "${HERE}/logprob_single_rank_gather.patch"
elif git apply -R --check "${HERE}/logprob_single_rank_gather.patch" 2>/dev/null; then
  echo "[install] single-rank logprob gather fast path already applied"
else
  echo "[install] single-rank logprob gather fast path does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/internal_eval_guard.patch" 2>/dev/null; then
  git apply "${HERE}/internal_eval_guard.patch"
elif git apply -R --check "${HERE}/internal_eval_guard.patch" 2>/dev/null; then
  echo "[install] empty-pipeline evaluation guard already applied"
else
  echo "[install] internal evaluation guard does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/resume_after_early_stop.patch" 2>/dev/null; then
  git apply "${HERE}/resume_after_early_stop.patch"
elif git apply -R --check "${HERE}/resume_after_early_stop.patch" 2>/dev/null; then
  echo "[install] early-stop resume guard already applied"
else
  echo "[install] early-stop resume guard does not apply at ${BASE}" >&2
  exit 1
fi

if git apply --check "${HERE}/ppo_phase_d_metrics.patch" 2>/dev/null; then
  git apply "${HERE}/ppo_phase_d_metrics.patch"
elif git apply -R --check "${HERE}/ppo_phase_d_metrics.patch" 2>/dev/null; then
  echo "[install] PPO Phase D metrics hook already applied"
else
  echo "[install] PPO Phase D metrics hook does not apply at ${BASE}" >&2
  exit 1
fi

append_line() {
  # append_line <file> <line>: add once, with a newline first if the file lacks one
  local f="$1" line="$2"
  grep -qxF "${line}" "${f}" && return 0
  [[ -s "${f}" && -n "$(tail -c1 "${f}")" ]] && echo >> "${f}"
  echo "${line}" >> "${f}"
}

rm -rf "${IGPO}/vdra_core" && cp -r "${PKG_ROOT}/vdra_core" "${IGPO}/vdra_core" && find "${IGPO}/vdra_core" -name __pycache__ -prune -exec rm -rf {} +
cp "${HERE}/treetune/inference_strategies/vdra_inference_strategy.py" \
   "${IGPO}/treetune/inference_strategies/vdra_inference_strategy.py"
INIT="${IGPO}/treetune/inference_strategies/__init__.py"
append_line "${INIT}" "from .vdra_inference_strategy import VDRAInferenceStrategy, SPOLoggedInferenceStrategy"

cp "${HERE}/treetune/episode_generators/resumable_hybrid_episode_generator.py" \
   "${IGPO}/treetune/episode_generators/resumable_hybrid_episode_generator.py"
EGINIT="${IGPO}/treetune/episode_generators/__init__.py"
# repair a glued line left by an earlier installer version, then append once
sed -i 's/InGPOEpisodeGeneratorfrom \.resumable/InGPOEpisodeGenerator\nfrom .resumable/' "${EGINIT}"
append_line "${EGINIT}" "from .resumable_hybrid_episode_generator import ResumableHybridEpisodeGenerator"

mkdir -p "${IGPO}/configs/vdra"
cp "${HERE}"/configs/vdra/* "${IGPO}/configs/vdra/"
cp "${HERE}"/configs/polIter_deepseekR1Qwen_vdra_MATH.jsonnet "${IGPO}/configs/"
cp "${HERE}"/configs/polIter_deepseekR1Qwen_spo_logged_MATH.jsonnet "${IGPO}/configs/"

git add -A vdra_core treetune/inference_strategies treetune/episode_generators configs/vdra \
  configs/polIter_deepseekR1Qwen_vdra_MATH.jsonnet configs/polIter_deepseekR1Qwen_spo_logged_MATH.jsonnet \
  treetune/main.py treetune/common/vllm_server.py \
  treetune/inference_strategies/tree_inference/expansion.py \
  treetune/trainers/restem_trainer.py treetune/trainers/ppo_trainer.py
if ! git diff --cached --quiet; then
  git -c user.name="${GIT_AUTHOR_NAME:-vdra}" -c user.email="${GIT_AUTHOR_EMAIL:-vdra@localhost}" \
    commit -q -m "Add paper-faithful VDRA strategy (vdra_core, configs, spo_logged)"
fi
echo "[install] igpo branch vdra-paper at $(git rev-parse --short HEAD)"
