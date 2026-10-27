#!/usr/bin/env bash
# Clone TreePO at the audited commit, apply the VDRA divergence policy and
# make vdra_core importable from the verl workers.
#   bash integrations/treepo/install.sh /path/to/TreePO
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
DEST="${1:?usage: install.sh /path/to/TreePO}"
COMMIT="${TREEPO_COMMIT:-f05552394e734ea2838cf9e3b8191f5093319dcf}"
if [[ ! -e "${DEST}/.git" ]]; then
  git clone https://github.com/multimodal-art-projection/TreePO.git "${DEST}"
fi
cd "${DEST}"
git checkout -q "${COMMIT}"
if git apply --check "${HERE}/vdra_treepo.patch" 2>/dev/null; then
  git apply "${HERE}/vdra_treepo.patch"
  echo "[treepo] patch applied"
elif git apply -R --check "${HERE}/vdra_treepo.patch" 2>/dev/null; then
  echo "[treepo] patch already applied"
else
  echo "[treepo] patch does not apply at ${COMMIT}" >&2; exit 1
fi
if git apply --unidiff-zero --check "${HERE}/training_stability.patch" 2>/dev/null; then
  git apply --unidiff-zero "${HERE}/training_stability.patch"
  echo "[treepo] empty-response fallback and prompt-logprob cache fixes applied"
elif git apply --unidiff-zero -R --check "${HERE}/training_stability.patch" 2>/dev/null; then
  echo "[treepo] training stability fixes already applied"
else
  echo "[treepo] training stability patch does not apply at ${COMMIT} + VDRA patch" >&2; exit 1
fi
if git apply --unidiff-zero --check "${HERE}/ragged_dataproto.patch" 2>/dev/null; then
  git apply --unidiff-zero "${HERE}/ragged_dataproto.patch"
  echo "[treepo] ragged non-tensor batch concatenation fixed"
elif git apply --unidiff-zero -R --check "${HERE}/ragged_dataproto.patch" 2>/dev/null; then
  echo "[treepo] ragged non-tensor batch concatenation already fixed"
else
  echo "[treepo] DataProto concat patch does not apply at ${COMMIT}" >&2; exit 1
fi
if git apply --unidiff-zero --check "${HERE}/reward_timeout.patch" 2>/dev/null; then
  git apply --unidiff-zero "${HERE}/reward_timeout.patch"
  echo "[treepo] math verifier warm-start timeout increased to 30 seconds"
elif git apply --unidiff-zero -R --check "${HERE}/reward_timeout.patch" 2>/dev/null; then
  echo "[treepo] math verifier reward timeout already fixed"
else
  echo "[treepo] reward timeout patch does not apply at ${COMMIT}" >&2; exit 1
fi
if git apply --check "${HERE}/metric_sync.patch" 2>/dev/null; then
  git apply "${HERE}/metric_sync.patch"
  echo "[treepo] packed data-metric synchronization applied"
elif git apply -R --check "${HERE}/metric_sync.patch" 2>/dev/null; then
  echo "[treepo] packed data-metric synchronization already applied"
else
  echo "[treepo] metric synchronization patch does not apply at ${COMMIT}" >&2; exit 1
fi
if git apply --check "${HERE}/advantage_cpu_staging.patch" 2>/dev/null; then
  git apply "${HERE}/advantage_cpu_staging.patch"
  echo "[treepo] one-shot tree-advantage score staging applied"
elif git apply -R --check "${HERE}/advantage_cpu_staging.patch" 2>/dev/null; then
  echo "[treepo] one-shot tree-advantage score staging already applied"
else
  echo "[treepo] tree-advantage score staging patch does not apply at ${COMMIT}" >&2; exit 1
fi
if git apply --check "${HERE}/vllm_text_mm_limit.patch" 2>/dev/null; then
  git apply "${HERE}/vllm_text_mm_limit.patch"
  echo "[treepo] text-only vLLM multimodal-limit compatibility applied"
elif git apply -R --check "${HERE}/vllm_text_mm_limit.patch" 2>/dev/null; then
  echo "[treepo] text-only vLLM multimodal-limit compatibility already applied"
else
  echo "[treepo] text-only vLLM multimodal-limit patch does not apply at ${COMMIT}" >&2; exit 1
fi
if git apply --check "${HERE}/vllm_text_mm_cache.patch" 2>/dev/null; then
  git apply "${HERE}/vllm_text_mm_cache.patch"
  echo "[treepo] text-model vLLM multimodal-cache compatibility applied"
elif git apply -R --check "${HERE}/vllm_text_mm_cache.patch" 2>/dev/null; then
  echo "[treepo] text-model vLLM multimodal-cache compatibility already applied"
else
  echo "[treepo] text-model vLLM multimodal-cache patch does not apply at ${COMMIT}" >&2; exit 1
fi
rm -rf "${DEST}/vdra_core" && cp -r "${PKG_ROOT}/vdra_core" "${DEST}/vdra_core"
cp "${HERE}/vdra_math_reward.py" "${DEST}/vdra_math_reward.py"
echo "[treepo] ready at ${DEST} ($(git rev-parse --short HEAD) + VDRA, rollout, ragged-batch, reward, and vLLM compatibility patches)"
