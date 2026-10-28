#!/usr/bin/env bash
# Clone TreeRL at the audited commit, apply the VDRA fork-allocation hook,
# local-verifier switch, training-speed patches, and make vdra_core importable.
#   bash integrations/treerl/install.sh /path/to/TreeRL
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
DEST="${1:?usage: install.sh /path/to/TreeRL}"
COMMIT="${TREERL_COMMIT:-63ea99e0253cc320e53b2b47eb1bcfa8adafd666}"
VDRA_PATCH="${HERE}/vdra_treerl.patch"
SPEED_PATCH="${HERE}/treerl_speed.patch"
REPEAT_PATCH="${HERE}/repeat_detector.patch"
STATUS_PATCH="${HERE}/reinforce_status_sync.patch"
UPDATE_SPEED_PATCH="${HERE}/reinforce_update_speed.patch"
REPLAY_PATCH="${HERE}/replay_info_staging.patch"
TRANSFORMERS_PATCH="${HERE}/transformers_451.patch"
TRANSFORMERS_EMPTY_INIT_PATCH="${HERE}/transformers_451_empty_init.patch"
API_CONFIG_PATCH="${HERE}/api_config_optional.patch"
PROMPT_JSON_PATCH="${HERE}/prompt_json_shape.patch"
MCTS_REMAINDER_PATCH="${HERE}/mcts_inference_remainder.patch"
if [[ ! -e "${DEST}/.git" ]]; then
  git clone https://github.com/THUDM/TreeRL.git "${DEST}"
fi
cd "${DEST}"
git checkout -q "${COMMIT}"
if git apply -R --check "${TRANSFORMERS_PATCH}" 2>/dev/null; then
  git apply -R "${TRANSFORMERS_PATCH}"
fi
if git apply -R --check "${REPLAY_PATCH}" 2>/dev/null; then
  git apply -R "${REPLAY_PATCH}"
fi
if git apply -R --check "${UPDATE_SPEED_PATCH}" 2>/dev/null; then
  git apply -R "${UPDATE_SPEED_PATCH}"
fi
if git apply -R --check "${STATUS_PATCH}" 2>/dev/null; then
  # Remove the follow-on status hunk before replaying the existing patch stack.
  git apply -R "${STATUS_PATCH}"
fi
if git apply -R --check "${REPEAT_PATCH}" 2>/dev/null; then
  # Remove this follow-on hunk before checking/replaying the speed patch, whose
  # call-site context includes the original repeated-pattern expression.
  git apply -R "${REPEAT_PATCH}"
fi
if git apply -R --check "${SPEED_PATCH}" 2>/dev/null; then
  # Temporarily remove the speed patch before checking the original VDRA patch;
  # one speed hunk changes a line that is also part of the VDRA patch context.
  git apply -R "${SPEED_PATCH}"
  if git apply -R --check "${VDRA_PATCH}" 2>/dev/null; then
    git apply "${SPEED_PATCH}"
    echo "[treerl] VDRA and speed patches already applied"
  else
    git apply "${SPEED_PATCH}"
    echo "[treerl] speed patch is applied but the VDRA patch is missing or changed" >&2; exit 1
  fi
else
  if git apply --check "${VDRA_PATCH}" 2>/dev/null; then
    git apply "${VDRA_PATCH}"
    echo "[treerl] VDRA patch applied"
  elif git apply -R --check "${VDRA_PATCH}" 2>/dev/null; then
    echo "[treerl] VDRA patch already applied"
  else
    echo "[treerl] VDRA patch does not apply at ${COMMIT}" >&2; exit 1
  fi
  if git apply --check "${SPEED_PATCH}" 2>/dev/null; then
    git apply "${SPEED_PATCH}"
    echo "[treerl] speed patch applied"
  elif git apply -R --check "${SPEED_PATCH}" 2>/dev/null; then
    echo "[treerl] speed patch already applied"
  else
    echo "[treerl] speed patch does not apply at ${COMMIT} + VDRA patch" >&2; exit 1
  fi
fi
if git apply --check "${REPEAT_PATCH}" 2>/dev/null; then
  git apply "${REPEAT_PATCH}"
  echo "[treerl] repeated-pattern fast path applied"
elif git apply -R --check "${REPEAT_PATCH}" 2>/dev/null; then
  echo "[treerl] repeated-pattern fast path already applied"
else
  echo "[treerl] repeated-pattern patch does not apply at ${COMMIT} + VDRA + speed patches" >&2; exit 1
fi
if git apply --check "${STATUS_PATCH}" 2>/dev/null; then
  git apply "${STATUS_PATCH}"
  echo "[treerl] ReinforceTrainer status synchronization patch applied"
elif git apply -R --check "${STATUS_PATCH}" 2>/dev/null; then
  echo "[treerl] ReinforceTrainer status synchronization patch already applied"
else
  echo "[treerl] ReinforceTrainer status synchronization patch does not apply at ${COMMIT} + VDRA + speed + repeat patches" >&2; exit 1
fi
if git apply --check "${UPDATE_SPEED_PATCH}" 2>/dev/null; then
  git apply "${UPDATE_SPEED_PATCH}"
  echo "[treerl] ReinforceTrainer update batching and cached gradient norm patch applied"
elif git apply -R --check "${UPDATE_SPEED_PATCH}" 2>/dev/null; then
  echo "[treerl] ReinforceTrainer update batching and cached gradient norm patch already applied"
else
  echo "[treerl] ReinforceTrainer update speed patch does not apply after status synchronization at ${COMMIT}" >&2; exit 1
fi
if git apply --check "${REPLAY_PATCH}" 2>/dev/null; then
  git apply "${REPLAY_PATCH}"
  echo "[treerl] replay-buffer info staging patch applied"
elif git apply -R --check "${REPLAY_PATCH}" 2>/dev/null; then
  echo "[treerl] replay-buffer info staging patch already applied"
else
  echo "[treerl] replay-buffer info staging patch does not apply at ${COMMIT} + VDRA + speed + repeat + status patches" >&2; exit 1
fi
if git apply --check "${TRANSFORMERS_PATCH}"; then
  git apply "${TRANSFORMERS_PATCH}"
  echo "[treerl] Transformers 4.51 DeepSpeed import patch applied"
elif git apply -R --check "${TRANSFORMERS_PATCH}"; then
  echo "[treerl] Transformers 4.51 DeepSpeed import patch already applied"
else
  echo "[treerl] Transformers 4.51 DeepSpeed import patch does not apply at ${COMMIT} + VDRA patch stack" >&2; exit 1
fi
if git apply --check "${TRANSFORMERS_EMPTY_INIT_PATCH}"; then
  git apply "${TRANSFORMERS_EMPTY_INIT_PATCH}"
  echo "[treerl] Transformers 4.51 empty_init compatibility patch applied"
elif git apply -R --check "${TRANSFORMERS_EMPTY_INIT_PATCH}"; then
  echo "[treerl] Transformers 4.51 empty_init compatibility patch already applied"
else
  echo "[treerl] Transformers 4.51 empty_init compatibility patch does not apply at ${COMMIT} + VDRA patch stack" >&2; exit 1
fi
if git apply --check "${API_CONFIG_PATCH}"; then
  git apply "${API_CONFIG_PATCH}"
  echo "[treerl] optional API configuration and pure-binary MCTS patch applied"
elif git apply -R --check "${API_CONFIG_PATCH}"; then
  echo "[treerl] optional API configuration and pure-binary MCTS patch already applied"
else
  echo "[treerl] optional API configuration patch does not apply at ${COMMIT} + VDRA patch stack" >&2; exit 1
fi
if git apply --check "${PROMPT_JSON_PATCH}"; then
  git apply "${PROMPT_JSON_PATCH}"
  echo "[treerl] prompt JSON shape patch applied"
elif git apply -R --check "${PROMPT_JSON_PATCH}"; then
  echo "[treerl] prompt JSON shape patch already applied"
else
  echo "[treerl] prompt JSON shape patch does not apply at ${COMMIT} + VDRA patch stack" >&2; exit 1
fi
if git apply --check "${MCTS_REMAINDER_PATCH}"; then
  git apply "${MCTS_REMAINDER_PATCH}"
  echo "[treerl] MCTS final partial-batch patch applied"
elif git apply -R --check "${MCTS_REMAINDER_PATCH}"; then
  echo "[treerl] MCTS final partial-batch patch already applied"
else
  echo "[treerl] MCTS final partial-batch patch does not apply at ${COMMIT} + VDRA patch stack" >&2; exit 1
fi
rm -rf "${DEST}/vdra_core" && cp -r "${PKG_ROOT}/vdra_core" "${DEST}/vdra_core"
echo "[treerl] ready at ${DEST} ($(git rev-parse --short HEAD) + VDRA + speed + repeat-detector + status + update-speed + replay-info + Transformers + empty_init compatibility + optional API configuration + prompt JSON shape + MCTS remainder patches)"
