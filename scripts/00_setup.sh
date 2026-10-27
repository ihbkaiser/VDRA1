#!/usr/bin/env bash
# Clone / patch the three codebases.  Create one python env per framework
# yourself (they pin different torch / vLLM versions):
#   igpo:   pip install -r $IGPO_DIR/requirements.txt
#   TreePO: see $TREEPO_DIR/recipe/treepo/README.md (verl 0.3.1.dev, vLLM 0.8.5)
#   TreeRL: pip install -r $TREERL_DIR/requirements.txt
# and point CONDA_IGPO / CONDA_TREEPO / CONDA_TREERL at them in scripts/env.local.sh.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
mkdir -p "${WORK}" "${RUNS_DIR}/_logs" "${RESULTS_DIR}" "${BANKS_DIR}"
if [[ ! -d "${IGPO_DIR}/.git" ]]; then run git clone "${IGPO_REMOTE}" "${IGPO_DIR}"; fi
run bash "${VDRA_HOME}/integrations/igpo/install.sh" "${IGPO_DIR}"
run bash "${VDRA_HOME}/integrations/treepo/install.sh" "${TREEPO_DIR}"
run bash "${VDRA_HOME}/integrations/treerl/install.sh" "${TREERL_DIR}"
log "setup complete; next: scripts/01_prepare_data.sh"
