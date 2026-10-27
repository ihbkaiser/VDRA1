#!/usr/bin/env bash
# Start the six experiments in the background, one per GPU (0..5 by default), and return.
#   bash scripts/exp/launch_all.sh            # GPUs 0 1 2 3 4 5
#   EXP_GPUS="2 3 4 5 6 7" bash scripts/exp/launch_all.sh
# Each keeps running after you log out; follow with: tail -f ${RUNS_DIR}/_logs/R*_gpu*.log
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a gpus <<< "${EXP_GPUS:-0 1 2 3 4 5}"
(( ${#gpus[@]} == 6 )) || { echo "EXP_GPUS needs 6 GPU ids" >&2; exit 2; }
source "${D}/../env.sh"; mkdir -p "${RUNS_DIR}/_logs"
i=0
for f in R1_spo R2_spo_vdra R3_treepo R4_treepo_vdra R5_treerl R6_treerl_vdra; do
  GPU="${gpus[$i]}" setsid nohup bash "${D}/${f}.sh" > "${RUNS_DIR}/_logs/${f}.launcher.out" 2>&1 < /dev/null &
  echo "${f}: GPU ${gpus[$i]}, pid $!"
  i=$((i + 1))
done
