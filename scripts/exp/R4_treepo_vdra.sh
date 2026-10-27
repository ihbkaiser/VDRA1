#!/usr/bin/env bash
# Experiment R4: TreePO + VDRA (verl), on one GPU (default GPU 3).
#   bash scripts/exp/R4_treepo_vdra.sh                 # GPU 3
#   GPU=5 SEEDS="42" bash scripts/exp/R4_treepo_vdra.sh
# Trains every seed with evaluation every EVAL_EVERY_N updates on the same GPU.
# Logs: ${RUNS_DIR}/_logs/R4_gpu<GPU>.log.  Safe to re-run: resumes where it stopped.
EXP=R4 GPU_DEFAULT=3 source "$(dirname "${BASH_SOURCE[0]}")/_one_gpu.sh"
