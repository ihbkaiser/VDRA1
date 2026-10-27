#!/usr/bin/env bash
# Experiment R6: TreeRL + VDRA (OpenRLHF), on one GPU (default GPU 5).
#   bash scripts/exp/R6_treerl_vdra.sh                 # GPU 5
#   GPU=5 SEEDS="42" bash scripts/exp/R6_treerl_vdra.sh
# Trains every seed with evaluation every EVAL_EVERY_N updates on the same GPU.
# Logs: ${RUNS_DIR}/_logs/R6_gpu<GPU>.log.  Safe to re-run: resumes where it stopped.
EXP=R6 GPU_DEFAULT=5 source "$(dirname "${BASH_SOURCE[0]}")/_one_gpu.sh"
