#!/usr/bin/env bash
# Experiment R3: TreePO baseline (verl), on one GPU (default GPU 2).
#   bash scripts/exp/R3_treepo.sh                 # GPU 2
#   GPU=5 SEEDS="42" bash scripts/exp/R3_treepo.sh
# Trains every seed with evaluation every EVAL_EVERY_N updates on the same GPU, then the compute-matched continuation once R4 has finished.
# Logs: ${RUNS_DIR}/_logs/R3_gpu<GPU>.log.  Safe to re-run: resumes where it stopped.
EXP=R3 GPU_DEFAULT=2 source "$(dirname "${BASH_SOURCE[0]}")/_one_gpu.sh"
