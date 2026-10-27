#!/usr/bin/env bash
# Experiment R5: TreeRL baseline (OpenRLHF), on one GPU (default GPU 4).
#   bash scripts/exp/R5_treerl.sh                 # GPU 4
#   GPU=5 SEEDS="42" bash scripts/exp/R5_treerl.sh
# Trains every seed with evaluation every EVAL_EVERY_N updates on the same GPU, then the compute-matched continuation once R6 has finished.
# Logs: ${RUNS_DIR}/_logs/R5_gpu<GPU>.log.  Safe to re-run: resumes where it stopped.
EXP=R5 GPU_DEFAULT=4 source "$(dirname "${BASH_SOURCE[0]}")/_one_gpu.sh"
