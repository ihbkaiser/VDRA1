#!/usr/bin/env bash
# Experiment R1: SPO baseline (treetune / igpo), on one GPU (default GPU 0).
#   bash scripts/exp/R1_spo.sh                 # GPU 0
#   GPU=5 SEEDS="42" bash scripts/exp/R1_spo.sh
# Trains every seed with evaluation every EVAL_EVERY_N updates on the same GPU, then the compute-matched continuation once R2 has finished.
# Logs: ${RUNS_DIR}/_logs/R1_gpu<GPU>.log.  Safe to re-run: resumes where it stopped.
EXP=R1 GPU_DEFAULT=0 source "$(dirname "${BASH_SOURCE[0]}")/_one_gpu.sh"
