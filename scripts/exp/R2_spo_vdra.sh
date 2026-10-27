#!/usr/bin/env bash
# Experiment R2: SPO + VDRA (treetune / igpo), on one GPU (default GPU 1).
#   bash scripts/exp/R2_spo_vdra.sh                 # GPU 1
#   GPU=5 SEEDS="42" bash scripts/exp/R2_spo_vdra.sh
# Trains every seed with evaluation every EVAL_EVERY_N updates on the same GPU.
# Logs: ${RUNS_DIR}/_logs/R2_gpu<GPU>.log.  Safe to re-run: resumes where it stopped.
EXP=R2 GPU_DEFAULT=1 source "$(dirname "${BASH_SOURCE[0]}")/_one_gpu.sh"
