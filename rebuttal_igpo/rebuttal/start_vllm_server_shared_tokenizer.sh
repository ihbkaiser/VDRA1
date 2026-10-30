#!/usr/bin/env bash
set -euo pipefail

GPU_IDX=0
GPU_MEM_UTILIZATION=0.9
MAX_NUM_SEQS=256
ENABLE_PREFIX_CACHING=false
DISABLE_SLIDING_WINDOW=false
DISABLE_FRONTEND_MULTIPROCESSING=false
MAX_MODEL_LEN=""
DTYPE="${VLLM_DTYPE:-auto}"
MODEL=""
PORT=""
SEED=42
SWAP_SPACE=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --swap-space) SWAP_SPACE="$2"; shift 2 ;;
    --gpu-idx) GPU_IDX="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM_UTILIZATION="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --enable-prefix-caching) ENABLE_PREFIX_CACHING=true; shift ;;
    --disable-sliding-window) DISABLE_SLIDING_WINDOW=true; shift ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --disable-frontend-multiprocessing) DISABLE_FRONTEND_MULTIPROCESSING=true; shift ;;
    *) echo "Unknown parameter passed: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" || -z "$PORT" || -z "${APP_EVAL_TOKENIZER:-}" ]]; then
  echo "model, port and APP_EVAL_TOKENIZER are required" >&2
  exit 2
fi

ARGS=(
  --model "$MODEL"
  --tokenizer "$APP_EVAL_TOKENIZER"
  --host 127.0.0.1
  --port "$PORT"
  --seed "$SEED"
  --gpu-memory-utilization "$GPU_MEM_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --dtype "$DTYPE"
)
if [[ -n "$SWAP_SPACE" ]]; then ARGS+=(--swap-space "$SWAP_SPACE"); fi
if [[ "$ENABLE_PREFIX_CACHING" == true ]]; then ARGS+=(--enable-prefix-caching); fi
if [[ "$DISABLE_SLIDING_WINDOW" == true ]]; then ARGS+=(--disable-sliding-window); fi
if [[ -n "$MAX_MODEL_LEN" ]]; then ARGS+=(--max-model-len "$MAX_MODEL_LEN"); fi
if [[ "$DISABLE_FRONTEND_MULTIPROCESSING" == true ]]; then ARGS+=(--disable-frontend-multiprocessing); fi

CUDA_VISIBLE_DEVICES="$GPU_IDX" python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
