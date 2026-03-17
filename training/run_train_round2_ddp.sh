#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/round2_ddp_${STAMP}.log"

nohup torchrun \
  --standalone \
  --nproc_per_node=2 \
  training/train_lora.py \
  --config configs/lora_qwen3b_round2.yaml \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID=$PID"
echo "LOG_FILE=$LOG_FILE"
echo "$PID" > logs/round2_ddp_latest.pid
printf '%s' "$LOG_FILE" > logs/round2_ddp_latest.log
