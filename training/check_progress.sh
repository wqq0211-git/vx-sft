#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [[ -f logs/round2_ddp_latest.log ]]; then
  LOG_FILE=$(cat logs/round2_ddp_latest.log)
  echo "LOG_FILE=$LOG_FILE"
else
  echo "No latest log file found." >&2
  exit 1
fi

if [[ -f logs/round2_ddp_latest.pid ]]; then
  PID=$(cat logs/round2_ddp_latest.pid)
  echo "PID=$PID"
  ps -p "$PID" -o pid=,etime=,pcpu=,pmem=,command= || true
fi

echo '--- tail ---'
tail -n 40 "$LOG_FILE"

echo '--- gpu ---'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
  echo 'nvidia-smi not found'
fi
