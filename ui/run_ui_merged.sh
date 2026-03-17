#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

python3 ui/app.py \
  --base-model merged/vx-sft-merged \
  --port 7860
