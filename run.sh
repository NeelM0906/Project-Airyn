#!/usr/bin/env bash
# Airyn training launcher
# Usage:
#   ./run.sh                     # auto-detect GPUs
#   ./run.sh 1                   # single GPU
#   ./run.sh 5                   # 5 GPUs
#   ITERATIONS=100 ./run.sh 1    # override hyperparams via env

set -euo pipefail
cd "$(dirname "$0")"

NPROC="${1:-$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

echo "=== Airyn Training ==="
echo "GPUs: $NPROC"
echo "======================"

exec torchrun \
    --standalone \
    --nproc_per_node="$NPROC" \
    airyn/train.py
