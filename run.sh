#!/usr/bin/env bash
# Airyn training launcher
# Usage:
#   ./run.sh                     # auto-detect GPUs
#   ./run.sh 1                   # single GPU
#   ./run.sh 2                   # 2 GPUs
#   ITERATIONS=100 ./run.sh 1    # override hyperparams via env

set -euo pipefail
cd "$(dirname "$0")"

NPROC="${1:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"
VENV="venv/Scripts"

# PyTorch nightly on Windows lacks libuv — must be set before torchrun starts
export USE_LIBUV=0

echo "=== Airyn Training ==="
echo "GPUs: $NPROC"
echo "======================"

exec "$VENV/torchrun" \
    --standalone \
    --nproc_per_node="$NPROC" \
    airyn/train.py
