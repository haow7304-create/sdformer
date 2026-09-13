#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# PHYSICAL_GPU is the physical GPU number shown by nvidia-smi.  After it is
# masked here, the API process sees that GPU as cuda:0.
PHYSICAL_GPU="${PHYSICAL_GPU:-6}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-/home/liangjing/miniconda3/envs/llm/bin/python}"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export FORECAST_DEVICE="cuda:0"
export SDFORMER_PROJECT_ROOT="$PROJECT_ROOT"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m uvicorn forecast_api:app \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --workers 1
