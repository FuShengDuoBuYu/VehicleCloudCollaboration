#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${CAR_PYTHON:-/home/pi/miniconda3/envs/car/bin/python}"

cd "$REPO_DIR"

exec "$PYTHON_BIN" car/autodrive/run_lcc_web.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --host "${LCC_HOST:-0.0.0.0}" \
  --port "${LCC_PORT:-8080}" \
  --default-max-runtime-seconds "${LCC_MAX_RUNTIME_SECONDS:-120}" \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE
