#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${CAR_PYTHON:-$REPO_DIR/outputs/field_yolopv2/venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Jetson Python environment missing: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -r "$REPO_DIR/car/longtail/models/yolopv2.pt" ]]; then
  echo "YOLOPv2 weights missing; see docs/jetson-yolopv2.md" >&2
  exit 1
fi
cd "$REPO_DIR"
exec "$PYTHON_BIN" car/autodrive/run_onboard.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --vehicle-profile rosmaster_jetson_yolopv2 \
  --max-runtime-seconds 30 "$@"
