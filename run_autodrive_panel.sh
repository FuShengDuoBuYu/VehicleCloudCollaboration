#!/usr/bin/env bash
# Arms the local panel supervisor; hardware starts only on a later Start click.
set -euo pipefail
PANEL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PANEL_PYTHON="${CAR_PYTHON:-$PANEL_ROOT/outputs/field_yolopv2/venv/bin/python}"
cd "$PANEL_ROOT"
export PYTHONPATH="$PANEL_ROOT/car${PYTHONPATH:+:$PYTHONPATH}"
exec "$PANEL_PYTHON" -B -m autodrive.web.drive_sessions "$@"
