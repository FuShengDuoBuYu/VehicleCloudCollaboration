#!/usr/bin/env bash
set -euo pipefail

# Start only the read-only dashboard. Driving uses run_vehicle_experiment.sh.
VEHICLE_WEB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$VEHICLE_WEB_ROOT/run_vehicle_dashboard.sh" web "$@"
