#!/usr/bin/env bash
# Compatibility for already-loaded supervisors. Canonical entry: ./vehicle drive.
set -euo pipefail
exec "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/vehicle" drive "$@"
