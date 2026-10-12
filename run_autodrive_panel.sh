#!/usr/bin/env bash
# Compatibility for installed units. Canonical entry: ./vehicle panel.
set -euo pipefail
exec "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/vehicle" panel "$@"
