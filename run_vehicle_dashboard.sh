#!/usr/bin/env bash
# Compatibility for installed units. Canonical entry: ./vehicle dashboard/monitor.
set -euo pipefail
COMPAT_ROOT="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
mode="${1:-web}"
if (( $# )); then shift; fi
case "$mode" in
  web) exec "$COMPAT_ROOT/vehicle" dashboard "$@" ;;
  perception|serial|sensors) exec "$COMPAT_ROOT/vehicle" monitor "$mode" "$@" ;;
  *) echo 'Use ./vehicle --help' >&2; exit 2 ;;
esac
