#!/usr/bin/env bash
# Pause only the collectors needed by the requested existing runtime. Restore
# those previously active after the child exits, preserving its exit status.
set -euo pipefail
VEHICLE_EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paused=()
child_pid=''
signal_forwarded=0
restore() {
  local result=$?
  trap - EXIT
  for unit in "${paused[@]}"; do
    sudo -n systemctl start "$unit" || printf 'Monitor restore failed: %s\n' "$unit" >&2
  done
  exit "$result"
}
forward_signal() {
  if [[ -n "$child_pid" && "$signal_forwarded" == 0 ]]; then
    signal_forwarded=1
    kill -INT "$child_pid" 2>/dev/null || true
  fi
}
trap restore EXIT
trap forward_signal INT TERM
units=(vehicle-perception-monitor.service)
for arg in "$@"; do
  if [[ "$arg" == --enable-motors ]]; then units+=(vehicle-serial-monitor.service); break; fi
done
for unit in "${units[@]}"; do
  if systemctl is-active --quiet "$unit"; then
    sudo -n systemctl stop "$unit"
    paused+=("$unit")
  fi
done
# Isolate the child from terminal group signals: only this wrapper forwards the
# first interruption, rather than delivering it twice during runtime cleanup.
python3 -c 'import os,signal,sys; os.setsid(); signal.signal(signal.SIGINT,signal.SIG_DFL); os.execv(sys.argv[1],sys.argv[1:])' "$VEHICLE_EXPERIMENT_ROOT/run_jetson_yolopv2.sh" "$@" &
child_pid=$!
set +e
wait "$child_pid"
result=$?
while kill -0 "$child_pid" 2>/dev/null; do wait "$child_pid"; result=$?; done
exit "$result"
