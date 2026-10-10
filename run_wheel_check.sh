#!/usr/bin/env bash
# Local, wheels-lifted calibration using the existing guarded motor tool.
set -euo pipefail
WHEEL_CHECK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_CHECK_PYTHON="$WHEEL_CHECK_ROOT/outputs/field_yolopv2/venv/bin/python"
WHEEL_CHECK_TOOL="$WHEEL_CHECK_ROOT/car/autodrive/tools/check_wheel_directions.py"
command=("$WHEEL_CHECK_PYTHON" -u -B "$WHEEL_CHECK_TOOL" --vehicle-profile rosmaster_jetson_yolopv2 "$@")
if (( $# == 0 )); then exec "${command[@]}" --help; fi
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h ]]; then exec "${command[@]}"; fi
done
cd "$WHEEL_CHECK_ROOT"
mkdir -p outputs/manual_wheel_check
run_dir="$(mktemp -d "$WHEEL_CHECK_ROOT/outputs/manual_wheel_check/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
printf 'Console log: %s/console.log\n' "$run_dir"
exec > >(tee -a "$run_dir/console.log") 2>&1
unit=vehicle-serial-monitor.service
restore_needed=0
child_pid=''
restore() {
  local result=$?
  trap - EXIT
  if (( restore_needed )); then
    if sudo -n systemctl start "$unit"; then
      printf 'Restored serial monitor.\n'
    else
      printf 'FAILED to restore monitor. Run: sudo systemctl start %s\n' "$unit" >&2
      result=1
    fi
  fi
  printf 'Exit code: %s\n' "$result"
  printf '%s\n' "$result" > "$run_dir/exit-code.txt"
  exit "$result"
}
forward_signal() {
  if [[ -n "$child_pid" ]]; then
    kill -INT "$child_pid" 2>/dev/null || true
  else
    exit 130
  fi
}
trap restore EXIT
trap forward_signal INT TERM
if systemctl is-active --quiet "$unit"; then
  # Remember ownership before stop, so interruption also restores the monitor.
  restore_needed=1
  printf 'Temporarily releasing serial monitor.\n'
  sudo -n systemctl stop "$unit"
fi
# Reset the background shell's inherited SIGINT disposition before starting
# Python, whose KeyboardInterrupt executes the motor tool's stop/close finally.
python3 -c 'import os,signal,sys; signal.signal(signal.SIGINT,signal.SIG_DFL); os.execv(sys.argv[1],sys.argv[1:])' "${command[@]}" <&0 &
child_pid=$!
set +e
wait "$child_pid"
result=$?
while kill -0 "$child_pid" 2>/dev/null; do
  wait "$child_pid"
  result=$?
done
exit "$result"
