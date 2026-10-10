#!/usr/bin/env bash
# One real, bounded trial followed by an automatic motor-disabled replay.
set -euo pipefail
OUTER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
seconds="${1:-30}"
if [[ "$seconds" == --help || "$seconds" == -h ]]; then
  echo 'Usage: bash run_outer_loop.sh [1..120 seconds; default 30]'
  echo 'MOVES THE VEHICLE at up to 30 PWM, records, stops, then replays without motors.'
  exit 0
fi
if (( $# > 1 )) || [[ ! "$seconds" =~ ^[0-9]+$ ]] || (( 10#$seconds < 1 || 10#$seconds > 120 )); then
  echo 'Runtime must be an integer between 1 and 120 seconds.' >&2; exit 2
fi
cd "$OUTER_ROOT"
started="$(date +%s.%N)"
unit=vehicle-perception-monitor.service
restore_needed=0
child_pid=''
interrupted=0
restore() {
  local result=$?
  trap - EXIT
  if (( restore_needed )); then
    sudo -n systemctl start "$unit" || result=1
  fi
  exit "$result"
}
forward_signal() {
  interrupted=1
  if [[ -n "$child_pid" ]]; then kill -INT "$child_pid" 2>/dev/null || true; else exit 130; fi
}
trap restore EXIT
trap forward_signal INT TERM
if systemctl is-active --quiet "$unit"; then
  restore_needed=1
  sudo -n systemctl stop "$unit"
fi
run_child() {
  python3 -c 'import os,signal,sys; signal.signal(signal.SIGINT,signal.SIG_DFL); os.execvp(sys.argv[1],sys.argv[1:])' "$@" &
  child_pid=$!
  local result=0
  wait "$child_pid" || result=$?
  while kill -0 "$child_pid" 2>/dev/null; do wait "$child_pid" || result=$?; done
  child_pid=''
  return "$result"
}
printf 'REAL YOLO outer-loop trial: 30 PWM limit, %s seconds. Ctrl+C stops.\n' "$seconds"
drive_result=0
run_child bash "$OUTER_ROOT/run_vehicle_experiment.sh" \
  --vehicle-profile rosmaster_jetson_yolopv2_visual_feedback \
  --field-trial --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE \
  --max-runtime-seconds "$seconds" || drive_result=$?
if (( interrupted )); then exit 130; fi
if (( drive_result != 0 )); then echo 'Driving failed; inspect saved logs before another trial.' >&2; fi
printf 'Vehicle trial ended. Replaying recorded video with motors DISABLED.\n'
run_child "$OUTER_ROOT/outputs/field_yolopv2/venv/bin/python" -B \
  "$OUTER_ROOT/car/autodrive/tools/replay_outer_loop.py" \
  --status "$OUTER_ROOT/outputs/onboard_runtime/rosmaster_jetson_yolopv2_visual_feedback/status.json" --after "$started" \
  --profile recorded
exit "$drive_result"
