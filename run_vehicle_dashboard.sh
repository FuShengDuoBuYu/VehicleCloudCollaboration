#!/usr/bin/env bash
set -euo pipefail
DASHBOARD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CAR_PYTHON:-}" ]]; then
  DASHBOARD_PYTHON="$CAR_PYTHON"
elif [[ -x "$DASHBOARD_ROOT/outputs/field_yolopv2/venv/bin/python" ]]; then
  DASHBOARD_PYTHON="$DASHBOARD_ROOT/outputs/field_yolopv2/venv/bin/python"
elif [[ -x /home/pi/miniconda3/envs/car/bin/python ]]; then
  DASHBOARD_PYTHON=/home/pi/miniconda3/envs/car/bin/python
else
  DASHBOARD_PYTHON=python3
fi
cd "$DASHBOARD_ROOT"
export PYTHONPATH="$DASHBOARD_ROOT/car${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-web}" in
  web)
    shift
    exec "$DASHBOARD_PYTHON" -B -m autodrive.web.dashboard --host "${DASHBOARD_HOST:-0.0.0.0}" --port "${DASHBOARD_PORT:-8080}" \
      --runtime-dir "$DASHBOARD_ROOT/outputs/vehicle_dashboard/monitor" \
      --runtime-dir "$DASHBOARD_ROOT/outputs/onboard_runtime/rosmaster_jetson_yolopv2" \
      --runtime-dir "$DASHBOARD_ROOT/outputs/onboard_runtime" \
      --sensor-dir "$DASHBOARD_ROOT/outputs/vehicle_dashboard/sensors" \
      --session-socket "$DASHBOARD_ROOT/outputs/vehicle_dashboard/control.sock" "$@"
    ;;
  perception)
    shift
    exec "$DASHBOARD_PYTHON" -B -m autodrive.web.perception_monitor "$@"
    ;;
  serial)
    shift
    exec "$DASHBOARD_PYTHON" -B -m autodrive.web.sensor_publishers serial \
      --output-dir "$DASHBOARD_ROOT/outputs/vehicle_dashboard/sensors" "$@"
    ;;
  sensors)
    shift
    mkdir -p "$DASHBOARD_ROOT/outputs/vehicle_dashboard/sensors"
    exec docker run --rm --name vehicle-dashboard-sensors --network none --cap-drop ALL --cap-add DAC_OVERRIDE \
      --device /dev/rplidar:/dev/rplidar --device-cgroup-rule 'c 189:* rmw' \
      -v /dev/bus/usb:/dev/bus/usb \
      -v "$DASHBOARD_ROOT/car:/workspace/car:ro" \
      -v "$DASHBOARD_ROOT/outputs/vehicle_dashboard/sensors:/telemetry" \
      -e PYTHONPATH=/workspace/car -e ROS_DOMAIN_ID=67 \
      yahboomtechnology/ros-foxy:4.0.5 \
      /bin/bash -c 'source /opt/ros/foxy/setup.bash; for setup in /root/yahboomcar_ros2_ws/software/library_ws/install/setup.bash; do if [ -f "$setup" ]; then source "$setup"; fi; done; exec python3 -B -m autodrive.web.sensor_publishers ros --output-dir /telemetry --start-drivers "$@"' dashboard-sensors "$@"
    ;;
  *) echo 'Usage: run_vehicle_dashboard.sh {web|perception|serial|sensors}' >&2; exit 2 ;;
esac
