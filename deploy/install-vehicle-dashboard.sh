#!/usr/bin/env bash
# Deliberate installer: validates all files before installing new units.
set -euo pipefail
DASHBOARD_INSTALL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$DASHBOARD_INSTALL_ROOT" == /home/jetson/VehicleCloudCollaboration ]] || { echo 'Install from the main Jetson project after review.' >&2; exit 2; }
[[ -x "$DASHBOARD_INSTALL_ROOT/outputs/field_yolopv2/venv/bin/python" ]] || exit 2
units=(vehicle-dashboard vehicle-perception-monitor vehicle-serial-monitor vehicle-sensors)
for name in "${units[@]}"; do
  target="/etc/systemd/system/$name.service"
  if [[ -e "$target" ]] && ! cmp -s "$DASHBOARD_INSTALL_ROOT/deploy/$name.service" "$target"; then
    echo "Existing service differs: $target; review it before replacing." >&2; exit 2
  fi
done
unit_files=()
for name in "${units[@]}"; do unit_files+=("$DASHBOARD_INSTALL_ROOT/deploy/$name.service"); done
systemd-analyze verify "${unit_files[@]}"
for name in "${units[@]}"; do sudo -n install -m 0644 "$DASHBOARD_INSTALL_ROOT/deploy/$name.service" "/etc/systemd/system/$name.service"; done
sudo -n systemctl daemon-reload
sudo -n systemctl enable vehicle-dashboard.service vehicle-perception-monitor.service vehicle-serial-monitor.service vehicle-sensors.service
# Starting is separate so an active experiment is never interrupted by installation.
printf '%s\n' 'Installed and enabled. Start when hardware is free: sudo systemctl start vehicle-dashboard vehicle-perception-monitor vehicle-serial-monitor vehicle-sensors'
