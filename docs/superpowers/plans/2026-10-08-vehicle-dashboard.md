# Vehicle Dashboard Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution. The user selected implementation in this chat after approving the design.

**Goal:** Deliver an automatically started HTTP vehicle dashboard with truthful freshness and sensor/autonomy status.

**Architecture:** Extend existing LCC Web state and HTML with a read-only dashboard adapter. Optional sensor publishers produce bounded atomic snapshots; hardware has one owner and unavailable feeds remain explicit. Deploy an independent systemd HTTP service without motion authorization.

**Tech Stack:** Python 3.8 stdlib HTTP/threading/JSON, existing OpenCV/PySerial and ROS Foxy sensor image, HTML/CSS/JavaScript, systemd.

**Spec:** `docs/superpowers/specs/2026-10-08-vehicle-dashboard-design.md`

## Global Constraints

- Preserve existing dirty work, use isolated worktree, integrate only reviewed increments after hash checks.
- HTTP default read-only; boot never enables motors or gimbal.
- Missing, stale, failed, unsupported and historical readings remain distinguishable.
- Do not invent battery percentage, cloud confidence or global navigation.
- No new cloud calls, no secrets in outputs, no restarting the car during active field work.

## Review Focus

- Old status surviving restart must not become live; test age and producer identity.
- Non-finite or malformed sensor values must not crash JSON or show healthy.
- HTTP control must be denied by default including cross-origin requests.
- Browser/network latency must not multiply hardware polling or block control.
- Serial/camera contention must preserve the current device owner and expose unavailable status.

### Task 1: State aggregation and HTTP contract

Files: new `car/autodrive/web/dashboard.py`, extend `server.py`/`cli.py`, new `car/test/test_vehicle_dashboard.py`.

Interfaces: `DashboardState(output_dir, sensor_dir).enrich(state) -> dict`; schema v1 with `system`, `sensors`, `cloud`, `navigation`, `read_only`. Snapshot publishing uses monotonic time and boot ID, bounded JSON and atomic replacement.

- [x] Add failing tests for stale/old boot/malformed readings, explicit unsupported cloud/navigation, confidential fields and denied controls.
- [x] Implement adapter, health endpoint, allowlisted static assets and default read-only option for service entrypoint.
- [x] Run new tests and old Web regression; expected all pass without hardware.

### Task 2: Browser and sensor publishing

Files: new `dashboard.js`/`dashboard.css`, extend `index.html`, new telemetry publisher/collector module and sensor bridge script.

Interfaces: browser consumes Task 1 schema at 2 Hz without overlapping fetches. Publishers expose battery/IMU/encoder and lidar/depth summaries with sample age, frequency, bounded diagnostics and no raw secrets.

- [x] Cover sensor normalization and failure/busy behavior using injected inputs before implementation.
- [x] Add overview cards, camera health, sensor table/radar view, cloud event and goal panels, disconnected/stale display. Keep existing diagnostics.
- [x] Provide one-owner publishers and explicit launch controls. Connect available sources without modifying the other chat's active control implementation.
- [x] Verify HTTP/DOM/static JavaScript and bounded actual sensor acquisition when devices are free; record unavailable live feeds honestly.

### Task 3: Deployment, review and handoff

Files: new `run_vehicle_dashboard.sh`, `deploy/vehicle-dashboard.service`, deployment helper, documentation and coordination report.

- [x] Prepare motor-disabled, read-only HTTP service with bounded restart and logs; verify service syntax and exact executable paths.
- [x] Review implementation and targeted tests, address functional/safety findings.
- [x] Integrate only files whose original hashes still match; install and enable the new service under existing user authorization.
- [x] Verify HTTP on actual LAN address, enabled/active state and real data source labels; do not claim actual reboot verification without a reboot.
- [x] Record commit/dirty state, commands, logs, hashes, remaining integrations and next owner.

## Execution ledger

- User approved design and requested implementation in this chat. Ruling: preserve this explicit execution authorization and proceed without another generic confirmation; native execution, one final review.
- Baseline: 6 existing Web tests pass; remote fetch yields HEAD/origin equality at 6bf8693; 8080 free; device occupancy check currently empty.
- Ruling: no blanket dependency installation or commits of unrelated existing changes. Existing Jetson Python and ROS sensor image will be reused.

- Complete: state/UI/publishers/deployment and review completed; 200 host tests, real browser and live static acquisition passed. All services enabled/active. No whole-car reboot or motion claim.
- Ruling: concurrent main cloud update fce828b was preserved before integration; only narrow telemetry hooks merged. Docker output permission corrected with scoped DAC_OVERRIDE; no privileged container or chassis device mapping.
