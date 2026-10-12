# Session replay implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement these steps in this existing dirty workspace. The user's repeated “继续” authorizes implementation; preserve prior changes and do not commit them.

**Goal:** Make recorded driving sessions reusable for offline controller debugging without operating the vehicle.

**Architecture:** Keep the existing recorded-mask controller replay. Default to the saved resolved configuration, build a timestamp-aligned index of the raw sensor journals, and save source hashes and command differences separately from the original data.

**Tech Stack:** Existing Python, unittest, NumPy, OpenCV and YAML; no new dependencies.

**Spec:** User's 2026-10-10 requests: fresh YOLO-driven incremental driving; panel start/end; full session recording; replay primarily for subsequent offline debugging.

## Global Constraints

- No motors, hardware acquisition, service restarts or installation in this coding phase (AGENTS.md).
- Preserve original logs/videos and all existing dirty files.
- Recorded observations cannot predict a different physical trajectory or prove successful driving.
- Unknown, missing, stale and unclosed sensor data must remain explicit.

## Review Focus

- Frozen configuration missing: reject instead of merging today's defaults.
- Sensor journal gaps or writer errors: reject full-data replay, allow explicitly labelled historical partial replay only.
- Future samples: never use them for a previous control cycle.
- Escaping data paths: reject references outside the source archive roots.
- Dashboard PUT/DELETE: must never invoke start/stop.

## Task 1: Frozen configuration and replay bundle

Files: modify `car/autodrive/tools/replay_outer_loop.py`; create `car/autodrive/tools/replay_session.py`, `car/test/test_session_replay.py`.

- [x] Add failing tests for frozen config, timestamp alignment, integrity failure and original preservation.
- [x] Run the CPU tests and confirm the new interface is missing.
- [x] Implement `recorded_replay_config(run, profile)` and `replay_session(run, output, profile='recorded', allow_incomplete_sensors=False)`; no device/model construction.
- [x] Verify real historical live04 through the new tool, explicitly labelling missing historical raw sensors.

## Task 2: Recording end and panel method checks

Files: modify `car/autodrive/runtime/sensor_recording.py`, `car/autodrive/web/dashboard.py`; tests `car/test/test_run_sensors.py`, `car/test/test_vehicle_dashboard.py`.

- [x] Add failing tests for inactive recording publication and forbidden PUT/DELETE with valid control headers.
- [x] Fix inactive ROS recording status and HTTP method handling.
- [x] Run relevant tests and mocked dashboard browser verification.

## Task 3: Evidence and delivery

- [x] Run the appropriate CPU regression modules and syntax checks.
- [x] Save source hashes, commands, replay results and incremental changes under `outputs/visual_feedback/20261010/`.
- [x] Update `PROJECT_INDEX.md`; write `coordination/2026-10-10-visual-feedback-panel-report.md` and offline replay instructions.
- [x] Report software implementation separately from deployment and physical driving validation.

Execution record: tasks complete. Ruling: preserve existing dirty main workspace, no commits or worktree migration; continuing this vehicle workspace avoids losing prior untracked implementation. Fresh reviewer found two important issues; both RED→GREEN, rechecked independently. Full 341-test suite passed before final veto-time anchor change; 32 focused tests passed afterward. Additional full run approval timed out; user requests minimal necessary testing and data-driven physical iteration. No further broad tests. User then explicitly authorized immediate loading for one physical run; two services reloaded and supervisor armed/idle, no Start sent.
