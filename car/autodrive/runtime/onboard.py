#!/usr/bin/env python3
"""Safe onboard outer-loop LCC; dry-run unless motors are explicitly enabled."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

import cv2


AUTODRIVE_DIR = Path(__file__).resolve().parents[1]
CAR_DIR = AUTODRIVE_DIR.parent
CONTROL_DIR = CAR_DIR / "control"
for path in (CAR_DIR, CONTROL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autodrive.runtime.paths import REPO_ROOT
from autodrive.runtime.reporting import (control_log_fields, build_control_row, build_live_status,
    ENCODER_LOG_FIELDS, APPLICATION_LOG_FIELDS)

from autodrive.control.drive_runtime import (
    CommandWatchdog,
    CornerContinuationConfig,
    CornerContinuationGate,
    PerceptionMotionGate,
    SafeWheelDriver,
    WheelMappingConfig,
)
from autodrive.camera.gimbal import initialize_configured_gimbal
from autodrive.camera.transform import CameraTransformConfig, transform_frame
from autodrive.control.lane_centering import (
    DifferentialDriveCommand,
    LCCConfig,
    LaneEstimate,
    LaneCenteringController,
    RoadCenterlineEstimator,
)
from autodrive.perception.outer_loop import (
    BoundaryTrackResult,
    OuterLoopBoundaryConfig,
    OuterLoopBoundaryTracker,
)
from autodrive.perception.perspective import (
    PerspectiveMapper,
    camera_pose_from_mapping,
    validate_calibration_camera_pose,
)
from autodrive.perception.yolopv2_fusion import (
    YOLOPv2FusionConfig,
    YOLOPv2FusionDetector,
)
from autodrive.perception.visualization import render_debug_frame
from vehicle_control.factory import (
    create_chassis,
    validate_motion_configuration,
)
from vehicle_control.profile import load_runtime_config
from autodrive.control.cloud_arbitration import CloudArbitrationConfig
from autodrive.control.cloud_worker import CloudCoordinator
from autodrive.control.cloud_recovery import RecoveryConfig, build_evidence
from autodrive.control.recovery_worker import RecoveryCoordinator
from autodrive.control.field_trial import (
    validate_field_trial_config, field_trial_recording_ready, validate_stationary_corner_config,
)
from autodrive.control.stationary_corner import StationaryCornerController, StationaryCornerConfig
from autodrive.runtime.sensor_recording import raw_sensor_recording_ready

# Compatibility exports; implementations live in focused modules.
from autodrive.runtime.io_utils import _offer_latest
from autodrive.runtime.io_utils import write_status
from autodrive.runtime.sources import VideoSource
from autodrive.runtime.sources import CameraSource
from autodrive.runtime.archive import RunArchive
from autodrive.runtime.archive import AsyncCsvLogger
from autodrive.runtime.publication import LivePublisher
from autodrive.runtime.publication import DebugFrameWriter
from autodrive.runtime.publication import render_birdeye_debug
from autodrive.runtime.corner_runtime import STATIONARY_BOUNDARY_DIAGNOSTICS
from autodrive.runtime.corner_runtime import STATIONARY_LOG_FIELDS, VISUAL_CORNER_DIAGNOSTICS
from autodrive.runtime.corner_runtime import StationaryCornerRuntime
from autodrive.runtime.corner_runtime import apply_runtime_command
from autodrive.runtime.configuration import build_parser
from autodrive.runtime.configuration import load_config
from autodrive.runtime.configuration import repo_path
from autodrive.runtime.configuration import validate_motor_request
from autodrive.runtime.configuration import cloud_configuration
from autodrive.runtime.perception_pipeline import SurfaceOnlyDetector
from autodrive.runtime.perception_pipeline import fuse_tracking_confidence
from autodrive.runtime.perception_pipeline import analyze as _analyze

# Compatibility exports; implementations live in focused modules.
from autodrive.runtime.components import build_components as _build_components
LOCAL_CONFIG = AUTODRIVE_DIR / "config" / "onboard_runtime.yaml"
DEFAULT_CONFIG = LOCAL_CONFIG
MOTOR_CONFIRMATION = "I_UNDERSTAND_MOTORS_WILL_MOVE"


def build_components(config, motors_enabled, calibration_path, field_trial=False):
    """Compatibility facade; hardware dependency is explicit and replaceable."""
    return _build_components(config, motors_enabled, calibration_path, field_trial,
                             chassis_factory=create_chassis,
                             motion_validator=validate_motion_configuration)


def analyze(*args, **kwargs):
    """Compatibility entry with explicit rendering dependency for offline callers."""
    return _analyze(*args, renderer=render_debug_frame, **kwargs)


@contextmanager
def shield_shutdown_signals():
    """Let the first interruption stop the run; finish its resource cleanup."""
    previous = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, signal.SIG_IGN)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    args = build_parser().parse_args()
    config_path, config = load_config(args.config, args.vehicle_profile)
    stationary_settings = validate_stationary_corner_config(config, motors_enabled=args.enable_motors)
    vehicle_config = config.get("vehicle", {})
    vehicle_summary = {
        "profile": vehicle_config.get("profile"),
        "backend": vehicle_config.get("backend"),
        "profile_path": vehicle_config.get("profile_path"),
        "field_trial": args.field_trial,
    }
    print(
        f"Vehicle profile: {vehicle_summary['profile']} "
        f"(backend={vehicle_summary['backend']})",
        flush=True,
    )
    if args.max_samples < 0:
        raise ValueError("--max-samples must not be negative")
    if not math.isfinite(args.max_runtime_seconds) or args.max_runtime_seconds < 0:
        raise ValueError("--max-runtime-seconds must not be negative")
    from autodrive.control.field_trial import TrialMotionWindow
    motion_window = TrialMotionWindow(args.max_motion_seconds)
    motion_window_stopped_at = None
    config.setdefault('runtime', {})['max_motion_seconds'] = args.max_motion_seconds
    calibration_path = repo_path(
        config.get("perspective", {}).get("calibration")
    )
    camera_config = dict(config.get("camera", {}))
    if args.camera_index is not None:
        camera_config["index"] = args.camera_index
    config['camera'] = camera_config
    safety_config = config.get("safety", {})
    maximum_inference_time = float(
        safety_config.get("maximum_inference_time", 0.25)
    )
    watchdog_timeout = float(safety_config.get("watchdog_timeout", 0.40))
    if maximum_inference_time <= 0:
        raise ValueError("maximum_inference_time must be positive")
    if watchdog_timeout <= maximum_inference_time:
        raise ValueError(
            "watchdog_timeout must be greater than maximum_inference_time"
        )
    if args.field_trial:
        if not args.enable_motors or args.no_run_archive:
            raise ValueError("field trial requires explicit motors and run archival")
        validate_field_trial_config(config)
        config.setdefault('runtime', {})['field_trial'] = True
    if args.enable_motors and not args.field_trial:
        validate_motion_configuration(
            vehicle_config,
            config.get("wheels", {}),
        )
    validate_motor_request(
        args.enable_motors,
        args.video,
        args.confirm_motor_motion,
        calibration_path,
        camera_config,
        image_space_trial=args.field_trial,
        max_runtime_seconds=args.max_runtime_seconds,
        panel_controlled=bool(args.panel_session_dir),
    )
    if calibration_path is not None and not args.enable_motors:
        try:
            validate_calibration_camera_pose(calibration_path, camera_config)
        except (OSError, ValueError) as exc:
            print(
                f"WARNING: dry-run calibration is not valid for this camera pose: {exc}",
                flush=True,
            )
    sample_every = (
        args.sample_every
        if args.sample_every is not None
        else int(config.get("runtime", {}).get("video_sample_every", 6))
    )
    if sample_every < 1:
        raise ValueError("sample interval must be at least 1")

    runtime_config = config.get("runtime", {})
    output_dir = repo_path(
        args.output_dir
        or runtime_config.get("output_dir", "outputs/onboard_runtime")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "onboard_log.csv"
    status_path = output_dir / "status.json"
    latest_frame_path = output_dir / "latest.jpg"
    latest_birdeye_path = output_dir / "latest_birdeye.jpg"
    source = (
        VideoSource(args.video, sample_every)
        if args.video
        else None
    )
    if source is None:
        gimbal_commands = initialize_configured_gimbal(
            camera_config,
            vehicle_config=vehicle_config,
        )
        if gimbal_commands:
            print(
                f"Initialized camera gimbal before capture: {gimbal_commands}",
                flush=True,
            )
        source = CameraSource(camera_config)
    archive_fps = (
        source.fps / source.sample_every
        if isinstance(source, VideoSource)
        else float(camera_config.get("fps", 20))
    )
    archive = None
    try:
        archive = RunArchive(
            output_dir,
            enabled=(
                bool(runtime_config.get("archive_runs", True))
                and not args.no_run_archive
            ),
            fps=float(runtime_config.get("archive_fps", archive_fps)),
            mode="hardware" if args.enable_motors else "dryrun",
            queue_frames=int(runtime_config.get("archive_queue_frames", 32)),
        )
        archive.snapshot_file(config_path, "runtime_config.yaml")
        archive.snapshot_mapping(config, "resolved_runtime_config.yaml")
        if config.get('perception',{}).get('track_colors',{}).get('enabled'):
            from autodrive.perception.track_colors import PARAMETERS,SURFACE_PARAMETERS,ROTATION_PARAMETERS
            archive.snapshot_mapping(PARAMETERS, 'track_color_parameters.yaml')
            if config['perception']['track_colors'].get('surface_assist') is True:
                archive.snapshot_mapping(SURFACE_PARAMETERS,'track_surface_parameters.yaml')
                archive.snapshot_mapping(ROTATION_PARAMETERS,'track_rotation_parameters.yaml')
        archive.snapshot_file(
            vehicle_config.get("profile_path"),
            "vehicle_profile.yaml",
        )
        archive.snapshot_file(calibration_path, "perspective_calibration.yaml")
        if archive.enabled:
            from autodrive.runtime.provenance import write_run_provenance
            model_settings = config.get('perception', {}).get('yolopv2', {})
            model_paths = ({'int8': model_settings.get('int8_weights')}
                           if model_settings.get('adaptive_precision') else {})
            write_run_provenance(archive.run_dir, REPO_ROOT,
                model_settings.get('weights') if model_settings.get('enabled') else None,
                model_paths=model_paths if model_settings.get('enabled') else {})
        debug_output_dir = None
        if args.save_debug_frames:
            debug_output_dir = (
                archive.run_dir / "frames"
                if archive.enabled
                else output_dir / f"debug_{int(time.time())}"
            )
            debug_output_dir.mkdir(parents=True, exist_ok=False)
    except BaseException as exc:
        initialization_cleanup_errors = []
        actions = [('source.close', source.close)]
        if archive is not None:
            actions.append(('archive.close', archive.close))
        for name, action in actions:
            try:
                action()
            except Exception as cleanup_exc:
                initialization_cleanup_errors.append(dict(action=name,
                    type=type(cleanup_exc).__name__, message=str(cleanup_exc)))
        if initialization_cleanup_errors:
            exc.initialization_cleanup_errors = initialization_cleanup_errors
        raise
    detector = estimator = controller = mapper = boundary_tracker = driver = watchdog = None
    motion_gate = None
    corner_gate = None
    stationary_runtime = None
    log_writer = None
    live_publisher = None
    debug_writer = None
    cloud_coordinator = None
    telemetry_publisher = None
    motion_guard = None
    sensor_writer = None
    sensor_recording_state = {}
    sensor_request_path = None
    sensor_request_next = 0.
    last_row = None
    previous_timestamp = None
    termination_reason = "runtime shutdown"
    try:
        cloud_settings = cloud_configuration(config)
        recovery_enabled = isinstance(cloud_settings, RecoveryConfig) and cloud_settings.enabled
        cloud_client = None
        if cloud_settings.enabled:
            if not archive.enabled:
                raise ValueError("cloud arbitration requires run evidence archival")
            from cloud_client.client import CloudClient
            cloud_client = CloudClient(timeout=cloud_settings.request_timeout_seconds,
                **({'contract':'road-recovery-v1','provider':'qwen-realtime',
                    'model':'qwen3.8-omni-flash-realtime'} if recovery_enabled else {}))
            if not recovery_enabled and getattr(cloud_client.config, "contract", "road-scene-v1") != "road-scene-v1":
                raise ValueError("vehicle arbitration requires CAR_CLOUD_CONTRACT=road-scene-v1")
            cloud_client._require_key()
        cloud_coordinator = (RecoveryCoordinator if recovery_enabled else CloudCoordinator)(
            cloud_settings, cloud_client, (archive.run_dir or output_dir) / "cloud"
        )
        (
            detector,
            estimator,
            controller,
            mapper,
            boundary_tracker,
            driver,
            watchdog,
        ) = build_components(config, args.enable_motors, calibration_path,
                             **({'field_trial': True} if args.field_trial else {}))
        driver.stop("initialized; waiting for fresh perception")
        if stationary_settings.visual_feedback or args.panel_session_dir:
            from autodrive.control.motion_guard import MotionGuard
            motion_guard = MotionGuard(driver,args.panel_session_dir)
        shared_session = getattr(driver.chassis, "controller", None)
        if archive.enabled and not args.video:
            from autodrive.runtime.sensor_recording import RunSensorWriter, atomic_json
            sensor_writer=RunSensorWriter(archive.run_dir/'sensors')
            sensor_request_path=Path(runtime_config.get('telemetry_dir',str(REPO_ROOT/'outputs/vehicle_dashboard/sensors')))/'recording_request.json'
            if shared_session is not None and hasattr(shared_session,'recording_sink'):
                shared_session.recording_sink=sensor_writer.submit
        if shared_session is not None and hasattr(shared_session, "poll_available"):
            from autodrive.web.sensor_publishers import SharedSerialPublisher
            telemetry_publisher = SharedSerialPublisher(
                shared_session, runtime_config.get("telemetry_dir", str(REPO_ROOT / "outputs/vehicle_dashboard/sensors"))
            )
        frame, timestamp, frame_age = source.read()
        if frame is None:
            raise RuntimeError("no fresh startup frame")
        if not args.video:
            expected_pose = camera_pose_from_mapping(camera_config)
            actual_size = (int(frame.shape[1]), int(frame.shape[0]))
            expected_size = (
                expected_pose["image_width"],
                expected_pose["image_height"],
            )
            if actual_size != expected_size:
                raise RuntimeError(
                    "camera returned a frame size that does not match the "
                    f"calibrated runtime geometry: actual={actual_size[0]}x"
                    f"{actual_size[1]}, expected={expected_size[0]}x"
                    f"{expected_size[1]}"
                )

        print(
            "Warming up boundary perception; "
            "wheels remain stopped ...",
            flush=True,
        )
        analyze(
            detector,
            estimator,
            controller,
            mapper,
            frame,
            str(config.get("runtime", {}).get("route_hint", "center")),
            None,
            maximum_inference_time,
            boundary_tracker,
            captured_at=time.monotonic() - (frame_age or 0.0),
        )
        controller.reset()
        if boundary_tracker is not None:
            boundary_tracker.reset()
        driver.stop("warmup complete; waiting for fresh frame")
        watchdog.arm()

        fieldnames = control_log_fields()
        log_writer = AsyncCsvLogger(
            log_path,
            fieldnames,
            queue_rows=int(runtime_config.get("log_queue_rows", 256)),
        )
        debug_writer = DebugFrameWriter(
            debug_output_dir,
            queue_frames=int(runtime_config.get("debug_queue_frames", 8)),
        )

        sample = 0
        motion_gate = PerceptionMotionGate(
            resume_valid_frames=int(
                safety_config.get("resume_valid_frames", 4)
            ),
            resume_min_confidence=float(
                safety_config.get("resume_min_confidence", 0.0)
            ),
            maximum_lateral_jump=float(
                safety_config.get("resume_maximum_lateral_jump", 0.0)
            ),
            maximum_heading_jump=float(
                safety_config.get("resume_maximum_heading_jump", 0.0)
            ),
            require_consistent_source=bool(
                safety_config.get("resume_require_consistent_source", False)
            ),
        )
        corner_gate = CornerContinuationGate(
            CornerContinuationConfig(
                **safety_config.get("corner_continuation", {})
            )
        )
        if stationary_settings.enabled:
            # Physical runs were evidence-validated before any model/device was
            # opened; dry-run/replay uses the same proposals without actuation.
            stationary_runtime = StationaryCornerRuntime(
                stationary_settings, motion_gate, pivot_verified=True)
        route_hint = str(config.get("runtime", {}).get("route_hint", "center"))
        max_inference = maximum_inference_time
        runtime_started = time.monotonic()
        live_update_hz = float(runtime_config.get("live_update_hz", 5.0))
        console_update_hz = float(runtime_config.get("console_update_hz", 4.0))
        if live_update_hz <= 0.0 or console_update_hz <= 0.0:
            raise ValueError("runtime update frequencies must be positive")
        live_update_interval = 1.0 / live_update_hz
        console_update_interval = 1.0 / console_update_hz
        next_live_update_at = 0.0
        next_console_update_at = 0.0
        save_latest = bool(
            config.get("runtime", {}).get("save_latest_frame", True)
        )
        live_publisher = LivePublisher(
            status_path,
            archive.status_path if archive.enabled else None,
            latest_frame_path,
            latest_birdeye_path,
            save_latest,
        )
        previous_loop_started = None
        print(
            f"Onboard runtime started: mode={'HARDWARE' if args.enable_motors else 'DRY-RUN'}, "
            f"source={args.video or 'camera'}, calibration={calibration_path or 'none'}",
            flush=True,
        )
        if archive.enabled:
            print(f"Run archive: {archive.run_dir}", flush=True)

        while True:
            loop_started = time.monotonic()
            if motion_guard is not None and motion_guard.cancelled.is_set():
                termination_reason=motion_guard.reason
                break
            if sensor_request_path is not None and loop_started>=sensor_request_next:
                from autodrive.web.dashboard import boot_id
                atomic_json(sensor_request_path,dict(session_id=archive.run_dir.name,active=True,
                    boot_id=boot_id(),expires_monotonic=loop_started+2.))
                sensor_request_next=loop_started+.25
                try:
                    sensor_recording_state=json.loads(sensor_request_path.with_name('recording_state.json').read_text())
                except (OSError,ValueError):sensor_recording_state={}
            if motion_window.expired(loop_started):
                termination_reason = f"maximum motion window reached ({args.max_motion_seconds:.3f}s)"
                driver.stop(termination_reason)
                motion_window_stopped_at = time.monotonic()
                break
            loop_start_interval = (
                None
                if previous_loop_started is None
                else loop_started - previous_loop_started
            )
            previous_loop_started = loop_started
            if (
                args.max_runtime_seconds
                and time.monotonic() - runtime_started
                >= args.max_runtime_seconds
            ):
                termination_reason = (
                    "maximum runtime reached "
                    f"({args.max_runtime_seconds:.3f}s)"
                )
                print("Maximum runtime reached; stopping.", flush=True)
                break
            source_wait_started = time.monotonic()
            frame, timestamp, frame_age = source.read()
            source_wait_time = time.monotonic() - source_wait_started
            if frame is None:
                termination_reason = "camera/video source ended"
                break
            dt = (
                None
                if previous_timestamp is None
                else max(1e-3, float(timestamp - previous_timestamp))
            )
            previous_timestamp = timestamp
            analysis_started = time.monotonic()
            render_jobs = []
            (
                estimate,
                command,
                annotated,
                inference_time,
                boundary_time,
                boundary_result,
            ) = analyze(
                detector,
                estimator,
                controller,
                mapper,
                frame,
                route_hint,
                dt,
                max_inference,
                boundary_tracker,
                captured_at=time.monotonic() - (frame_age or 0.0),
                deferred_render=(render_jobs.append if stationary_settings.visual_feedback else None),
            )
            analysis_time = time.monotonic() - analysis_started
            gate_started = time.monotonic()
            displayed_command = command
            telemetry_snapshot = (shared_session.telemetry()
                if shared_session is not None and hasattr(shared_session, 'telemetry') else {})
            telemetry_snapshot = telemetry_snapshot if isinstance(telemetry_snapshot, dict) else {}
            if sensor_writer is not None:
                sensor_writer.submit('telemetry',dict(received_monotonic=time.monotonic(),samples=telemetry_snapshot))
            imu_sample = telemetry_snapshot.get('attitude', {})
            encoder_sample = telemetry_snapshot.get('encoder', {})
            encoder_sample = encoder_sample if isinstance(encoder_sample, dict) else {}
            encoder_value = encoder_sample.get('value')
            encoder_ticks = (encoder_value.get('native_ticks')
                             if isinstance(encoder_value, dict) else None)
            if not isinstance(encoder_ticks, (list, tuple)) or len(encoder_ticks) != 4:
                encoder_ticks = (None, None, None, None)
            if stationary_runtime is not None:
                control_now = time.monotonic()
                recording_ready = (not args.field_trial or
                    field_trial_recording_ready(archive.get_state(), log_writer.get_state()))
                if sensor_writer is not None:
                    sensor_state=sensor_writer.get_state()
                    recording_ready=recording_ready and sensor_state['error'] is None and sensor_state['dropped']==0
                if args.enable_motors and runtime_config.get('require_sensor_recording',False):
                    recording_ready=bool(recording_ready and raw_sensor_recording_ready(
                        None if sensor_writer is None else sensor_writer.get_state(),
                        sensor_recording_state,archive.run_dir.name,control_now,
                        None if telemetry_publisher is None else telemetry_publisher.get_state()))
                effective_frame_age = (frame_age + control_now-analysis_started
                    if StationaryCornerRuntime._finite(frame_age) else None)
                command = stationary_runtime.filter(command, estimate, boundary_result,
                    now=control_now, imu_sample=imu_sample, frame_age_seconds=effective_frame_age,
                    perception_budget_valid=(inference_time+boundary_time <= max_inference),
                    external_motion_allowed=bool(recording_ready and not watchdog.get_state()['tripped']))
                if recovery_enabled:
                    evidence=build_evidence(detector.get_consumed_observation(),boundary_result,command,
                        now=control_now,frame_age=effective_frame_age,
                        external_allowed=bool(recording_ready and not watchdog.get_state()['tripped']),
                        perception_valid=(inference_time+boundary_time<=max_inference),
                        pivot_verified=stationary_runtime.pivot_verified,imu_sample=imu_sample,
                        local_state=stationary_runtime.get_state())
                    command=cloud_coordinator.filter(command,evidence,control_now,
                        local_deadline=stationary_runtime.controller.get_state()['motion_deadline'])
                elif cloud_settings.enabled:
                    command = cloud_coordinator.filter(command, detector.get_consumed_observation(),
                        local_safe=stationary_runtime.get_state()['local_safe'], reason=command.reason,
                        now=control_now)
                corner_state = corner_gate.get_state(now=timestamp)
            elif cloud_settings.enabled:
                command = cloud_coordinator.filter(
                    command, detector.get_consumed_observation(),
                    local_safe=bool(estimate.valid and boundary_result.valid and command.action != "stop"),
                    reason=command.reason,
                )
            if stationary_runtime is None:
                command = corner_gate.filter(
                    command, estimate, boundary_result,
                    # The capture timeline keeps bounded legacy continuation
                    # identical between live control and recorded replay.
                    now=timestamp,
                )
                corner_state = corner_gate.get_state(now=timestamp)
                command = motion_gate.filter(
                    command, estimate, boundary_result,
                    allow_discontinuity=bool(corner_state["active"]),
                )
            if hasattr(detector, "update_route_state"):
                detector.update_route_state(boundary_result, command)
            gate_time = time.monotonic() - gate_started
            hardware_started = time.monotonic()
            if motion_window.expired(hardware_started):
                termination_reason = f"maximum motion window reached ({args.max_motion_seconds:.3f}s)"
                driver.stop(termination_reason)
                motion_window_stopped_at = time.monotonic()
                break
            if args.field_trial and not field_trial_recording_ready(archive.get_state(), log_writer.get_state()):
                command = DifferentialDriveCommand('stop', 0., 0., 0., 0.,
                                                   'waiting for first recorded video frame')
            watchdog.heartbeat()
            application_now = time.monotonic()
            semantic_captured_at = getattr(boundary_result, 'semantic_captured_at', None)
            semantic_application_age = (application_now-semantic_captured_at
                if StationaryCornerRuntime._finite(semantic_captured_at) else None)
            frame_application_age = (frame_age+application_now-analysis_started
                if StationaryCornerRuntime._finite(frame_age) else None)
            imu_receipt = imu_sample.get('received_monotonic') if isinstance(imu_sample, dict) else None
            imu_application_age = (application_now-imu_receipt
                if StationaryCornerRuntime._finite(imu_receipt) else None)
            application_reason = ''
            if stationary_runtime is not None:
                active_pivot = command.action == 'pivot-right'
                semantic_current = (StationaryCornerRuntime._finite(semantic_application_age)
                    and 0 <= semantic_application_age <= stationary_settings.semantic_max_age_seconds)
                camera_current = (StationaryCornerRuntime._finite(frame_application_age)
                    and 0 <= frame_application_age <= .25)
                imu_current = (not active_pivot or
                    StationaryCornerRuntime._finite(imu_application_age)
                    and 0 <= imu_application_age <= stationary_settings.imu_max_age_seconds)
                if not semantic_current or not camera_current or not imu_current:
                    application_reason = ('stationary corner: IMU expired before application'
                        if not imu_current else 'stationary corner: consumed semantic or camera frame expired before application')
                    command = stationary_runtime.apply_veto(application_reason,now=application_now)
            command_before_driver=command
            command,wheel_state=apply_runtime_command(driver,command,stationary_runtime,
                recovery=cloud_coordinator if recovery_enabled else None)
            if recovery_enabled and not cloud_coordinator.record_application(command,wheel_state,time.monotonic()):
                command=cloud_coordinator.veto('recovery application archive failed',time.monotonic())
                stationary_runtime.apply_veto(command.reason,now=time.monotonic())
                wheel_state=driver.stop(command.reason)
            if command!=command_before_driver:
                application_reason=command.reason
                application_now=(stationary_runtime.controller.get_state()['stopped_at']
                    if stationary_settings.visual_feedback else time.monotonic())
            motion_window.record_output(
                tuple(wheel_state[k] for k in ('front_left_pwm', 'rear_left_pwm',
                                              'front_right_pwm', 'rear_right_pwm')),
                hardware_started,
            )
            hardware_apply_time = time.monotonic() - hardware_started
            # Render the same consumed frame only after the command and its
            # independent deadline are applied. Display work is not permission
            # to extend motion or reuse a stale observation.
            control_render_started = time.monotonic()
            if render_jobs:
                annotated = render_jobs[0]()
            control_render_time = time.monotonic() - control_render_started
            if command != displayed_command:
                cv2.rectangle(annotated, (0, annotated.shape[0] - 32),
                    (annotated.shape[1], annotated.shape[0]), (0, 0, 120), -1)
                cv2.putText(annotated,
                    f"APPLIED: {command.action} steer={command.steering:+.3f} - {command.reason}",
                    (8, annotated.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    .46, (255, 255, 255), 1, cv2.LINE_AA)
            if hasattr(detector, "semantic_corridor"):
                mode_label = "HARDWARE" if args.enable_motors else "DRY RUN (motors disabled)"
                pwm = tuple(wheel_state[key] for key in (
                    "front_left_pwm", "rear_left_pwm", "front_right_pwm", "rear_right_pwm"))
                cv2.rectangle(annotated, (0, 112), (annotated.shape[1], 139), (40, 40, 40), -1)
                cv2.putText(annotated, f"{mode_label} | {command.action} | PWM {pwm}",
                            (8, 131), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
            # Starting motion can include a bounded ramp.  Refresh after the
            # hardware call so that intentional transition time does not eat
            # into the much shorter stale-command watchdog window.
            watchdog.heartbeat()
            birdeye_started = time.monotonic()
            birdeye = (
                None
                if boundary_result is None
                else render_birdeye_debug(boundary_result, estimate)
            )
            birdeye_render_time = time.monotonic() - birdeye_started
            diagnostic_now = time.monotonic()
            live_update_due = diagnostic_now >= next_live_update_at
            archive_state = archive.get_state()
            log_state = log_writer.get_state()
            live_state = live_publisher.get_state()
            debug_state = debug_writer.get_state()
            motion_state = motion_gate.get_state()
            stationary_state = {} if stationary_runtime is None else stationary_runtime.get_state()
            row = build_control_row(
                stationary_state=stationary_state,
                analysis_time=analysis_time,
                application_now=application_now,
                application_reason=application_reason,
                archive_state=archive_state,
                birdeye_render_time=birdeye_render_time,
                boundary_result=boundary_result,
                boundary_time=boundary_time,
                cloud_coordinator=cloud_coordinator,
                command=command,
                control_render_time=control_render_time,
                corner_state=corner_state,
                debug_state=debug_state,
                detector=detector,
                dt=dt,
                encoder_sample=encoder_sample,
                encoder_ticks=encoder_ticks,
                estimate=estimate,
                frame_age=frame_age,
                frame_application_age=frame_application_age,
                gate_time=gate_time,
                hardware_apply_time=hardware_apply_time,
                imu_application_age=imu_application_age,
                inference_time=inference_time,
                live_state=live_state,
                log_state=log_state,
                loop_start_interval=loop_start_interval,
                motion_state=motion_state,
                sample=sample,
                semantic_application_age=semantic_application_age,
                source_wait_time=source_wait_time,
                timestamp=timestamp,
                vehicle_summary=vehicle_summary,
                wheel_state=wheel_state,
            )
            last_row = row
            # Every operation below is a non-blocking memory enqueue.  Slow
            # disks/JPEG/MP4 encoding can drop diagnostics, but can no longer
            # leave the previous wheel command applied through a corner.
            log_writer.submit(row)
            archive.write_frames(frame, annotated, birdeye, row,
                semantic_observation=(detector.get_consumed_observation()
                    if hasattr(detector, 'get_consumed_observation') else None))
            debug_writer.submit(sample, frame, annotated, birdeye)
            if live_update_due:
                live_status = build_live_status(
                    archive=archive,
                    args=args,
                    calibration_path=calibration_path,
                    camera_config=camera_config,
                    cloud_coordinator=cloud_coordinator,
                    config_path=config_path,
                    corner_state=corner_state,
                    debug_writer=debug_writer,
                    detector=detector,
                    driver=driver,
                    live_publisher=live_publisher,
                    log_writer=log_writer,
                    motion_state=motion_state,
                    row=row,
                    runtime_config=runtime_config,
                    sensor_recording_state=sensor_recording_state,
                    sensor_writer=sensor_writer,
                    stationary_runtime=stationary_runtime,
                    stationary_state=stationary_state,
                    telemetry_publisher=telemetry_publisher,
                    vehicle_summary=vehicle_summary,
                    watchdog=watchdog,
                    wheel_state=wheel_state,
                )
                live_publisher.publish(live_status, annotated, birdeye)
                next_live_update_at = diagnostic_now + live_update_interval
            if diagnostic_now >= next_console_update_at:
                print(
                    f"sample={sample} action={command.action} "
                    f"boundary={boundary_result.source if boundary_result else 'off'} "
                    f"conf={estimate.confidence:.2f} steer={command.steering:+.3f} "
                    f"tight={float(command.tight_turn_factor or 0.0):.2f} "
                    f"pwm=({wheel_state['front_left_pwm']:+d},"
                    f"{wheel_state['rear_left_pwm']:+d},"
                    f"{wheel_state['front_right_pwm']:+d},"
                    f"{wheel_state['rear_right_pwm']:+d}) "
                    f"inference={inference_time * 1000:.0f}ms "
                    "yolopv2="
                    f"{boundary_result.semantic_fusion_source if boundary_result else 'off'}:"
                    f"{boundary_result.semantic_precision if boundary_result else 'none'} "
                    f"boundary_ms={boundary_time * 1000:.1f} "
                    f"hardware_ms={hardware_apply_time * 1000:.1f} "
                    f"loop_gap_ms={float(loop_start_interval or 0.0) * 1000:.1f}",
                    flush=True,
                )
                next_console_update_at = (
                    diagnostic_now + console_update_interval
                )
            sample += 1
            if args.max_samples and sample >= args.max_samples:
                termination_reason = f"maximum samples reached ({args.max_samples})"
                break
    except KeyboardInterrupt:
        termination_reason = "interrupted by operator"
        print("Interrupted; stopping.", flush=True)
    except Exception as exc:
        termination_reason = f"runtime error: {type(exc).__name__}: {exc}"
        raise
    finally:
        with shield_shutdown_signals():
            active_exception = sys.exc_info()[1]
            cleanup_errors = []
            def cleanup_action(name, action):
                try:
                    action()
                    return True
                except (Exception, KeyboardInterrupt) as exc:
                    cleanup_errors.append((name, exc))
                    return False

            # Send zero independently before any worker join can block or fail.
            driver_stopped = (driver is not None and
                              cleanup_action('driver.stop', lambda: driver.stop(termination_reason)))
            if motion_guard is not None:
                cleanup_action('motion_guard.close',motion_guard.close)
            if sensor_request_path is not None:
                cleanup_action('sensor_request.end',lambda: atomic_json(sensor_request_path,
                    dict(session_id=archive.run_dir.name,active=False)))
            if driver_stopped:
                if motion_window_stopped_at is None:
                    motion_window_stopped_at = time.monotonic()
            watchdog_stopped = (watchdog is None or
                                cleanup_action('watchdog.close', watchdog.close))
            if cloud_coordinator is not None:
                cleanup_action('cloud_coordinator.close', cloud_coordinator.close)
            if detector is not None and hasattr(detector, "close"):
                cleanup_action('detector.close', detector.close)
            cleanup_action('source.close', source.close)
            # Drain diagnostic workers only after the wheels are stopped.  Closing
            # may legitimately wait for buffered I/O, but it is no longer on the
            # vehicle-control critical path.  Stop the live publisher first so an
            # older queued update cannot overwrite the final stopped status.
            if telemetry_publisher is not None:
                cleanup_action('telemetry_publisher.close', telemetry_publisher.close)
            if sensor_writer is not None:
                if shared_session is not None and hasattr(shared_session,'recording_sink'):
                    shared_session.recording_sink=None
                cleanup_action('sensor_writer.close',sensor_writer.close)
            if live_publisher is not None:
                cleanup_action('live_publisher.close', live_publisher.close)
            if debug_writer is not None:
                cleanup_action('debug_writer.close', debug_writer.close)
            if log_writer is not None:
                cleanup_action('log_writer.close', log_writer.close)
            cleanup_action('archive.close', archive.close)
            if driver is not None:
                close_chassis = getattr(driver.chassis, "close", None)
                if close_chassis is not None:
                    cleanup_action('chassis.close', lambda: close_chassis(stop=False))
            if driver is not None:
                final_wheel_state = driver.get_state()
                final_status = {
                    "vehicle": vehicle_summary,
                    "mode": final_wheel_state["mode"],
                    "source": str(
                        args.video or f"camera:{camera_config.get('index', 0)}"
                    ),
                    "config": str(config_path),
                    "calibration": (
                        str(calibration_path) if calibration_path else None
                    ),
                    "termination": {
                        "reason": termination_reason,
                        "stopped": bool(driver_stopped and watchdog_stopped),
                        "cleanup_errors": [dict(action=name, type=type(exc).__name__, message=str(exc))
                                           for name, exc in cleanup_errors],
                        "completed_at": time.time(),
                    },
                    "motion_window": {
                        "duration_seconds": motion_window.duration_seconds,
                        "started_at_monotonic": motion_window.started_at,
                        "deadline_monotonic": (
                            None if not motion_window.duration_seconds or motion_window.started_at is None
                            else motion_window.started_at + motion_window.duration_seconds
                        ),
                        "stop_completed_at_monotonic": motion_window_stopped_at,
                        "limitation": "command timing; loop-checked with existing watchdog, not measured displacement",
                    },
                    "last_result": last_row,
                    "wheel_driver": final_wheel_state,
                    "watchdog": (
                        None if watchdog is None else watchdog.get_state()
                    ),
                    "corner_continuation": (
                        None
                        if corner_gate is None
                        else corner_gate.get_state(now=previous_timestamp)
                    ),
                    "motion_gate": (
                        None if motion_gate is None else motion_gate.get_state()
                    ),
                    "cloud_arbitration": (
                        None if cloud_coordinator is None else cloud_coordinator.get_state()
                    ),
                    "stationary_corner": (
                        None if stationary_runtime is None else stationary_runtime.get_state()
                    ),
                    'sensor_recording':dict(local=None if sensor_writer is None else sensor_writer.get_state(),
                        ros=sensor_recording_state,
                        serial_publisher=None if telemetry_publisher is None else telemetry_publisher.get_state(),
                        ros_directory=(None if sensor_request_path is None else
                            str(sensor_request_path.parent/'recordings'/archive.run_dir.name))),
                    "run_archive": (
                        None if not archive.enabled else str(archive.run_dir)
                    ),
                    "diagnostics": {
                        "archive": archive.get_state(),
                        "control_log": (
                            None if log_writer is None else log_writer.get_state()
                        ),
                        "live_publisher": (
                            None
                            if live_publisher is None
                            else live_publisher.get_state()
                        ),
                        "debug_frames": (
                            None if debug_writer is None else debug_writer.get_state()
                        ),
                        "yolopv2_fusion": (
                            detector.get_state()
                            if detector is not None and hasattr(detector, "get_state")
                            else {"enabled": False}
                        ),
                    },
                }
                cleanup_action('write_status', lambda: write_status(status_path, final_status))
                cleanup_action('archive.write_status', lambda: archive.write_status(final_status))
            if cleanup_errors and active_exception is None:
                raise cleanup_errors[0][1]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
