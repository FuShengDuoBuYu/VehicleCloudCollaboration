#!/usr/bin/env python3
"""Safe onboard outer-loop LCC; dry-run unless motors are explicitly enabled."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import yaml


AUTODRIVE_DIR = Path(__file__).resolve().parent
CAR_DIR = AUTODRIVE_DIR.parent
CONTROL_DIR = CAR_DIR / "control"
REPO_ROOT = CAR_DIR.parent
for path in (CAR_DIR, CONTROL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autodrive.drive_runtime import (
    CommandWatchdog,
    CornerContinuationConfig,
    CornerContinuationGate,
    PerceptionMotionGate,
    SafeWheelDriver,
    WheelMappingConfig,
)
from autodrive.camera_gimbal import initialize_configured_gimbal
from autodrive.camera_transform import CameraTransformConfig, transform_frame
from autodrive.lane_centering import (
    DifferentialDriveCommand,
    LCCConfig,
    LaneEstimate,
    LaneCenteringController,
    RoadCenterlineEstimator,
)
from autodrive.outer_loop import (
    BoundaryTrackResult,
    OuterLoopBoundaryConfig,
    OuterLoopBoundaryTracker,
)
from autodrive.perspective import (
    PerspectiveMapper,
    camera_pose_from_mapping,
    validate_calibration_camera_pose,
)
from autodrive.visualization import render_debug_frame
LOCAL_CONFIG = AUTODRIVE_DIR / "onboard_runtime.yaml"
DEFAULT_CONFIG = LOCAL_CONFIG
MOTOR_CONFIRMATION = "I_UNDERSTAND_MOTORS_WILL_MOVE"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run outer-loop lane centering (dry-run by default)"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="Use a recorded onboard video instead of a camera")
    source.add_argument("--camera-index", type=int, help="Override configured camera index")
    parser.add_argument("--sample-every", type=int, help="Video frame sampling interval")
    parser.add_argument(
        "--output-dir",
        help="Override output directory (useful for isolated offline replay)",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 runs until EOF/interrupt")
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="0 runs until EOF/interrupt; otherwise stop after this wall-clock duration",
    )
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        help="Save every raw, annotated, and bird's-eye frame for diagnosis",
    )
    parser.add_argument(
        "--no-run-archive",
        action="store_true",
        help="Disable per-run videos and snapshots (intended for local replay only)",
    )
    parser.add_argument(
        "--enable-motors",
        action="store_true",
        help="Enable physical I2C motor output; camera source and calibration are required",
    )
    parser.add_argument(
        "--confirm-motor-motion",
        default="",
        help=f"Required with --enable-motors: {MOTOR_CONFIRMATION}",
    )
    return parser


def load_config(path):
    config_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if data.get("version") != 1:
        raise ValueError("onboard runtime config version must be 1")
    return config_path, data


def repo_path(value):
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_motor_request(
    enable_motors,
    video,
    confirmation,
    calibration_path,
    camera_config=None,
):
    if not enable_motors:
        return
    if video:
        raise ValueError("physical motors cannot be enabled with --video")
    if confirmation != MOTOR_CONFIRMATION:
        raise ValueError(
            "--enable-motors requires the exact --confirm-motor-motion value"
        )
    if calibration_path is None:
        raise ValueError("physical motors require a perspective calibration")
    calibration_path = Path(calibration_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"perspective calibration does not exist: {calibration_path}"
        )
    if camera_config is not None:
        validate_calibration_camera_pose(calibration_path, camera_config)


class VideoSource:
    def __init__(self, path, sample_every):
        self.path = Path(path).expanduser().resolve()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open video: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.sample_every = max(1, int(sample_every))
        self.frame_index = -1
        self.timestamps = self._load_archive_timestamps()

    def _load_archive_timestamps(self):
        """Use the per-frame capture clock stored beside archived videos.

        Hardware capture may run below the MP4 writer's nominal frame rate.
        The CSV has one row per archived frame and preserves the real capture
        timestamps, which are required to reproduce time-bounded control gates.
        """
        sidecar = self.path.with_name("onboard_log.csv")
        if not sidecar.is_file():
            return None
        try:
            with sidecar.open("r", encoding="utf-8", newline="") as handle:
                timestamps = [
                    float(row["timestamp_s"])
                    for row in csv.DictReader(handle)
                ]
        except (KeyError, TypeError, ValueError, OSError):
            return None
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0 or len(timestamps) != frame_count:
            return None
        return timestamps

    def read(self):
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return None, None, None
            self.frame_index += 1
            if self.frame_index % self.sample_every == 0:
                timestamp = (
                    self.timestamps[self.frame_index]
                    if self.timestamps is not None
                    else self.frame_index / self.fps
                )
                return frame, timestamp, 0.0

    def close(self):
        self.capture.release()


class CameraSource:
    def __init__(self, camera_config):
        from dataclasses import replace as dc_replace
        from vehicle_control.camera import CameraStream
        from vehicle_control.settings import CAMERA_CONFIG

        self.startup_timeout = float(camera_config.get("startup_timeout", 10.0))
        self.stale_timeout = float(camera_config.get("stale_timeout", 1.0))
        self.frame_transform = CameraTransformConfig.from_mapping(camera_config)
        config = dc_replace(
            CAMERA_CONFIG,
            camera_index=int(camera_config.get("index", 0)),
            width=int(camera_config.get("width", 640)),
            height=int(camera_config.get("height", 480)),
            fps=int(camera_config.get("fps", 20)),
        )
        self.camera = CameraStream(config)
        self.camera.start()
        self.started_at = time.monotonic()
        self.last_sequence = 0

    def read(self):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            frame, captured_at, sequence = self.camera.get_frame_packet()
            age = (
                None
                if captured_at is None
                else time.monotonic() - captured_at
            )
            if (
                frame is not None
                and age is not None
                and age <= self.stale_timeout
                and sequence != self.last_sequence
            ):
                self.last_sequence = sequence
                return (
                    transform_frame(frame, self.frame_transform),
                    captured_at - self.started_at,
                    age,
                )
            time.sleep(0.05)
        return None, None, None

    def close(self):
        self.camera.stop()


class RunArchive:
    """Persist one self-contained experiment for deterministic diagnosis."""

    def __init__(self, output_dir, enabled, fps, mode):
        self.enabled = bool(enabled)
        self.fps = max(1.0, float(fps))
        self.run_dir = None
        self.status_path = None
        self.log_path = None
        self._writers = {}
        if not self.enabled:
            return
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{mode}"
        self.run_dir = Path(output_dir) / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.status_path = self.run_dir / "status.json"
        self.log_path = self.run_dir / "onboard_log.csv"

    def snapshot_file(self, source, target_name):
        if not self.enabled or source is None:
            return
        source = Path(source)
        if source.is_file():
            shutil.copy2(source, self.run_dir / target_name)

    def _writer(self, name, frame):
        writer = self._writers.get(name)
        if writer is not None:
            return writer
        height, width = frame.shape[:2]
        path = self.run_dir / name
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"unable to create archived video: {path}")
        self._writers[name] = writer
        return writer

    def write_frames(self, raw, annotated, birdeye):
        if not self.enabled:
            return
        self._writer("raw.mp4", raw).write(raw)
        self._writer("annotated.mp4", annotated).write(annotated)
        if birdeye is not None:
            self._writer("birdeye.mp4", birdeye).write(birdeye)

    def write_status(self, payload):
        if self.enabled:
            write_status(self.status_path, payload)

    def close(self):
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

    def copy_log(self, source):
        if self.enabled and Path(source).is_file():
            shutil.copy2(source, self.log_path)


class SurfaceOnlyDetector:
    """Provide mask geometry without running an unused semantic model.

    The classical outer-loop controller derives its corridor directly from
    every fresh camera frame. It only needs empty, correctly sized semantic
    masks so the common visualization and perspective code can stay shared.
    """

    source_name = "surface-only"

    def __init__(self, width=320, height=180):
        self.width = int(width)
        self.height = int(height)
        if self.width < 16 or self.height < 16:
            raise ValueError("surface mask dimensions must be at least 16 pixels")

    def predict_masks(self, _frame):
        shape = (self.height, self.width)
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)


def build_components(config, motors_enabled, calibration_path):
    perception = config.get("perception", {})
    outer_loop_config = OuterLoopBoundaryConfig(**config.get("outer_loop", {}))
    if not outer_loop_config.enabled:
        raise ValueError("outer_loop must be enabled for onboard LCC")
    if (
        outer_loop_config.navigation_mode == "boundary"
        and outer_loop_config.include_lane_mask
    ):
        raise ValueError("boundary mode requires include_lane_mask=false")
    detector = SurfaceOnlyDetector(
        width=int(perception.get("mask_width", 320)),
        height=int(perception.get("mask_height", 180)),
    )

    estimator = RoadCenterlineEstimator(**config.get("centerline", {}))
    controller = LaneCenteringController(LCCConfig(**config.get("lcc", {})))
    boundary_tracker = (
        OuterLoopBoundaryTracker(outer_loop_config)
        if outer_loop_config.enabled
        else None
    )
    mapper = (
        PerspectiveMapper.from_yaml(calibration_path)
        if calibration_path is not None
        else None
    )

    chassis = None
    if motors_enabled:
        from vehicle_control.hardware import RospbotChassis

        chassis = RospbotChassis()
    driver = SafeWheelDriver(
        chassis=chassis,
        motors_enabled=motors_enabled,
        config=WheelMappingConfig(**config.get("wheels", {})),
    )
    safety = config.get("safety", {})
    watchdog = CommandWatchdog(
        driver,
        timeout=float(safety.get("watchdog_timeout", 3.0)),
        check_interval=float(safety.get("watchdog_check_interval", 0.05)),
    )
    return detector, estimator, controller, mapper, boundary_tracker, driver, watchdog


def fuse_tracking_confidence(
    centerline_confidence: float,
    boundary_confidence: float,
) -> float:
    """Combine two confidence stages without double-penalizing one corridor.

    The centerline estimate is computed from the boundary tracker's corridor,
    so multiplying both values treats the same weak observation as two
    independent failures.  The weaker value is still a conservative joint
    confidence: either stage can stop the car, while a valid single-boundary
    corner does not lose confidence a second time.
    """
    return float(
        np.clip(
            min(float(centerline_confidence), float(boundary_confidence)),
            0.0,
            1.0,
        )
    )


def analyze(
    detector,
    estimator,
    controller,
    mapper,
    frame,
    route_hint,
    dt,
    maximum_inference_time,
    boundary_tracker=None,
):
    started = time.monotonic()
    drivable_mask, lane_mask = detector.predict_masks(frame)
    inference_time = time.monotonic() - started
    control_drivable = (
        mapper.warp_mask(drivable_mask) if mapper is not None else drivable_mask
    )
    control_lane = mapper.warp_mask(lane_mask) if mapper is not None else lane_mask
    boundary_started = time.monotonic()
    boundary_result = None
    yellow_hazard = False
    ego_yellow_ratio = 0.0
    display_drivable = drivable_mask
    if boundary_tracker is not None:
        yellow_mask = boundary_tracker.yellow_mask(frame, lane_mask.shape)
        # The calibration trapezoid can end where the lane boundaries leave
        # the image, well ahead of the physical chassis. The actual vehicle
        # footprint therefore stays in the raw image's narrow bottom-centre
        # zone; only boundary fitting uses the bird's-eye yellow mask.
        yellow_hazard, ego_yellow_ratio = boundary_tracker.yellow_under_ego(
            yellow_mask
        )
        road_surface_mask = boundary_tracker.road_surface_mask(
            frame, lane_mask.shape
        )
        control_yellow = (
            mapper.warp_mask(yellow_mask) if mapper is not None else yellow_mask
        )
        control_surface = (
            mapper.warp_mask(road_surface_mask)
            if mapper is not None
            else road_surface_mask
        )
        control_surface &= (control_yellow == 0).astype(np.uint8)
        if boundary_tracker.config.navigation_mode == "surface":
            boundary_result = BoundaryTrackResult(
                bool(np.any(control_surface)),
                1.0 if np.any(control_surface) else 0.0,
                control_surface,
                "surface",
                reason="non-green outer-route surface",
            )
        else:
            boundary_result = boundary_tracker.update(
                control_surface,
                control_lane,
                control_yellow,
            )
        boundary_result.ego_yellow_ratio = ego_yellow_ratio
        boundary_result.yellow_hazard = yellow_hazard
        control_drivable = boundary_result.corridor_mask
        display_drivable = (
            mapper.camera_mask(control_drivable)
            if mapper is not None
            else control_drivable
        )
    boundary_time = time.monotonic() - boundary_started
    if yellow_hazard:
        estimate = LaneEstimate(
            False,
            0.0,
            reason="yellow boundary entered the vehicle safety zone",
        )
    elif boundary_result is not None and not boundary_result.valid:
        estimate = LaneEstimate(
            False,
            boundary_result.confidence,
            reason=boundary_result.reason,
        )
    else:
        fresh_boundary_sources = {"both", "outer+width", "inner+width"}
        if (
            boundary_result is not None
            and boundary_result.source in fresh_boundary_sources
            and boundary_result.left_curve.size
            and boundary_result.right_curve.size
        ):
            estimate = estimator.estimate_from_boundaries(
                boundary_result.left_curve,
                boundary_result.right_curve,
                boundary_result.corridor_mask.shape[1],
            )
            # Retain the surface-mask estimator as a defensive fallback for a
            # malformed fitted curve. Fresh valid curves are the primary LCC
            # geometry; history/dropout sources still use the corridor path so
            # the bounded corner-continuation safety gate remains in control.
            if not estimate.valid:
                estimate = estimator.estimate(
                    control_drivable, control_lane, route_hint
                )
        else:
            estimate = estimator.estimate(
                control_drivable, control_lane, route_hint
            )
        if boundary_result is not None:
            estimate.confidence = fuse_tracking_confidence(
                estimate.confidence,
                boundary_result.confidence,
            )
    estimate.confidence = float(np.clip(estimate.confidence, 0.0, 1.0))
    command = controller.update(estimate, dt)
    if inference_time + boundary_time > maximum_inference_time:
        command = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            estimate.confidence,
            "inference exceeded safety limit",
        )
    display_estimate = (
        mapper.camera_estimate(estimate, drivable_mask.shape)
        if mapper is not None
        else estimate
    )
    annotated = render_debug_frame(
        frame,
        display_drivable,
        lane_mask,
        display_estimate,
        command,
        inference_time * 1000,
        latency_label=(
            "boundary"
            if boundary_result is None
            else f"boundary/{boundary_result.source}"
        ),
    )
    return (
        estimate,
        command,
        annotated,
        inference_time,
        boundary_time,
        boundary_result,
    )


def write_status(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def render_birdeye_debug(boundary_result, estimate):
    """Render the exact corridor geometry consumed by the controller."""
    corridor = (np.asarray(boundary_result.corridor_mask) > 0).astype(np.uint8)
    height, width = corridor.shape
    output = np.full((height, width, 3), 28, dtype=np.uint8)
    output[corridor > 0] = (40, 120, 40)
    for curve, color in (
        (boundary_result.left_curve, (255, 160, 0)),
        (boundary_result.right_curve, (255, 160, 0)),
    ):
        if curve.size == height:
            points = np.column_stack(
                [np.clip(curve, 0, width - 1), np.arange(height)]
            ).astype(np.int32)
            cv2.polylines(output, [points], False, color, 2)
    if estimate.centerline.size:
        cv2.polylines(output, [estimate.centerline], False, (0, 255, 255), 3)
    ego = (width // 2, int(height * 0.92))
    cv2.circle(output, ego, 5, (255, 255, 255), -1)
    if estimate.lookahead_point is not None:
        cv2.arrowedLine(
            output,
            ego,
            tuple(map(int, estimate.lookahead_point)),
            (255, 255, 255),
            2,
        )
    return output


def main():
    args = build_parser().parse_args()
    config_path, config = load_config(args.config)
    if args.max_samples < 0:
        raise ValueError("--max-samples must not be negative")
    if args.max_runtime_seconds < 0:
        raise ValueError("--max-runtime-seconds must not be negative")
    calibration_path = repo_path(
        config.get("perspective", {}).get("calibration")
    )
    camera_config = dict(config.get("camera", {}))
    if args.camera_index is not None:
        camera_config["index"] = args.camera_index
    safety_config = config.get("safety", {})
    maximum_inference_time = float(
        safety_config.get("maximum_inference_time", 2.5)
    )
    watchdog_timeout = float(safety_config.get("watchdog_timeout", 3.0))
    if maximum_inference_time <= 0:
        raise ValueError("maximum_inference_time must be positive")
    if watchdog_timeout <= maximum_inference_time:
        raise ValueError(
            "watchdog_timeout must be greater than maximum_inference_time"
        )
    validate_motor_request(
        args.enable_motors,
        args.video,
        args.confirm_motor_motion,
        calibration_path,
        camera_config,
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
        gimbal_commands = initialize_configured_gimbal(camera_config)
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
    archive = RunArchive(
        output_dir,
        enabled=(
            bool(runtime_config.get("archive_runs", True))
            and not args.no_run_archive
        ),
        fps=float(runtime_config.get("archive_fps", archive_fps)),
        mode="hardware" if args.enable_motors else "dryrun",
    )
    archive.snapshot_file(config_path, "runtime_config.yaml")
    archive.snapshot_file(calibration_path, "perspective_calibration.yaml")
    debug_output_dir = None
    if args.save_debug_frames:
        debug_output_dir = (
            archive.run_dir / "frames"
            if archive.enabled
            else output_dir / f"debug_{int(time.time())}"
        )
        debug_output_dir.mkdir(parents=True, exist_ok=False)
    detector = estimator = controller = mapper = boundary_tracker = driver = watchdog = None
    motion_gate = None
    corner_gate = None
    log_handle = None
    last_row = None
    termination_reason = "runtime shutdown"
    try:
        (
            detector,
            estimator,
            controller,
            mapper,
            boundary_tracker,
            driver,
            watchdog,
        ) = build_components(config, args.enable_motors, calibration_path)
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
        )
        controller.reset()
        if boundary_tracker is not None:
            boundary_tracker.reset()
        driver.stop("warmup complete; waiting for fresh frame")
        watchdog.arm()

        fieldnames = [
            "sample",
            "timestamp_s",
            "frame_age_s",
            "inference_ms",
            "boundary_ms",
            "boundary_source",
            "boundary_confidence",
            "lane_width_ratio",
            "boundary_visible_ratio",
            "ego_yellow_ratio",
            "yellow_hazard",
            "valid",
            "confidence",
            "estimate_reason",
            "lateral_error",
            "heading_error",
            "near_heading_error",
            "action",
            "steering",
            "tight_turn_factor",
            "left_speed",
            "right_speed",
            "left_pwm",
            "right_pwm",
            "front_left_pwm",
            "rear_left_pwm",
            "front_right_pwm",
            "rear_right_pwm",
            "reason",
            "corner_continuation_active",
            "corner_continuation_holding",
            "corner_continuation_hold_age_s",
            "corner_continuation_progress_age_s",
            "corner_continuation_best_heading",
            "corner_apex_active",
            "corner_apex_age_s",
            "corner_apex_exit_valid_count",
            "motion_gate_ready",
            "motion_gate_valid_frames",
        ]
        log_handle = log_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
        writer.writeheader()

        previous_timestamp = None
        sample = 0
        motion_gate = PerceptionMotionGate(
            resume_valid_frames=int(
                safety_config.get("resume_valid_frames", 4)
            ),
            resume_min_confidence=float(
                safety_config.get("resume_min_confidence", 0.0)
            ),
        )
        corner_gate = CornerContinuationGate(
            CornerContinuationConfig(
                **safety_config.get("corner_continuation", {})
            )
        )
        route_hint = str(config.get("runtime", {}).get("route_hint", "center"))
        max_inference = maximum_inference_time
        runtime_started = time.monotonic()
        save_latest = bool(
            config.get("runtime", {}).get("save_latest_frame", True)
        )
        print(
            f"Onboard runtime started: mode={'HARDWARE' if args.enable_motors else 'DRY-RUN'}, "
            f"source={args.video or 'camera'}, calibration={calibration_path or 'none'}",
            flush=True,
        )
        if archive.enabled:
            print(f"Run archive: {archive.run_dir}", flush=True)

        while True:
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
            frame, timestamp, frame_age = source.read()
            if frame is None:
                termination_reason = "camera/video source ended"
                break
            dt = (
                None
                if previous_timestamp is None
                else max(1e-3, float(timestamp - previous_timestamp))
            )
            previous_timestamp = timestamp
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
            )
            displayed_command = command
            command = corner_gate.filter(
                command,
                estimate,
                boundary_result,
                # Use the capture timeline for both live control and replay.
                # Wall-clock time makes an offline video run much faster than
                # the recorded vehicle and previously skipped the bounded
                # corner-continuation timeout entirely.
                now=timestamp,
            )
            corner_state = corner_gate.get_state(now=timestamp)
            command = motion_gate.filter(command)
            if command != displayed_command:
                cv2.rectangle(
                    annotated,
                    (0, annotated.shape[0] - 32),
                    (annotated.shape[1], annotated.shape[0]),
                    (0, 0, 120),
                    -1,
                )
                cv2.putText(
                    annotated,
                    (
                        f"APPLIED: {command.action} "
                        f"steer={command.steering:+.3f} - {command.reason}"
                    ),
                    (8, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            watchdog.heartbeat()
            wheel_state = driver.apply(command)
            birdeye = (
                None
                if boundary_result is None
                else render_birdeye_debug(boundary_result, estimate)
            )
            if save_latest:
                cv2.imwrite(str(latest_frame_path), annotated)
                if birdeye is not None:
                    cv2.imwrite(str(latest_birdeye_path), birdeye)
            archive.write_frames(frame, annotated, birdeye)
            if debug_output_dir is not None:
                stem = f"{sample:04d}"
                cv2.imwrite(str(debug_output_dir / f"{stem}_raw.jpg"), frame)
                cv2.imwrite(
                    str(debug_output_dir / f"{stem}_annotated.jpg"), annotated
                )
                if birdeye is not None:
                    cv2.imwrite(
                        str(debug_output_dir / f"{stem}_birdeye.png"),
                        birdeye,
                    )

            row = {
                "sample": sample,
                "timestamp_s": round(float(timestamp), 4),
                "frame_age_s": (
                    None if frame_age is None else round(float(frame_age), 4)
                ),
                "inference_ms": round(inference_time * 1000, 3),
                "boundary_ms": round(boundary_time * 1000, 3),
                "boundary_source": (
                    None if boundary_result is None else boundary_result.source
                ),
                "boundary_confidence": (
                    None
                    if boundary_result is None
                    else round(boundary_result.confidence, 5)
                ),
                "lane_width_ratio": (
                    None
                    if boundary_result is None
                    else round(boundary_result.lane_width_ratio, 5)
                ),
                "boundary_visible_ratio": (
                    None
                    if boundary_result is None
                    else round(boundary_result.boundary_visible_ratio, 5)
                ),
                "ego_yellow_ratio": (
                    None
                    if boundary_result is None
                    else round(boundary_result.ego_yellow_ratio, 5)
                ),
                "yellow_hazard": (
                    None
                    if boundary_result is None
                    else boundary_result.yellow_hazard
                ),
                "valid": estimate.valid,
                "confidence": round(estimate.confidence, 5),
                "estimate_reason": estimate.reason,
                "lateral_error": round(estimate.lateral_error, 5),
                "heading_error": round(estimate.heading_error, 5),
                "near_heading_error": round(
                    estimate.near_heading_error, 5
                ),
                "action": command.action,
                "steering": round(command.steering, 5),
                "tight_turn_factor": (
                    None
                    if command.tight_turn_factor is None
                    else round(command.tight_turn_factor, 5)
                ),
                "left_speed": round(command.left_speed, 5),
                "right_speed": round(command.right_speed, 5),
                "left_pwm": wheel_state["left_pwm"],
                "right_pwm": wheel_state["right_pwm"],
                "front_left_pwm": wheel_state["front_left_pwm"],
                "rear_left_pwm": wheel_state["rear_left_pwm"],
                "front_right_pwm": wheel_state["front_right_pwm"],
                "rear_right_pwm": wheel_state["rear_right_pwm"],
                "reason": command.reason,
                "corner_continuation_active": corner_state["active"],
                "corner_continuation_holding": corner_state["holding"],
                "corner_continuation_hold_age_s": (
                    None
                    if corner_state["hold_age_seconds"] is None
                    else round(corner_state["hold_age_seconds"], 5)
                ),
                "corner_continuation_progress_age_s": (
                    None
                    if corner_state["progress_age_seconds"] is None
                    else round(corner_state["progress_age_seconds"], 5)
                ),
                "corner_continuation_best_heading": (
                    None
                    if corner_state["best_heading_magnitude"] is None
                    else round(corner_state["best_heading_magnitude"], 5)
                ),
                "corner_apex_active": corner_state["apex_active"],
                "corner_apex_age_s": (
                    None
                    if corner_state["apex_age_seconds"] is None
                    else round(corner_state["apex_age_seconds"], 5)
                ),
                "corner_apex_exit_valid_count": corner_state[
                    "apex_exit_valid_count"
                ],
                "motion_gate_ready": motion_gate.get_state()["ready"],
                "motion_gate_valid_frames": motion_gate.get_state()[
                    "consecutive_valid"
                ],
            }
            last_row = row
            writer.writerow(row)
            log_handle.flush()
            live_status = {
                "mode": wheel_state["mode"],
                "source": str(
                    args.video or f"camera:{camera_config.get('index', 0)}"
                ),
                "config": str(config_path),
                "calibration": (
                    str(calibration_path) if calibration_path else None
                ),
                "last_result": row,
                "wheel_driver": driver.get_state(),
                "watchdog": watchdog.get_state(),
                "corner_continuation": corner_state,
                "motion_gate": motion_gate.get_state(),
                "run_archive": (
                    None if not archive.enabled else str(archive.run_dir)
                ),
            }
            write_status(status_path, live_status)
            archive.write_status(live_status)
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
                f"boundary_ms={boundary_time * 1000:.1f}",
                flush=True,
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
        if watchdog is not None:
            watchdog.close()
        if driver is not None:
            driver.stop(termination_reason)
            final_wheel_state = driver.get_state()
            final_status = {
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
                    "stopped": True,
                    "completed_at": time.time(),
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
                "run_archive": (
                    None if not archive.enabled else str(archive.run_dir)
                ),
            }
            write_status(status_path, final_status)
            archive.write_status(final_status)
        source.close()
        archive.close()
        if log_handle is not None:
            log_handle.close()
        archive.copy_log(log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
