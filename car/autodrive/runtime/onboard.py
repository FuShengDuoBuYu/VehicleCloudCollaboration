#!/usr/bin/env python3
"""Safe onboard outer-loop LCC; dry-run unless motors are explicitly enabled."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import sys
import threading
import time

import cv2
import numpy as np
import yaml


AUTODRIVE_DIR = Path(__file__).resolve().parents[1]
CAR_DIR = AUTODRIVE_DIR.parent
CONTROL_DIR = CAR_DIR / "control"
REPO_ROOT = CAR_DIR.parent
for path in (CAR_DIR, CONTROL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
from autodrive.control.field_trial import (
    validate_field_trial_config, field_trial_recording_ready, validate_stationary_corner_config,
)
from autodrive.control.stationary_corner import StationaryCornerController, StationaryCornerConfig
from autodrive.runtime.sensor_recording import raw_sensor_recording_ready
LOCAL_CONFIG = AUTODRIVE_DIR / "config" / "onboard_runtime.yaml"
DEFAULT_CONFIG = LOCAL_CONFIG
MOTOR_CONFIRMATION = "I_UNDERSTAND_MOTORS_WILL_MOVE"


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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run outer-loop lane centering (dry-run by default)"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--vehicle-profile",
        help=(
            "Vehicle profile ID or YAML path; overrides vehicle.profile in "
            "the runtime config"
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="Use a recorded onboard video instead of a camera")
    source.add_argument("--camera-index", type=int, help="Override configured camera index")
    parser.add_argument("--sample-every", type=int, help="Video frame sampling interval")
    parser.add_argument(
        "--output-dir",
        help="Override output directory (useful for isolated offline replay)",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 runs until EOF/interrupt")
    parser.add_argument('--panel-session-dir',help='Local panel cancellation/lease directory')
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="0 runs until EOF/interrupt; otherwise stop after this wall-clock duration",
    )
    parser.add_argument(
        "--max-motion-seconds", type=float, default=0.0,
        help="Stop this run after this duration from its first nonzero PWM (0 disables)",
    )
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        help=(
            "Best-effort raw, annotated, and bird's-eye snapshots for diagnosis; "
            "slow storage may drop old pending frames"
        ),
    )
    parser.add_argument(
        "--no-run-archive",
        action="store_true",
        help="Disable per-run videos and snapshots (intended for local replay only)",
    )
    parser.add_argument(
        "--enable-motors",
        action="store_true",
        help=(
            "Enable physical motor output through the selected vehicle backend; "
            "camera source, calibration, and a calibrated profile are required"
        ),
    )
    parser.add_argument(
        "--field-trial", action="store_true",
        help="Bounded, recorded raw-image YOLO trial after verified wheel/start-stop checks",
    )
    parser.add_argument(
        "--confirm-motor-motion",
        default="",
        help=f"Required with --enable-motors: {MOTOR_CONFIRMATION}",
    )
    return parser


def load_config(path, vehicle_profile=None):
    return load_runtime_config(path, vehicle_selector=vehicle_profile)


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
    image_space_trial=False,
    max_runtime_seconds=0,
    panel_controlled=False,
):
    if not enable_motors:
        return
    if video:
        raise ValueError("physical motors cannot be enabled with --video")
    if confirmation != MOTOR_CONFIRMATION:
        raise ValueError(
            "--enable-motors requires the exact --confirm-motor-motion value"
        )
    if image_space_trial:
        if (not math.isfinite(max_runtime_seconds) or max_runtime_seconds < 0
                or panel_controlled is not True and not 0 < max_runtime_seconds <= 120):
            raise ValueError("field trial requires a finite runtime in (0, 120] seconds")
        return
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


def _offer_latest(work_queue, item):
    """Enqueue without blocking, replacing the oldest queued diagnostic.

    Diagnostics must never delay a motor update.  When storage falls behind,
    retaining the newest observation is also more useful than allowing an old
    backlog to hide the vehicle's current state.
    """
    try:
        work_queue.put_nowait(item)
        return True, 0
    except queue.Full:
        pass

    try:
        work_queue.get_nowait()
        work_queue.task_done()
    except queue.Empty:
        # The consumer won the race after ``Full``; retrying below is safe.
        pass
    try:
        work_queue.put_nowait(item)
        return True, 1
    except queue.Full:
        # A consumer cannot cause this in the single-producer design, but do
        # not turn a diagnostic race back into a control-loop wait.
        return False, 1


class RunArchive:
    """Persist replay-aligned videos without blocking the control loop."""

    def __init__(self, output_dir, enabled, fps, mode, queue_frames=32):
        self.enabled = bool(enabled)
        self.fps = max(1.0, float(fps))
        self.run_dir = None
        self.status_path = None
        self.log_path = None
        self._writers = {}
        self._frame_queue = None
        self._writer_thread = None
        self._writer_error = None
        self._stats_lock = threading.Lock()
        self._enqueued_frames = 0
        self._written_frames = 0
        self._dropped_frames = 0
        self._archive_log_handle = None
        self._archive_log_writer = None
        self._archived_semantic_sequences = set()
        if not self.enabled:
            return
        queue_frames = int(queue_frames)
        if queue_frames < 1:
            raise ValueError("archive queue_frames must be at least 1")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{mode}"
        self.run_dir = Path(output_dir) / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.status_path = self.run_dir / "status.json"
        self.log_path = self.run_dir / "onboard_log.csv"
        self._frame_queue = queue.Queue(maxsize=queue_frames)
        self._writer_thread = threading.Thread(
            target=self._write_loop,
            name="lcc-run-archive",
            daemon=True,
        )
        self._writer_thread.start()

    def snapshot_file(self, source, target_name):
        if not self.enabled or source is None:
            return
        source = Path(source)
        if source.is_file():
            shutil.copy2(source, self.run_dir / target_name)

    def snapshot_mapping(self, mapping, target_name):
        if not self.enabled:
            return
        (self.run_dir / target_name).write_text(
            yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

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

    def _write_frames_now(self, raw, annotated, birdeye):
        self._writer("raw.mp4", raw).write(raw)
        self._writer("annotated.mp4", annotated).write(annotated)
        if birdeye is not None:
            self._writer("birdeye.mp4", birdeye).write(birdeye)

    def _write_record_now(self, raw, annotated, birdeye, row, semantic_observation=None):
        if semantic_observation is not None:
            sequence = int(semantic_observation['sequence'])
            if sequence not in self._archived_semantic_sequences:
                folder = self.run_dir / 'semantic_inputs'
                folder.mkdir(exist_ok=True)
                # Lossless actual model input and outputs; no JPEG/MP4 roundtrip
                # when reproducing a live decision. Written on archive thread.
                # Compression measured ~71 ms per observation on this Jetson,
                # overflowing the 20 Hz video queue. Uncompressed NPZ is still
                # lossless (~6.5 ms), with explicit disk/write failure stopping.
                np.savez(folder / ('%08d.npz' % sequence),
                    frame=semantic_observation['frame'], road=semantic_observation['mask'],
                    lane=semantic_observation['lane_mask'],
                    captured_at=semantic_observation['captured_at'],
                    sequence=sequence,
                    detections_json=json.dumps(semantic_observation.get('detections', [])))
                self._archived_semantic_sequences.add(sequence)
        self._write_frames_now(raw, annotated, birdeye)
        # Keep the archive sidecar exactly aligned with the frames that were
        # actually encoded.  If backpressure forced a frame drop, deterministic
        # replay still receives the correct timestamp for every retained frame.
        if self._archive_log_writer is None:
            self._archive_log_handle = self.log_path.open(
                "w", encoding="utf-8", newline=""
            )
            self._archive_log_writer = csv.DictWriter(
                self._archive_log_handle,
                fieldnames=list(row.keys()),
            )
            self._archive_log_writer.writeheader()
        self._archive_log_writer.writerow(row)
        self._archive_log_handle.flush()

    def _write_loop(self):
        try:
            while True:
                record = self._frame_queue.get()
                try:
                    if record is None:
                        break
                    if self._writer_error is None:
                        self._write_record_now(*record)
                        with self._stats_lock:
                            self._written_frames += 1
                    else:
                        with self._stats_lock:
                            self._dropped_frames += 1
                except Exception as exc:
                    self._writer_error = exc
                    with self._stats_lock:
                        self._dropped_frames += 1
                finally:
                    self._frame_queue.task_done()
        finally:
            for writer in self._writers.values():
                writer.release()
            self._writers.clear()
            if self._archive_log_handle is not None:
                self._archive_log_handle.close()
                self._archive_log_handle = None
                self._archive_log_writer = None

    def write_frames(self, raw, annotated, birdeye, row, semantic_observation=None):
        if not self.enabled:
            return False
        if self._writer_error is not None:
            with self._stats_lock:
                self._dropped_frames += 1
            return False
        accepted, dropped = _offer_latest(
            self._frame_queue,
            (raw, annotated, birdeye, dict(row), semantic_observation),
        )
        with self._stats_lock:
            self._enqueued_frames += int(accepted)
            self._dropped_frames += dropped + int(not accepted)
        return accepted

    def get_state(self):
        with self._stats_lock:
            state = {
                "enabled": self.enabled,
                "queue_capacity": (
                    0 if self._frame_queue is None else self._frame_queue.maxsize
                ),
                "queue_depth": (
                    0 if self._frame_queue is None else self._frame_queue.qsize()
                ),
                "enqueued_frames": self._enqueued_frames,
                "written_frames": self._written_frames,
                "dropped_frames": self._dropped_frames,
                "error": (
                    None
                    if self._writer_error is None
                    else str(self._writer_error)
                ),
            }
        return state

    def write_status(self, payload):
        if self.enabled:
            write_status(self.status_path, payload)

    def close(self):
        if not self.enabled or self._writer_thread is None:
            return
        self._frame_queue.put(None)
        self._writer_thread.join()
        self._writer_thread = None


class AsyncCsvLogger:
    """Write control records in the background with bounded memory."""

    def __init__(self, path, fieldnames, queue_rows=256):
        queue_rows = int(queue_rows)
        if queue_rows < 1:
            raise ValueError("CSV queue_rows must be at least 1")
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._queue = queue.Queue(maxsize=queue_rows)
        self._lock = threading.Lock()
        self._enqueued = 0
        self._written = 0
        self._dropped = 0
        self._error = None
        self._thread = threading.Thread(
            target=self._write_loop,
            name="lcc-csv-log",
            daemon=True,
        )
        self._thread.start()

    def _write_loop(self):
        try:
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                while True:
                    row = self._queue.get()
                    try:
                        if row is None:
                            break
                        writer.writerow(row)
                        handle.flush()
                        with self._lock:
                            self._written += 1
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            with self._lock:
                self._error = exc

    def submit(self, row):
        with self._lock:
            failed = self._error is not None
        if failed:
            with self._lock:
                self._dropped += 1
            return False
        accepted, dropped = _offer_latest(self._queue, dict(row))
        with self._lock:
            self._enqueued += int(accepted)
            self._dropped += dropped + int(not accepted)
        return accepted

    def get_state(self):
        with self._lock:
            return {
                "queue_capacity": self._queue.maxsize,
                "queue_depth": self._queue.qsize(),
                "enqueued_rows": self._enqueued,
                "written_rows": self._written,
                "dropped_rows": self._dropped,
                "error": None if self._error is None else str(self._error),
            }

    def close(self):
        if self._thread is None:
            return
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self._thread = None


class LivePublisher:
    """Publish replaceable web snapshots/status outside motor control."""

    def __init__(
        self,
        status_path,
        archive_status_path,
        latest_frame_path,
        latest_birdeye_path,
        save_latest,
    ):
        self.status_path = Path(status_path)
        self.archive_status_path = (
            None
            if archive_status_path is None
            else Path(archive_status_path)
        )
        self.latest_frame_path = Path(latest_frame_path)
        self.latest_birdeye_path = Path(latest_birdeye_path)
        self.save_latest = bool(save_latest)
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._published = 0
        self._dropped = 0
        self._error = None
        self._thread = threading.Thread(
            target=self._write_loop,
            name="lcc-live-publisher",
            daemon=True,
        )
        self._thread.start()

    def _write_loop(self):
        try:
            while True:
                record = self._queue.get()
                try:
                    if record is None:
                        break
                    payload, annotated, birdeye = record
                    if self.save_latest:
                        self._write_image(self.latest_frame_path, annotated)
                        if birdeye is not None:
                            self._write_image(self.latest_birdeye_path, birdeye)
                    write_status(self.status_path, payload)
                    if self.archive_status_path is not None:
                        write_status(self.archive_status_path, payload)
                    with self._lock:
                        self._published += 1
                finally:
                    self._queue.task_done()
        except Exception as exc:
            with self._lock:
                self._error = exc

    @staticmethod
    def _write_image(path, frame):
        # Readers see either complete old JPEG or complete new JPEG.
        temporary = path.with_name(path.stem + '.pending' + path.suffix)
        if not cv2.imwrite(str(temporary), frame):
            raise RuntimeError('failed to write live image')
        temporary.replace(path)

    def publish(self, payload, annotated, birdeye):
        with self._lock:
            failed = self._error is not None
        if failed:
            with self._lock:
                self._dropped += 1
            return False
        accepted, dropped = _offer_latest(
            self._queue,
            (dict(payload), annotated, birdeye),
        )
        with self._lock:
            self._dropped += dropped + int(not accepted)
        return accepted

    def get_state(self):
        with self._lock:
            return {
                "queue_depth": self._queue.qsize(),
                "published_updates": self._published,
                "dropped_updates": self._dropped,
                "error": None if self._error is None else str(self._error),
            }

    def close(self):
        if self._thread is None:
            return
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self._thread = None


class DebugFrameWriter:
    """Best-effort debug image writer that cannot stall vehicle control."""

    def __init__(self, output_dir, queue_frames=8):
        self.output_dir = None if output_dir is None else Path(output_dir)
        self._queue = None
        self._thread = None
        self._lock = threading.Lock()
        self._written = 0
        self._dropped = 0
        self._error = None
        if self.output_dir is None:
            return
        queue_frames = int(queue_frames)
        if queue_frames < 1:
            raise ValueError("debug queue_frames must be at least 1")
        self._queue = queue.Queue(maxsize=queue_frames)
        self._thread = threading.Thread(
            target=self._write_loop,
            name="lcc-debug-frames",
            daemon=True,
        )
        self._thread.start()

    def _write_loop(self):
        try:
            while True:
                record = self._queue.get()
                try:
                    if record is None:
                        break
                    sample, raw, annotated, birdeye = record
                    stem = f"{sample:04d}"
                    if not cv2.imwrite(
                        str(self.output_dir / f"{stem}_raw.jpg"), raw
                    ):
                        raise RuntimeError("failed to write raw debug frame")
                    if not cv2.imwrite(
                        str(self.output_dir / f"{stem}_annotated.jpg"),
                        annotated,
                    ):
                        raise RuntimeError("failed to write annotated debug frame")
                    if birdeye is not None and not cv2.imwrite(
                        str(self.output_dir / f"{stem}_birdeye.png"), birdeye
                    ):
                        raise RuntimeError("failed to write bird's-eye debug frame")
                    with self._lock:
                        self._written += 1
                finally:
                    self._queue.task_done()
        except Exception as exc:
            with self._lock:
                self._error = exc

    def submit(self, sample, raw, annotated, birdeye):
        if self._queue is None:
            return False
        with self._lock:
            failed = self._error is not None
        if failed:
            with self._lock:
                self._dropped += 1
            return False
        accepted, dropped = _offer_latest(
            self._queue,
            (int(sample), raw, annotated, birdeye),
        )
        with self._lock:
            self._dropped += dropped + int(not accepted)
        return accepted

    def get_state(self):
        with self._lock:
            return {
                "enabled": self._queue is not None,
                "queue_depth": 0 if self._queue is None else self._queue.qsize(),
                "written_frames": self._written,
                "dropped_frames": self._dropped,
                "error": None if self._error is None else str(self._error),
            }

    def close(self):
        if self._thread is None:
            return
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self._thread = None


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


def cloud_configuration(config):
    settings = CloudArbitrationConfig(**config.get("cloud_arbitration", {}))
    if settings.enabled and config.get("perception", {}).get("mode") != "yolopv2":
        raise ValueError("cloud arbitration requires primary YOLOPv2 observations")
    return settings


STATIONARY_BOUNDARY_DIAGNOSTICS = (
    'semantic_front_observed', 'semantic_observed_front_ratio',
    'semantic_front_spread_ratio', 'semantic_right_exit_support_ratio',
    'semantic_corner_reject_reason', 'semantic_pivot_road_support_ratio',
    'semantic_track_surface_pixels','semantic_turn_entry_ready','semantic_exit_heading_error',
    'semantic_forward_road_valid','semantic_rotation_yellow_ratio','semantic_rotation_heading_error',
)

VISUAL_CORNER_DIAGNOSTICS = (
    'visual_corner_active', 'visual_corner_phase', 'visual_corner_exit_count',
    'visual_corner_reason', 'visual_path_margin_px', 'visual_required_margin_px',
    'visual_forward_clear',
    'visual_rotation_margin_px',
    'visual_rotation_required_margin_px','visual_rotation_origin_x_px','visual_rotation_origin_y_px',
)


STATIONARY_LOG_FIELDS = (
    'stationary_state_before', 'stationary_phase', 'stationary_yaw_progress',
    'stationary_decision', 'stationary_reason', 'stationary_startup_ready',
    'stationary_pivot_verified', 'imu_yaw_rad', 'imu_received_monotonic',
    'imu_valid', 'imu_stale', 'imu_age_s', 'control_monotonic', 'frame_age_exact_s',
    'stationary_external_allowed', 'stationary_perception_budget_valid',
    'stationary_application_allowed',
    'stationary_continuous', 'stationary_motion_deadline',
    'semantic_front_boundary_ratio', 'semantic_right_exit_observed', 'semantic_hard_safe',
    *STATIONARY_BOUNDARY_DIAGNOSTICS,
    *VISUAL_CORNER_DIAGNOSTICS,
)


ENCODER_LOG_FIELDS = (
    'encoder_1', 'encoder_2', 'encoder_3', 'encoder_4',
    'encoder_received_monotonic', 'encoder_age_s', 'encoder_valid', 'encoder_stale',
)


APPLICATION_LOG_FIELDS = (
    'stationary_application_monotonic', 'stationary_application_reason',
    'semantic_application_age_s', 'frame_application_age_s', 'imu_application_age_s',
)


class StationaryCornerRuntime:
    """Deterministic adapter shared by live control and recorded-input replay.

    No clock or device reads occur here. ``external_motion_allowed`` feeds the
    previous final cloud veto and current recording/watchdog gates back into
    the controller. Cloud filtering remains the caller's final veto. A dry-run
    replay may set pivot_verified=True; physical callers must validate evidence.
    """
    def __init__(self, config, motion_gate, *, pivot_verified=False):
        if config.visual_feedback:
            from autodrive.control.visual_feedback import VisualFeedbackController
            self.controller = VisualFeedbackController(config.step_max_seconds, config.step_max_yaw_rad,
                config.step_settle_seconds, config.semantic_max_age_seconds,
                config.imu_max_age_seconds, config.yaw_sign)
        else:
            self.controller = StationaryCornerController(config)
        self.config = config
        self.motion_gate = motion_gate
        self.pivot_verified = pivot_verified is True
        self._startup_ready = False
        self._state = {}
        self._visual_corner_active = False
        self._visual_exit_count = 0
        self._visual_exit_capture = None
        self._visual_last_turn_at = None
        self._visual_exit_released_this_cycle = False

    @staticmethod
    def _finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    @staticmethod
    def _stop(reason):
        return DifferentialDriveCommand('stop', 0., 0., 0., 0., reason, 0.)

    def filter(self, command, estimate, boundary, *, now, imu_sample=None,
               frame_age_seconds=0., perception_budget_valid=True, external_motion_allowed=True):
        if not self.config.enabled:
            return self.motion_gate.filter(command, estimate, boundary)
        if self.config.visual_feedback:
            return self._filter_visual(command, estimate, boundary, now=now, imu_sample=imu_sample,
                frame_age_seconds=frame_age_seconds, perception_budget_valid=perception_budget_valid,
                external_motion_allowed=external_motion_allowed)
        controller_state = self.controller.get_state()
        before = controller_state['state']
        sequence = getattr(boundary, 'semantic_sequence', None)
        previous_sequence = controller_state['last_semantic_sequence']
        new_observation = (type(sequence) is int and sequence >= 0
                           and (previous_sequence is None or sequence > previous_sequence))
        sample = imu_sample if isinstance(imu_sample, dict) else {}
        value = sample.get('value') or {}
        angles = value.get('radians') if isinstance(value, dict) else None
        yaw = angles[2] if isinstance(angles, (list, tuple)) and len(angles) == 3 else None
        yaw = yaw if self._finite(yaw) else None
        receipt = sample.get('received_monotonic')
        receipt = receipt if self._finite(receipt) else None
        age = sample.get('age_s')
        age = age if self._finite(age) else None
        imu_valid = (sample.get('valid') is True and sample.get('stale') is False
                     and yaw is not None and receipt is not None and age is not None
                     and 0 <= age <= self.config.imu_max_age_seconds
                     and self._finite(now) and 0 <= now-receipt <= self.config.imu_max_age_seconds)
        semantic_age = getattr(boundary, 'semantic_result_age_seconds', None)
        semantic_captured_at = getattr(boundary, 'semantic_captured_at', None)
        if semantic_captured_at is not None:
            semantic_age = (now-semantic_captured_at
                if self._finite(now) and self._finite(semantic_captured_at) else None)
            # The same refreshed age is persisted for deterministic replay.
            boundary.semantic_result_age_seconds = semantic_age
        hard_safe = (getattr(boundary, 'semantic_hard_safe', False) is True
                     and self._finite(semantic_age) and 0 <= semantic_age <= self.config.semantic_max_age_seconds
                     and self._finite(frame_age_seconds) and 0 <= frame_age_seconds <= .25
                     and perception_budget_valid is True)
        ordinary_safe = (hard_safe and estimate.valid is True and boundary.valid is True
                         and command.action != 'stop')
        ordinary = command if ordinary_safe else self._stop('stationary corner: ordinary path unsafe')
        active_pivot = before == 'pivot-right' and self.pivot_verified
        if not active_pivot and (not ordinary_safe or external_motion_allowed is not True):
            self.motion_gate.reset('stationary corner: current permission vetoed')
            self._startup_ready = False
        elif before == 'exit':
            # Exit alignment and ordinary recovery are separate evidence
            # windows: three fresh exit observations precede five new ordinary
            # observations. No exit frame is banked toward ordinary recovery.
            self.motion_gate.reset('stationary corner: awaiting exit confirmation')
            self._startup_ready = False
        elif not active_pivot:
            if self.motion_gate.get_state()['ready'] or new_observation:
                ordinary = self.motion_gate.filter(ordinary, estimate, boundary)
            else:
                ordinary = self._stop('stationary corner: waiting for a new ordinary observation')
            self._startup_ready = self.motion_gate.get_state()['ready']
        local_safe = hard_safe and (imu_valid if active_pivot else
                                   ordinary_safe and (before == 'exit' or self._startup_ready))
        local_valid = bool(local_safe and external_motion_allowed is True
                           and (self.pivot_verified or before not in ('settle', 'pivot-right')))
        decision = self.controller.update(now=now,
            semantic_sequence=getattr(boundary, 'semantic_sequence', None), semantic_age_seconds=semantic_age,
            front_boundary_ratio=getattr(boundary, 'semantic_front_boundary_ratio', None),
            right_exit_observed=getattr(boundary, 'semantic_right_exit_observed', False),
            yaw_rad=yaw, imu_timestamp=receipt, imu_valid=bool(imu_valid),
            exit_aligned=bool(ordinary_safe and abs(estimate.heading_error) <= .18
                              and abs(estimate.lateral_error) <= .35), local_valid=local_valid)
        if decision.action == 'pivot-right' and self.pivot_verified and local_valid:
            self.motion_gate.reset('stationary pivot; ordinary recovery must be revalidated')
            self._startup_ready = False
            output = DifferentialDriveCommand('pivot-right', 1., .3, -.3, 1., decision.reason, 0.)
        elif decision.action == 'stop' or not local_valid:
            # Controller-generated zero during corner confirmation/settle is
            # expected. It does not revoke current ordinary-path evidence.
            # An active pivot keeps its own yaw/deadline across a short veto.
            output = self._stop('stationary corner: ' + decision.reason)
        elif before == 'exit':
            output = self._stop('stationary corner: ' + decision.reason
                                + '; ordinary recovery must be revalidated')
        elif decision.action == 'straight':
            if ordinary.action == 'stop':
                output = ordinary
            else:
                speed = max(0., min(command.left_speed, command.right_speed))
                output = DifferentialDriveCommand('forward', 0., speed, speed, command.confidence, decision.reason, 0.)
        else:
            output = ordinary
        self._state = dict(
            stationary_state_before=before, stationary_phase=decision.state,
            stationary_yaw_progress=decision.yaw_progress_rad, stationary_decision=decision.action,
            stationary_reason=decision.reason, stationary_startup_ready=self._startup_ready,
            stationary_pivot_verified=self.pivot_verified, imu_yaw_rad=yaw,
            imu_received_monotonic=receipt, imu_valid=bool(imu_valid),
            imu_stale=sample.get('stale') is not False, imu_age_s=age,
            control_monotonic=now, frame_age_exact_s=frame_age_seconds,
            stationary_external_allowed=external_motion_allowed is True,
            stationary_perception_budget_valid=perception_budget_valid is True,
            stationary_application_allowed=True,
            semantic_front_boundary_ratio=getattr(boundary, 'semantic_front_boundary_ratio', None),
            semantic_right_exit_observed=getattr(boundary, 'semantic_right_exit_observed', False),
            semantic_hard_safe=getattr(boundary, 'semantic_hard_safe', False), local_safe=bool(local_safe))
        self._state.update({field: getattr(boundary, field, None)
                            for field in STATIONARY_BOUNDARY_DIAGNOSTICS})
        return output

    def _filter_visual(self, command, estimate, boundary, *, now, imu_sample,
                       frame_age_seconds, perception_budget_valid, external_motion_allowed):
        self._visual_exit_released_this_cycle = False
        before = self.controller.get_state()
        sequence = getattr(boundary, 'semantic_sequence', None)
        captured = getattr(boundary, 'semantic_captured_at', None)
        age = now-captured if self._finite(now) and self._finite(captured) else None
        if captured is None:
            age = getattr(boundary, 'semantic_result_age_seconds', None)
            captured = now-age if self._finite(age) and self._finite(now) else None
        sample = imu_sample if isinstance(imu_sample, dict) else {}
        angles = (sample.get('value') or {}).get('radians')
        yaw = angles[2] if isinstance(angles, (list, tuple)) and len(angles)==3 else None
        stamp = sample.get('received_monotonic')
        imu_valid = (sample.get('valid') is True and sample.get('stale') is False
            and self._finite(yaw) and self._finite(stamp) and self._finite(now)
            and 0 <= now-stamp <= self.config.imu_max_age_seconds)
        evidence_safe = (getattr(boundary, 'semantic_hard_safe', False) is True
            and boundary.valid is True
            and self._finite(age) and 0 <= age <= self.config.semantic_max_age_seconds
            and self._finite(frame_age_seconds) and 0 <= frame_age_seconds <= .25
            and perception_budget_valid is True and external_motion_allowed is True
            and self._finite(estimate.heading_error) and self._finite(estimate.lateral_error)
            and self._finite(estimate.near_heading_error))
        new = (type(sequence) is int and (before['last_semantic_sequence'] is None
                                        or sequence > before['last_semantic_sequence']))
        desired = command.action
        near_aligned = (abs(estimate.near_heading_error) <= self.config.visual_pivot_heading
                        and abs(estimate.lateral_error) <= .35)
        front_observed = getattr(boundary, 'semantic_front_observed', False) is True
        front_ratio = getattr(boundary, 'semantic_observed_front_ratio', None)
        front_near = front_observed and (not self._finite(front_ratio)
                                         or front_ratio >= self.config.front_near_ratio
                                         or getattr(boundary,'semantic_turn_entry_ready',False) is True)
        exit_heading=getattr(boundary,'semantic_exit_heading_error',None)
        rotation_heading=getattr(boundary,'semantic_rotation_heading_error',None)
        rotation_candidate=bool(self.config.visual_corner_guard
            and getattr(boundary,'semantic_track_surface_pixels',0)>0
            and self.pivot_verified and imu_valid and
            (front_near and getattr(boundary,'semantic_right_exit_observed',False) is True
             and self._finite(exit_heading) and exit_heading>self.config.visual_pivot_heading
             or self._visual_corner_active and self._finite(rotation_heading)
             and rotation_heading>self.config.visual_pivot_heading))
        local_safe=bool(evidence_safe and (rotation_candidate or
            estimate.valid is True and command.action!='stop'
            and getattr(boundary,'semantic_forward_road_valid',True) is True))
        cruise = bool(self.config.continuous_cruise and near_aligned and not front_near
                      and self._finite(command.steering) and abs(command.steering) <= .35
                      and command.action in ('forward', 'turn-left', 'turn-right'))
        clearance = {};rotation={};phase='cruise';corner_reason='guard disabled';exit_confirm=False
        if self.config.visual_corner_guard:
            from autodrive.perception.visual_clearance import assess_visible_clearance,assess_rotation_clearance
            clearance=assess_visible_clearance(getattr(boundary,'corridor_mask',None),
                getattr(boundary,'semantic_yellow_mask',None),estimate.centerline,
                self.config.visual_path_margin_ratio)
            # Exit confirmation must obey the controller's global sequence,
            # capture and clock ordering before changing corner memory.
            ordered=(type(sequence) is int and sequence>=0 and
                (before['last_semantic_sequence'] is None or sequence>=before['last_semantic_sequence']))
            capture_ordered=(self._finite(captured) and
                (before['last_semantic_captured_at'] is None or captured>=before['last_semantic_captured_at']))
            clock_ordered=(self._finite(now) and now>=0 and
                (before['last_control_monotonic'] is None or now>=before['last_control_monotonic']))
            rotation=assess_rotation_clearance(getattr(boundary,'semantic_rotation_mask',None),
                getattr(boundary,'semantic_yellow_mask',None))
            local_safe=local_safe and (rotation['rotation_clear'] if rotation_candidate else clearance['path_safe']) and ordered and capture_ordered and clock_ordered
            if rotation_candidate:cruise=False
            corner_reason=clearance['reason']
            # A briefly straight-looking patch after a pivot is not a corner
            # exit. Keep this memory across missing/expired evidence and vetoes.
            if self._visual_corner_active:
                exit_confirm=bool(local_safe and near_aligned and not front_observed
                                  and clearance['forward_clear'] and
                                  (not getattr(boundary,'semantic_track_surface_pixels',0)
                                   or abs(estimate.heading_error)<=self.config.visual_pivot_heading))
                stopped=before.get('stopped_at')
                exit_ready=(exit_confirm and before['action']=='stop' and self._finite(captured)
                    and self._finite(stopped) and self._visual_last_turn_at is not None
                    and captured>max(stopped,self._visual_last_turn_at))
                if not exit_ready:
                    self._visual_exit_count=0;self._visual_exit_capture=None
                elif new:
                    if self._visual_exit_capture is None or captured>self._visual_exit_capture:
                        self._visual_exit_count+=1;self._visual_exit_capture=captured
                    else:
                        self._visual_exit_count=0;self._visual_exit_capture=None
                    if self._visual_exit_count>=2:
                        self._visual_corner_active=False;exit_confirm=False
                        self._visual_exit_released_this_cycle=True
            cruise=cruise and not front_observed and not self._visual_corner_active
            phase=('turn' if self._visual_corner_active else
                   'approach' if front_observed else 'cruise')
            if exit_confirm:phase='exit-confirm'
        # A far exit can be visible while the near road is still straight.
        # Use the current near path to decide this short physical step.
        approach=(self.config.visual_corner_guard and front_observed
                  and not front_near and not self._visual_corner_active)
        if exit_confirm:
            desired='stop';corner_reason='visual-await-post-stop-exit-observations'
        elif rotation_candidate:
            desired='pivot-right';corner_reason='visual-current-rotation-support'
        elif approach:
            desired='forward';local_safe=local_safe and clearance['forward_clear']
            if not clearance['forward_clear']:corner_reason='visual-forward-approach-not-clear'
        elif (estimate.near_heading_error > self.config.visual_pivot_heading
              or self.config.visual_corner_guard and (front_near or self._visual_corner_active and self._finite(exit_heading))
              and getattr(boundary,'semantic_right_exit_observed',False) is True
              and (self._finite(exit_heading) and exit_heading>self.config.visual_pivot_heading
                   or estimate.heading_error>self.config.visual_pivot_heading)):
            desired = 'pivot-right'
            local_safe = local_safe and self.pivot_verified and imu_valid
            if self.config.visual_corner_guard:
                turn_allowed=(front_near and getattr(boundary,'semantic_right_exit_observed',False) is True
                    or self._visual_corner_active and (getattr(boundary,'semantic_right_exit_observed',False) is True
                        or getattr(boundary,'semantic_track_surface_pixels',0)>0))
                local_safe=local_safe and turn_allowed
                if not turn_allowed:corner_reason='visual-turn-await-current-front-and-exit'
        elif near_aligned and not cruise:
            desired = 'forward'
            if self.config.visual_corner_guard:
                local_safe=local_safe and clearance['forward_clear']
                if not clearance['forward_clear']:corner_reason='visual-forward-step-not-clear'
        if not local_safe:
            self.motion_gate.reset('visual current path veto')
            self._startup_ready = False
        elif not self._startup_ready and new:
            gate_command=command;gate_estimate=estimate
            if rotation_candidate:
                gate_command=DifferentialDriveCommand('pivot-right',1.,.3,-.3,
                    getattr(boundary,'confidence',0.),'current rotation support',0.)
                gate_estimate=replace(estimate,valid=True,confidence=getattr(boundary,'confidence',0.),
                    lateral_error=0.,heading_error=exit_heading if self._finite(exit_heading) else rotation_heading)
            gated = self.motion_gate.filter(gate_command, gate_estimate, boundary)
            self._startup_ready = gated.action != 'stop' and self.motion_gate.get_state()['ready']
        decision = self.controller.update(now=now, semantic_sequence=sequence, captured_at=captured,
            desired_action=desired, local_valid=bool(local_safe and self._startup_ready),
            yaw_rad=yaw, imu_timestamp=stamp, imu_valid=bool(imu_valid), continuous=cruise)
        if self.config.visual_corner_guard and decision.action=='pivot-right':
            self._visual_corner_active=True;self._visual_exit_count=0;self._visual_exit_capture=None
            self._visual_last_turn_at=now;phase='turn'
        if decision.action != 'stop' and self.controller.get_state()['continuous']:
            output = replace(command, reason=decision.reason, tight_turn_factor=0.)
        elif decision.action=='pivot-right':
            output = DifferentialDriveCommand('pivot-right', 1., .3, -.3, estimate.confidence, decision.reason, 0.)
        elif decision.action=='forward':
            speed = max(0., min(command.left_speed, command.right_speed))
            output = DifferentialDriveCommand('forward', 0., speed, speed, estimate.confidence, decision.reason, 0.)
        elif decision.action=='stop':
            output = self._stop(decision.reason)
        else:
            output = command
        self._state = dict(stationary_state_before=before['state'], stationary_phase=decision.state,
            stationary_yaw_progress=decision.yaw_progress_rad, stationary_decision=decision.action,
            stationary_reason=decision.reason, stationary_startup_ready=self._startup_ready,
            stationary_pivot_verified=self.pivot_verified, imu_yaw_rad=yaw,
            imu_received_monotonic=stamp, imu_valid=bool(imu_valid), imu_stale=sample.get('stale') is not False,
            imu_age_s=sample.get('age_s'), control_monotonic=now, frame_age_exact_s=frame_age_seconds,
            stationary_external_allowed=external_motion_allowed is True,
            stationary_perception_budget_valid=perception_budget_valid is True,
            stationary_application_allowed=True, local_safe=bool(local_safe), visual_feedback=True,
            stationary_continuous=self.controller.get_state()['continuous'],
            stationary_motion_deadline=self.controller.get_state()['motion_deadline'])
        self._state.update(visual_corner_active=self._visual_corner_active,
            visual_corner_phase=phase,visual_corner_exit_count=self._visual_exit_count,
            visual_corner_reason=corner_reason,visual_path_margin_px=clearance.get('minimum_margin_px'),
            visual_required_margin_px=clearance.get('required_margin_px'),
            visual_forward_clear=clearance.get('forward_clear'))
        self._state['visual_rotation_margin_px']=rotation.get('minimum_margin_px')
        self._state['visual_rotation_required_margin_px']=rotation.get('required_margin_px')
        self._state['visual_rotation_origin_x_px']=rotation.get('origin_x_px')
        self._state['visual_rotation_origin_y_px']=rotation.get('origin_y_px')
        self._state.update({field: getattr(boundary, field, None) for field in STATIONARY_BOUNDARY_DIAGNOSTICS})
        self._state.update({field: getattr(boundary, field, None) for field in
            ('semantic_front_boundary_ratio', 'semantic_right_exit_observed', 'semantic_hard_safe')})
        return output

    def apply_veto(self, reason, now=None):
        """Record a final zero after filtering without another timed update.

        Live and replay use this explicit event when a previously gated
        observation expires before application. The controller retains its
        angle/deadline and requires a newer semantic frame to resume a pivot.
        """
        decision = (self.controller.veto(reason,now=now) if self.config.visual_feedback
                    else self.controller.veto(reason))
        self.motion_gate.reset(reason)
        self._startup_ready = False
        self._state.update(stationary_phase=decision.state,
            stationary_yaw_progress=decision.yaw_progress_rad,
            stationary_decision=decision.action, stationary_reason=decision.reason,
            stationary_startup_ready=False, stationary_application_allowed=False, local_safe=False)
        if self.config.visual_feedback:
            self._state.update(stationary_continuous=False, stationary_motion_deadline=None)
            if self.config.visual_corner_guard:
                # A proposal is not an applied exit. Discard confirmation if
                # the final application rejects this cycle, including a cycle
                # that provisionally released corner memory.
                if self._visual_exit_released_this_cycle:
                    self._visual_corner_active=True
                self._visual_exit_released_this_cycle=False
                self._visual_exit_count=0;self._visual_exit_capture=None
                self._state.update(visual_corner_active=self._visual_corner_active,
                    visual_corner_exit_count=0,
                    visual_corner_phase='turn' if self._visual_corner_active else 'cruise',
                    visual_corner_reason=reason)
        return self._stop(reason)

    def get_state(self):
        return dict(self._state, controller=self.controller.get_state(), motion_gate=self.motion_gate.get_state())


def apply_runtime_command(driver,command,stationary_runtime=None):
    """Preserve a driver's final rejection for logs and recorded replay."""
    if (stationary_runtime is not None and stationary_runtime.config.visual_feedback
            and command.action!='stop'):
        state=driver.apply(command,deadline=stationary_runtime.controller.get_state()['motion_deadline'])
    else:
        state=driver.apply(command)
    if stationary_runtime is not None and command.action!='stop' and state.get('action')=='stopped':
        command=stationary_runtime.apply_veto(state.get('reason') or 'driver rejected motion',
                                              now=time.monotonic())
    return command,state


def build_components(config, motors_enabled, calibration_path, field_trial=False):
    validate_stationary_corner_config(config, motors_enabled=motors_enabled)
    if field_trial:
        validate_field_trial_config(config)
    cloud_configuration(config)
    perception = config.get("perception", {})
    primary = perception.get("mode", "classical") == "yolopv2"
    if perception.get("mode", "classical") not in {"classical", "yolopv2"}:
        raise ValueError("perception.mode must be classical or yolopv2")
    if primary:
        if not perception.get("yolopv2", {}).get("enabled", False):
            raise ValueError("YOLOPv2 primary mode requires an enabled model")
        if config.get("outer_loop", {}).get("enabled", False):
            raise ValueError("YOLOPv2 primary mode requires outer_loop.enabled=false")
        if config.get("safety", {}).get("corner_continuation", {}).get("enabled", False):
            raise ValueError("YOLOPv2 primary mode requires corner_continuation.enabled=false")
    outer_loop_config = OuterLoopBoundaryConfig(**config.get("outer_loop", {}))
    if not primary and not outer_loop_config.enabled:
        raise ValueError("outer_loop must be enabled for onboard LCC")
    if (
        not primary and outer_loop_config.navigation_mode == "boundary"
        and outer_loop_config.include_lane_mask
    ):
        raise ValueError("boundary mode requires include_lane_mask=false")
    detector = chassis = driver = watchdog = None
    try:
        mask_width = int(perception.get("mask_width", 320))
        mask_height = int(perception.get("mask_height", 180))
        yolopv2_settings = dict(perception.get("yolopv2", {}))
        if bool(yolopv2_settings.get("enabled", False)):
            weights = repo_path(yolopv2_settings.get("weights"))
            yolopv2_settings["weights"] = "" if weights is None else str(weights)
            if bool(yolopv2_settings.get("adaptive_precision", False)):
                int8_weights = repo_path(yolopv2_settings.get("int8_weights"))
                yolopv2_settings["int8_weights"] = (
                    "" if int8_weights is None else str(int8_weights)
                )
            detector_class = YOLOPv2FusionDetector
            if primary:
                from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
                detector_class = YOLOPv2SemanticDetector
            detector_options = ({'outer_loop_config': perception.get('semantic_outer_loop'),
                                 'stability_config': perception.get('semantic_stability'),
                                 'track_color_config': perception.get('track_colors')}
                                if primary else {})
            if primary:
                from autodrive.perception.semantic_outer_route import route_options
                detector_options['outer_loop_config'] = route_options(config)
            if primary and perception.get('track_colors',{}).get('enabled') and calibration_path is not None:
                raise ValueError('track colors require original camera coordinates without perspective warp')
            detector = detector_class(
                YOLOPv2FusionConfig(**yolopv2_settings),
                output_width=mask_width,
                output_height=mask_height,
                **detector_options,
            )
        else:
            detector = SurfaceOnlyDetector(width=mask_width, height=mask_height)

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
            if not field_trial:
                validate_motion_configuration(config.get("vehicle", {}), config.get("wheels", {}))
            chassis = create_chassis(config.get("vehicle", {}), require_motion_calibrated=not field_trial)
        driver = SafeWheelDriver(
            chassis=chassis,
            motors_enabled=motors_enabled,
            config=WheelMappingConfig(**config.get("wheels", {})),
        )
        safety = config.get("safety", {})
        watchdog = CommandWatchdog(
            driver,
            timeout=float(safety.get("watchdog_timeout", 0.40)),
            check_interval=float(safety.get("watchdog_check_interval", 0.05)),
        )
        return detector, estimator, controller, mapper, boundary_tracker, driver, watchdog
    except BaseException as exc:
        cleanup_errors = []
        actions = []
        if watchdog is not None:
            actions.append(('watchdog.close', watchdog.close))
        if driver is not None:
            actions.append(('driver.stop', lambda: driver.stop('component initialization failed')))
        elif chassis is not None:
            actions.append(('chassis.stop', chassis.stop))
        if chassis is not None and hasattr(chassis, 'close'):
            actions.append(('chassis.close', lambda: chassis.close(stop=False)))
        if detector is not None and hasattr(detector, 'close'):
            actions.append(('detector.close', detector.close))
        for name, action in actions:
            try:
                action()
            except Exception as cleanup_exc:
                cleanup_errors.append(dict(action=name, type=type(cleanup_exc).__name__,
                                           message=str(cleanup_exc)))
        if cleanup_errors:
            exc.component_cleanup_errors = cleanup_errors
        raise


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
    captured_at=None,
    deferred_render=None,
):
    started = time.monotonic()
    if hasattr(detector, "semantic_corridor"):
        drivable_mask, lane_mask = detector.predict_masks(frame, captured_at=captured_at)
    else:
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
    display_lane = lane_mask
    semantic_fusion = None
    if hasattr(detector, "semantic_corridor"):
        boundary_result = detector.semantic_corridor(control_drivable, control_lane)
        if getattr(boundary_result,'semantic_exclusion_mask',None) is not None:
            control_lane = boundary_result.semantic_exclusion_mask
            display_lane = control_lane
        yellow_hazard = boundary_result.yellow_hazard
        ego_yellow_ratio = boundary_result.ego_yellow_ratio
        control_drivable = boundary_result.corridor_mask
        display_drivable = (mapper.camera_mask(control_drivable)
                            if mapper is not None else control_drivable)
    elif boundary_tracker is not None:
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
        # Publish the current raw-image safety evidence before adaptive
        # precision selection.  In particular, a yellow-under-ego event must
        # reject an in-flight INT8 result in this same control frame.
        boundary_result.ego_yellow_ratio = ego_yellow_ratio
        boundary_result.yellow_hazard = yellow_hazard
        if hasattr(detector, "fuse_corridor"):
            if hasattr(detector, "observe_boundary"):
                detector.observe_boundary(boundary_result)
            fused_corridor, semantic_fusion = detector.fuse_corridor(
                boundary_result.corridor_mask,
                control_drivable,
            )
            boundary_result.corridor_mask = fused_corridor
            boundary_result.semantic_fusion_source = semantic_fusion["source"]
            boundary_result.semantic_result_age_seconds = semantic_fusion[
                "result_age_seconds"
            ]
            boundary_result.semantic_overlap_ratio = semantic_fusion[
                "overlap_ratio"
            ]
            boundary_result.semantic_drivable_ratio = semantic_fusion[
                "drivable_ratio"
            ]
            boundary_result.semantic_inference_seconds = semantic_fusion[
                "inference_seconds"
            ]
            boundary_result.semantic_precision = semantic_fusion["precision"]
            boundary_result.semantic_requested_precision = semantic_fusion[
                "requested_precision"
            ]
            boundary_result.confidence *= semantic_fusion["confidence_scale"]
            if not semantic_fusion["motion_allowed"]:
                boundary_result.valid = False
                boundary_result.confidence = 0.0
                boundary_result.reason = (
                    "YOLOPv2 fusion required for motion: "
                    f"{semantic_fusion['source']}"
                )
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
        if hasattr(detector, "semantic_corridor"):
            estimate = estimator.estimate(
                control_drivable, control_lane, route_hint, preserve_exclusions=True,
                semantic_preview_point=boundary_result.semantic_preview_point,
            )
        elif (
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
    control_estimate = estimate
    if getattr(getattr(detector, 'outer_selector', None), 'visual_feedback', False):
        control_estimate = replace(estimate, heading_error=estimate.near_heading_error)
    command = controller.update(control_estimate, dt)
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
    display_frame = frame
    if hasattr(detector, 'semantic_corridor') and hasattr(detector, 'get_consumed_frame'):
        consumed_frame = detector.get_consumed_frame()
        if consumed_frame is not None:
            display_frame = consumed_frame
    def render():
        return render_debug_frame(
            display_frame,
            display_drivable,
            display_lane,
            display_estimate,
            command,
            ((boundary_result.semantic_inference_seconds or 0.0) * 1000
             if hasattr(detector, "semantic_corridor") else inference_time * 1000),
            control_label=("YOLOPv2 -> control" if hasattr(detector, "semantic_corridor") else "LCC"),
            latency_label=(
                "YOLOPv2" if hasattr(detector, "semantic_corridor") else "boundary"
                if boundary_result is None
                else f"boundary/{boundary_result.source}"
            ),
            boundary_source=(
                "missing" if boundary_result is None else boundary_result.source
            ),
            semantic_mask=(
                drivable_mask if hasattr(detector, "fuse_corridor") else None
            ),
            semantic_label=(
                ("YOLOPv2 matched frame; raw road blue; filtered corridor green; "
                 + ("white allowed; yellow/unknown excluded red; "
                    if getattr(boundary_result,'semantic_track_colors',False) else "")
                 +
                 f"age={float(boundary_result.semantic_result_age_seconds or 0):.2f}s")
                if hasattr(detector, "semantic_corridor") else ""
                if semantic_fusion is None
                else (
                    "blue outline=YOLOPv2 green=fused LCC  "
                    f"{semantic_fusion['source']} "
                    f"{semantic_fusion['precision'] or 'none'}->"
                    f"{semantic_fusion['requested_precision']} "
                    f"overlap={float(semantic_fusion['overlap_ratio'] or 0.0):.2f} "
                    f"age={float(semantic_fusion['result_age_seconds'] or 0.0):.2f}s"
                )
            ),
        )
    if deferred_render is None:
        annotated = render()
    else:
        deferred_render(render)
        annotated = None
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
    historical = boundary_result.source in {"history", "visible-history"}
    for curve, inferred in (
        (
            boundary_result.left_curve,
            historical or boundary_result.source == "inner+width",
        ),
        (
            boundary_result.right_curve,
            historical or boundary_result.source == "outer+width",
        ),
    ):
        if curve.size == height:
            points = np.column_stack(
                [np.clip(curve, 0, width - 1), np.arange(height)]
            ).astype(np.int32)
            if inferred:
                for start in range(0, height - 1, 12):
                    end = min(height - 1, start + 6)
                    cv2.line(
                        output,
                        tuple(map(int, points[start])),
                        tuple(map(int, points[end])),
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                cv2.polylines(output, [points], False, (255, 160, 0), 2)
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
    cv2.putText(
        output,
        f"{boundary_result.source}: cyan=measured magenta=inferred",
        (5, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


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
        cloud_client = None
        if cloud_settings.enabled:
            if not archive.enabled:
                raise ValueError("cloud arbitration requires run evidence archival")
            from cloud_client.client import CloudClient
            cloud_client = CloudClient(timeout=cloud_settings.request_timeout_seconds)
            if getattr(cloud_client.config, "contract", "road-scene-v1") != "road-scene-v1":
                raise ValueError("vehicle arbitration requires CAR_CLOUD_CONTRACT=road-scene-v1")
            cloud_client._require_key()
        cloud_coordinator = CloudCoordinator(
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

        fieldnames = [
            *STATIONARY_LOG_FIELDS,
            *ENCODER_LOG_FIELDS,
            *APPLICATION_LOG_FIELDS,
            "vehicle_profile",
            "vehicle_backend",
            "sample",
            "timestamp_s",
            "frame_age_s",
            "capture_interval_s",
            "loop_start_interval_s",
            "source_wait_ms",
            "analysis_ms",
            "control_render_ms",
            "control_gate_ms",
            "hardware_apply_ms",
            "birdeye_render_ms",
            "inference_ms",
            "semantic_inference_ms",
            "semantic_result_age_s",
            "semantic_result_age_exact_s",
            "semantic_track_colors",
            "semantic_allowed_white_pixels",
            "semantic_yellow_pixels",
            "semantic_sequence",
            "semantic_preview_x",
            "semantic_preview_y",
            "semantic_stability_mode",
            "semantic_filter_changed_ratio",
            "semantic_fusion_source",
            "semantic_precision",
            "semantic_requested_precision",
            "semantic_overlap_ratio",
            "semantic_drivable_ratio",
            "boundary_ms",
            "boundary_source",
            "boundary_confidence",
            "boundary_left_rows",
            "boundary_right_rows",
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
            "corner_continuation_best_lateral",
            "corner_apex_active",
            "corner_apex_age_s",
            "corner_apex_trigger_reason",
            "corner_apex_completion_reason",
            "corner_apex_exit_valid_count",
            "corner_apex_both_valid_count",
            "motion_gate_ready",
            "motion_gate_valid_frames",
            "cloud_phase",
            "cloud_reason",
            "cloud_event_id",
            "archive_queue_depth",
            "archive_dropped_frames",
            "log_queue_depth",
            "log_dropped_rows",
            "live_dropped_updates",
            "debug_dropped_frames",
        ]
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
                    external_motion_allowed=bool(recording_ready and not watchdog.get_state()['tripped']
                        and cloud_coordinator.get_state()['motion_allowed']))
                if cloud_settings.enabled:
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
                active_pivot = stationary_runtime.get_state()['controller']['state'] == 'pivot-right'
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
            command,wheel_state=apply_runtime_command(driver,command,stationary_runtime)
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
            row = {
                **{field: stationary_state.get(field) for field in STATIONARY_LOG_FIELDS},
                'stationary_application_monotonic': application_now,
                'stationary_application_reason': application_reason,
                'semantic_application_age_s': semantic_application_age,
                'frame_application_age_s': frame_application_age,
                'imu_application_age_s': imu_application_age,
                **{'encoder_'+str(index+1): value for index, value in enumerate(encoder_ticks)},
                'encoder_received_monotonic': encoder_sample.get('received_monotonic'),
                'encoder_age_s': encoder_sample.get('age_s'),
                'encoder_valid': encoder_sample.get('valid') is True,
                'encoder_stale': encoder_sample.get('stale') is not False,
                "semantic_sequence": getattr(boundary_result, 'semantic_sequence', None),
                "semantic_preview_x": (boundary_result.semantic_preview_point[0]
                    if boundary_result is not None and boundary_result.semantic_preview_point is not None else None),
                "semantic_preview_y": (boundary_result.semantic_preview_point[1]
                    if boundary_result is not None and boundary_result.semantic_preview_point is not None else None),
                "semantic_result_age_exact_s": getattr(boundary_result, 'semantic_result_age_seconds', None),
                "semantic_track_colors": getattr(boundary_result,'semantic_track_colors',False),
                "semantic_allowed_white_pixels": getattr(boundary_result,'semantic_allowed_white_pixels',0),
                "semantic_yellow_pixels": getattr(boundary_result,'semantic_yellow_pixels',0),
                "semantic_stability_mode": (detector.stabilizer.state['mode']
                    if hasattr(detector, 'stabilizer') else None),
                "semantic_filter_changed_ratio": (detector.stabilizer.state['changed_ratio']
                    if hasattr(detector, 'stabilizer') else None),
                "vehicle_profile": vehicle_summary["profile"],
                "vehicle_backend": vehicle_summary["backend"],
                "sample": sample,
                "timestamp_s": round(float(timestamp), 4),
                "frame_age_s": (
                    None if frame_age is None else round(float(frame_age), 4)
                ),
                "capture_interval_s": (
                    None if dt is None else round(float(dt), 4)
                ),
                "loop_start_interval_s": (
                    None
                    if loop_start_interval is None
                    else round(float(loop_start_interval), 4)
                ),
                "source_wait_ms": round(source_wait_time * 1000, 3),
                "analysis_ms": round(analysis_time * 1000, 3),
                "control_render_ms": round(control_render_time * 1000, 3),
                "control_gate_ms": round(gate_time * 1000, 3),
                "hardware_apply_ms": round(hardware_apply_time * 1000, 3),
                "birdeye_render_ms": round(
                    birdeye_render_time * 1000, 3
                ),
                "inference_ms": round(inference_time * 1000, 3),
                "semantic_inference_ms": (
                    None
                    if boundary_result is None
                    or boundary_result.semantic_inference_seconds is None
                    else round(
                        boundary_result.semantic_inference_seconds * 1000,
                        3,
                    )
                ),
                "semantic_result_age_s": (
                    None
                    if boundary_result is None
                    or boundary_result.semantic_result_age_seconds is None
                    else round(
                        boundary_result.semantic_result_age_seconds,
                        4,
                    )
                ),
                "semantic_fusion_source": (
                    None
                    if boundary_result is None
                    else boundary_result.semantic_fusion_source
                ),
                "semantic_precision": (
                    None
                    if boundary_result is None
                    else boundary_result.semantic_precision
                ),
                "semantic_requested_precision": (
                    None
                    if boundary_result is None
                    else boundary_result.semantic_requested_precision
                ),
                "semantic_overlap_ratio": (
                    None
                    if boundary_result is None
                    or boundary_result.semantic_overlap_ratio is None
                    else round(boundary_result.semantic_overlap_ratio, 5)
                ),
                "semantic_drivable_ratio": (
                    None
                    if boundary_result is None
                    or boundary_result.semantic_drivable_ratio is None
                    else round(boundary_result.semantic_drivable_ratio, 5)
                ),
                "boundary_ms": round(boundary_time * 1000, 3),
                "boundary_source": (
                    None if boundary_result is None else boundary_result.source
                ),
                "boundary_confidence": (
                    None
                    if boundary_result is None
                    else round(boundary_result.confidence, 5)
                ),
                "boundary_left_rows": (
                    None
                    if boundary_result is None
                    else boundary_result.observed_left_rows
                ),
                "boundary_right_rows": (
                    None
                    if boundary_result is None
                    else boundary_result.observed_right_rows
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
                "corner_continuation_best_lateral": (
                    None
                    if corner_state["best_lateral_magnitude"] is None
                    else round(corner_state["best_lateral_magnitude"], 5)
                ),
                "corner_apex_active": corner_state["apex_active"],
                "corner_apex_age_s": (
                    None
                    if corner_state["apex_age_seconds"] is None
                    else round(corner_state["apex_age_seconds"], 5)
                ),
                "corner_apex_trigger_reason": corner_state[
                    "apex_trigger_reason"
                ],
                "corner_apex_completion_reason": corner_state[
                    "apex_completion_reason"
                ],
                "corner_apex_exit_valid_count": corner_state[
                    "apex_exit_valid_count"
                ],
                "corner_apex_both_valid_count": corner_state[
                    "apex_both_valid_count"
                ],
                "motion_gate_ready": motion_state["ready"],
                "motion_gate_valid_frames": motion_state["consecutive_valid"],
                "cloud_phase": cloud_coordinator.get_state()["phase"],
                "cloud_reason": cloud_coordinator.get_state()["reason"],
                "cloud_event_id": cloud_coordinator.get_state()["event_id"],
                "archive_queue_depth": archive_state["queue_depth"],
                "archive_dropped_frames": archive_state["dropped_frames"],
                "log_queue_depth": log_state["queue_depth"],
                "log_dropped_rows": log_state["dropped_rows"],
                "live_dropped_updates": live_state["dropped_updates"],
                "debug_dropped_frames": debug_state["dropped_frames"],
            }
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
                live_status = {
                    "vehicle": vehicle_summary,
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
                    "motion_gate": motion_state,
                    "cloud_arbitration": cloud_coordinator.get_state(),
                    "stationary_corner": None if stationary_runtime is None else stationary_state,
                    'sensor_recording':dict(local=None if sensor_writer is None else sensor_writer.get_state(),
                        ros=sensor_recording_state,required=runtime_config.get('require_sensor_recording',False),
                        serial_publisher=None if telemetry_publisher is None else telemetry_publisher.get_state()),
                    "run_archive": (
                        None if not archive.enabled else str(archive.run_dir)
                    ),
                    "diagnostics": {
                        "archive": archive.get_state(),
                        "control_log": log_writer.get_state(),
                        "live_publisher": live_publisher.get_state(),
                        "debug_frames": debug_writer.get_state(),
                        "yolopv2_fusion": (
                            detector.get_state()
                            if hasattr(detector, "get_state")
                            else {"enabled": False}
                        ),
                    },
                }
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
