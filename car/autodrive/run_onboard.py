#!/usr/bin/env python3
"""Safe onboard YOLOPv2-LCC runner; dry-run unless motors are explicitly enabled."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml


AUTODRIVE_DIR = Path(__file__).resolve().parent
CAR_DIR = AUTODRIVE_DIR.parent
CONTROL_DIR = CAR_DIR / "control"
LONGTAIL_DIR = CAR_DIR / "longtail"
REPO_ROOT = CAR_DIR.parent
for path in (CAR_DIR, CONTROL_DIR, LONGTAIL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autodrive.drive_runtime import (
    CommandWatchdog,
    SafeWheelDriver,
    WheelMappingConfig,
)
from autodrive.lane_centering import (
    DifferentialDriveCommand,
    LCCConfig,
    LaneCenteringController,
    RoadCenterlineEstimator,
)
from autodrive.perspective import PerspectiveMapper
from autodrive.temporal import TemporalMaskPropagator
from autodrive.visualization import render_debug_frame
from detectors.yolopv2_detector import YOLOPv2Detector


DEFAULT_CONFIG = AUTODRIVE_DIR / "onboard_runtime.example.yaml"
MOTOR_CONFIRMATION = "I_UNDERSTAND_MOTORS_WILL_MOVE"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run onboard YOLOPv2 lane centering (dry-run by default)"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="Use a recorded onboard video instead of a camera")
    source.add_argument("--camera-index", type=int, help="Override configured camera index")
    parser.add_argument("--sample-every", type=int, help="Video frame sampling interval")
    parser.add_argument("--max-samples", type=int, default=0, help="0 runs until EOF/interrupt")
    temporal = parser.add_mutually_exclusive_group()
    temporal.add_argument(
        "--temporal",
        dest="temporal",
        action="store_true",
        default=None,
        help="Enable optical-flow mask propagation between YOLOPv2 keyframes",
    )
    temporal.add_argument(
        "--no-temporal",
        dest="temporal",
        action="store_false",
        help="Disable optical-flow mask propagation",
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


class VideoSource:
    def __init__(self, path, sample_every):
        self.path = Path(path).expanduser().resolve()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open video: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.sample_every = max(1, int(sample_every))
        self.frame_index = -1

    def read(self):
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return None, None, None
            self.frame_index += 1
            if self.frame_index % self.sample_every == 0:
                timestamp = self.frame_index / self.fps
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
                return frame, captured_at - self.started_at, age
            time.sleep(0.05)
        return None, None, None

    def close(self):
        self.camera.stop()


def build_components(config, motors_enabled, calibration_path):
    model = config.get("model", {})
    weights = repo_path(model.get("weights"))
    if weights is None or not weights.exists():
        raise FileNotFoundError(f"YOLOPv2 weights not found: {weights}")
    detector = YOLOPv2Detector(
        {
            "weights_path": str(weights),
            "device": str(model.get("device", "cpu")),
            "img_size": int(model.get("img_size", 320)),
            "use_full_model": True,
            "fast_mask": True,
        }
    )
    estimator = RoadCenterlineEstimator(**config.get("centerline", {}))
    controller = LaneCenteringController(LCCConfig(**config.get("lcc", {})))
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
    return detector, estimator, controller, mapper, driver, watchdog


def analyze(
    detector,
    estimator,
    controller,
    mapper,
    frame,
    route_hint,
    dt,
    maximum_inference_time,
    temporal_propagator=None,
    force_keyframe=True,
):
    started = time.monotonic()
    if (
        temporal_propagator is not None
        and not force_keyframe
        and not temporal_propagator.needs_keyframe
    ):
        drivable_mask, lane_mask, temporal_confidence = (
            temporal_propagator.propagate(frame)
        )
        perception_source = "optical-flow"
    else:
        drivable_mask, lane_mask = detector.predict_masks(frame)
        temporal_confidence = 1.0
        perception_source = "yolopv2"
        if temporal_propagator is not None:
            temporal_propagator.reset(frame, drivable_mask, lane_mask)
    inference_time = time.monotonic() - started
    control_drivable = (
        mapper.warp_mask(drivable_mask) if mapper is not None else drivable_mask
    )
    control_lane = mapper.warp_mask(lane_mask) if mapper is not None else lane_mask
    estimate = estimator.estimate(control_drivable, control_lane, route_hint)
    estimate.confidence = float(
        np.clip(estimate.confidence * temporal_confidence, 0.0, 1.0)
    )
    command = controller.update(estimate, dt)
    if inference_time > maximum_inference_time:
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
        drivable_mask,
        lane_mask,
        display_estimate,
        command,
        inference_time * 1000,
        show_control=True,
        latency_label=perception_source,
    )
    return (
        estimate,
        command,
        annotated,
        inference_time,
        perception_source,
        temporal_confidence,
    )


def write_status(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main():
    args = build_parser().parse_args()
    config_path, config = load_config(args.config)
    if args.max_samples < 0:
        raise ValueError("--max-samples must not be negative")

    calibration_path = repo_path(
        config.get("perspective", {}).get("calibration")
    )
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
    )

    camera_config = dict(config.get("camera", {}))
    if args.camera_index is not None:
        camera_config["index"] = args.camera_index
    sample_every = (
        args.sample_every
        if args.sample_every is not None
        else int(config.get("runtime", {}).get("video_sample_every", 6))
    )
    if sample_every < 1:
        raise ValueError("sample interval must be at least 1")

    output_dir = repo_path(
        config.get("runtime", {}).get("output_dir", "outputs/onboard_runtime")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "onboard_log.csv"
    status_path = output_dir / "status.json"
    latest_frame_path = output_dir / "latest.jpg"

    source = (
        VideoSource(args.video, sample_every)
        if args.video
        else CameraSource(camera_config)
    )
    detector = estimator = controller = mapper = driver = watchdog = None
    log_handle = None
    try:
        detector, estimator, controller, mapper, driver, watchdog = build_components(
            config, args.enable_motors, calibration_path
        )
        temporal_config = config.get("temporal", {})
        temporal_propagator = None
        keyframe_interval = int(temporal_config.get("keyframe_interval", 4))
        if keyframe_interval < 1:
            raise ValueError("temporal.keyframe_interval must be at least 1")
        temporal_enabled = (
            bool(temporal_config.get("enabled", False))
            if args.temporal is None
            else bool(args.temporal)
        )
        if temporal_enabled:
            temporal_propagator = TemporalMaskPropagator(
                max_steps=int(
                    temporal_config.get(
                        "max_steps", max(1, keyframe_interval - 1)
                    )
                ),
                confidence_decay=float(
                    temporal_config.get("confidence_decay", 0.90)
                ),
            )
        frame, timestamp, frame_age = source.read()
        if frame is None:
            raise RuntimeError("no fresh startup frame")

        print("Warming up YOLOPv2; wheels remain stopped ...", flush=True)
        analyze(
            detector,
            estimator,
            controller,
            mapper,
            frame,
            str(config.get("runtime", {}).get("route_hint", "center")),
            None,
            maximum_inference_time,
            temporal_propagator,
            True,
        )
        controller.reset()
        driver.stop("warmup complete; waiting for fresh frame")
        watchdog.arm()

        fieldnames = [
            "sample",
            "timestamp_s",
            "frame_age_s",
            "inference_ms",
            "perception_source",
            "temporal_confidence",
            "valid",
            "confidence",
            "lateral_error",
            "heading_error",
            "action",
            "steering",
            "left_speed",
            "right_speed",
            "left_pwm",
            "right_pwm",
            "reason",
        ]
        log_handle = log_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
        writer.writeheader()

        previous_timestamp = None
        sample = 0
        route_hint = str(config.get("runtime", {}).get("route_hint", "center"))
        max_inference = maximum_inference_time
        save_latest = bool(
            config.get("runtime", {}).get("save_latest_frame", True)
        )
        print(
            f"Onboard runtime started: mode={'HARDWARE' if args.enable_motors else 'DRY-RUN'}, "
            f"source={args.video or 'camera'}, calibration={calibration_path or 'none'}",
            flush=True,
        )

        while True:
            frame, timestamp, frame_age = source.read()
            if frame is None:
                break
            dt = (
                None
                if previous_timestamp is None
                else max(1e-3, float(timestamp - previous_timestamp))
            )
            previous_timestamp = timestamp
            force_keyframe = (
                temporal_propagator is None
                or sample % keyframe_interval == 0
                or temporal_propagator.needs_keyframe
            )
            (
                estimate,
                command,
                annotated,
                inference_time,
                perception_source,
                temporal_confidence,
            ) = analyze(
                detector,
                estimator,
                controller,
                mapper,
                frame,
                route_hint,
                dt,
                max_inference,
                temporal_propagator,
                force_keyframe,
            )
            watchdog.heartbeat()
            wheel_state = driver.apply(command)
            if save_latest:
                cv2.imwrite(str(latest_frame_path), annotated)

            row = {
                "sample": sample,
                "timestamp_s": round(float(timestamp), 4),
                "frame_age_s": (
                    None if frame_age is None else round(float(frame_age), 4)
                ),
                "inference_ms": round(inference_time * 1000, 3),
                "perception_source": perception_source,
                "temporal_confidence": round(temporal_confidence, 5),
                "valid": estimate.valid,
                "confidence": round(estimate.confidence, 5),
                "lateral_error": round(estimate.lateral_error, 5),
                "heading_error": round(estimate.heading_error, 5),
                "action": command.action,
                "steering": round(command.steering, 5),
                "left_speed": round(command.left_speed, 5),
                "right_speed": round(command.right_speed, 5),
                "left_pwm": wheel_state["left_pwm"],
                "right_pwm": wheel_state["right_pwm"],
                "reason": command.reason,
            }
            writer.writerow(row)
            log_handle.flush()
            write_status(
                status_path,
                {
                    "mode": wheel_state["mode"],
                    "source": str(args.video or f"camera:{camera_config.get('index', 0)}"),
                    "config": str(config_path),
                    "calibration": (
                        str(calibration_path) if calibration_path else None
                    ),
                    "last_result": row,
                    "wheel_driver": driver.get_state(),
                    "watchdog": watchdog.get_state(),
                },
            )
            print(
                f"sample={sample} action={command.action} "
                f"source={perception_source} "
                f"conf={estimate.confidence:.2f} steer={command.steering:+.3f} "
                f"pwm=({wheel_state['left_pwm']:+d},{wheel_state['right_pwm']:+d}) "
                f"inference={inference_time * 1000:.0f}ms",
                flush=True,
            )
            sample += 1
            if args.max_samples and sample >= args.max_samples:
                break
    except KeyboardInterrupt:
        print("Interrupted; stopping.", flush=True)
    finally:
        if watchdog is not None:
            watchdog.close()
        elif driver is not None:
            driver.stop("runtime shutdown")
        source.close()
        if log_handle is not None:
            log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
