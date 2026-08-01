#!/usr/bin/env python3
"""Capture a still image or short video from the fixed onboard camera."""

import argparse
from pathlib import Path
import time

import cv2
import yaml

from camera_gimbal import initialize_configured_gimbal
from camera_transform import CameraTransformConfig, transform_frame


REPO_ROOT = Path(__file__).resolve().parents[2]
GIMBAL_CONFIRMATION = "CAMERA_GIMBAL_IS_CLEAR"


def build_parser():
    parser = argparse.ArgumentParser(description="Capture onboard-camera calibration data")
    parser.add_argument(
        "--config",
        help=(
            "Optional runtime YAML; its camera geometry and configured gimbal "
            "pose override the individual camera options"
        ),
    )
    parser.add_argument(
        "--confirm-camera-gimbal-clear",
        default="",
        help=(
            "Required when --config initializes the gimbal: "
            f"{GIMBAL_CONFIRMATION}"
        ),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--rotation-degrees",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Correct sensor roll; this does not correct a left/right-facing lens",
    )
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "onboard_capture"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    camera_config = {
        "index": args.camera_index,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "rotation_degrees": args.rotation_degrees,
        "flip_horizontal": args.flip_horizontal,
        "flip_vertical": args.flip_vertical,
    }
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        runtime_config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        ) or {}
        if runtime_config.get("version") != 1:
            raise ValueError("runtime config version must be 1")
        camera_config = dict(runtime_config.get("camera", {}))
        gimbal = camera_config.get("gimbal") or {}
        if bool(gimbal.get("initialize_on_startup", False)):
            if args.confirm_camera_gimbal_clear != GIMBAL_CONFIRMATION:
                raise ValueError(
                    "configured gimbal initialization requires "
                    "--confirm-camera-gimbal-clear " + GIMBAL_CONFIRMATION
                )
            commands = initialize_configured_gimbal(camera_config)
            print(f"Initialized camera gimbal: {commands}", flush=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "onboard_calibration.mp4"
    still_path = output_dir / "onboard_calibration_frame.jpg"
    frame_transform = CameraTransformConfig(
        rotation_degrees=int(camera_config.get("rotation_degrees", 0)),
        flip_horizontal=bool(camera_config.get("flip_horizontal", False)),
        flip_vertical=bool(camera_config.get("flip_vertical", False)),
    )

    camera_index = int(camera_config.get("index", 0))
    width = int(camera_config.get("width", 640))
    height = int(camera_config.get("height", 480))
    fps = int(camera_config.get("fps", 20))
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera index {camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    writer = None
    frames = 0
    last_frame = None
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = transform_frame(frame, frame_transform)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"unable to create video: {video_path}")
            writer.write(frame)
            last_frame = frame
            frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if last_frame is None:
        raise RuntimeError("camera opened but returned no frames")
    if not cv2.imwrite(str(still_path), last_frame):
        raise RuntimeError(f"unable to write still image: {still_path}")
    print(f"Captured {frames} frames")
    print(f"Video: {video_path}")
    print(f"Calibration frame: {still_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
