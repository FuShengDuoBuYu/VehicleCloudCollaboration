#!/usr/bin/env python3
"""Capture a still image or short video from the fixed onboard camera."""

import argparse
from pathlib import Path
import time

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description="Capture onboard-camera calibration data")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
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
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "onboard_calibration.mp4"
    still_path = output_dir / "onboard_calibration_frame.jpg"

    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera index {args.camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)

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
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps),
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
