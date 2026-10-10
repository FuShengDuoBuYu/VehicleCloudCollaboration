#!/usr/bin/env python3
"""Move the selected vehicle camera PTZ with an explicit confirmation."""

import argparse
from pathlib import Path
import sys
import time

import cv2


AUTODRIVE_DIR = Path(__file__).resolve().parents[1]
CAR_DIR = AUTODRIVE_DIR.parent
CONTROL_DIR = CAR_DIR / "control"
REPO_ROOT = CAR_DIR.parent
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.camera.gimbal import CameraGimbalPose
from autodrive.camera.transform import CameraTransformConfig, transform_frame
from vehicle_control.factory import create_gimbal
from vehicle_control.profile import DEFAULT_PROFILE, load_vehicle_profile


CONFIRMATION = "CAMERA_GIMBAL_IS_CLEAR"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Move the camera pan/tilt servos and save a verification frame"
    )
    parser.add_argument(
        "--vehicle-profile",
        default=DEFAULT_PROFILE,
        help="Vehicle profile ID or YAML path",
    )
    parser.add_argument(
        "--confirm-camera-gimbal-clear",
        default="",
        help=f"Required exact value: {CONFIRMATION}",
    )
    parser.add_argument(
        "--pan-angle",
        type=int,
        help="Horizontal angle; exact safe range comes from the vehicle profile",
    )
    parser.add_argument(
        "--tilt-angle",
        type=int,
        help="Vertical angle; exact safe range comes from the vehicle profile",
    )
    parser.add_argument("--settle-time", type=float, default=0.8)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--rotation-degrees",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
    )
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument(
        "--preview-output",
        default=str(REPO_ROOT / "outputs" / "onboard_capture" / "gimbal_alignment.jpg"),
    )
    parser.add_argument("--no-preview", action="store_true")
    return parser


def capture_preview(args):
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera index {args.camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    transform = CameraTransformConfig(
        rotation_degrees=args.rotation_degrees,
        flip_horizontal=args.flip_horizontal,
        flip_vertical=args.flip_vertical,
    )
    deadline = time.monotonic() + 3.0
    latest = None
    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if ok:
                latest = transform_frame(frame, transform)
            else:
                time.sleep(0.05)
    finally:
        capture.release()
    if latest is None:
        raise RuntimeError("camera opened but returned no preview frame")
    output = Path(args.preview_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), latest):
        raise RuntimeError(f"unable to save preview: {output}")
    return output


def main():
    args = build_parser().parse_args()
    if args.confirm_camera_gimbal_clear != CONFIRMATION:
        raise ValueError(
            "refusing servo access; clear the PTZ mechanism and pass "
            f"--confirm-camera-gimbal-clear {CONFIRMATION}"
        )
    pose = CameraGimbalPose(
        pan_angle=args.pan_angle,
        tilt_angle=args.tilt_angle,
        settle_time=args.settle_time,
    )

    _, vehicle_config = load_vehicle_profile(args.vehicle_profile)
    # This guarded alignment tool is permitted before a profile is marked as
    # calibrated; the normal runtime remains blocked.
    controller = create_gimbal(
        vehicle_config,
        require_gimbal_calibrated=False,
    )
    try:
        commands = pose.apply(controller)
    finally:
        controller.close()
    print(
        f"Applied {vehicle_config['id']} camera servo commands: {commands}",
        flush=True,
    )
    if not args.no_preview:
        preview = capture_preview(args)
        print(f"Preview: {preview}", flush=True)
    print(
        "This only positions the PTZ. Re-capture and re-calibrate perspective "
        "before enabling wheel motors.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
