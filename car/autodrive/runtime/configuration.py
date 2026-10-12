"""CLI and runtime configuration validation without device access."""
from __future__ import annotations

from autodrive.control.cloud_arbitration import CloudArbitrationConfig
from autodrive.control.cloud_recovery import RecoveryConfig
from autodrive.perception.perspective import validate_calibration_camera_pose
from autodrive.runtime.paths import DEFAULT_CONFIG
from autodrive.runtime.paths import MOTOR_CONFIRMATION
from autodrive.runtime.paths import REPO_ROOT
from pathlib import Path
from vehicle_control.profile import load_runtime_config
import argparse
import math


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


def cloud_configuration(config):
    settings = CloudArbitrationConfig(**config.get("cloud_arbitration", {}))
    recovery = RecoveryConfig(**config.get('cloud_recovery', {}))
    if recovery.enabled:
        if settings.enabled:
            raise ValueError('choose cloud_recovery or legacy cloud_arbitration')
        stationary=config.get('stationary_corner',{})
        wheels=config.get('wheels',{})
        if (config.get('perception',{}).get('mode')!='yolopv2'
                or not config.get('perception',{}).get('track_colors',{}).get('enabled')
                or not stationary.get('enabled') or not stationary.get('visual_feedback')
                or wheels.get('pwm_limit',100)>30 or wheels.get('transition_time',0)!=0
                or recovery.yaw_sign!=stationary.get('yaw_sign',-1)):
            raise ValueError('recovery requires primary track-color YOLO, visual feedback, PWM<=30 and no ramp')
        return recovery
    if settings.enabled and config.get("perception", {}).get("mode") != "yolopv2":
        raise ValueError("cloud arbitration requires primary YOLOPv2 observations")
    return settings

