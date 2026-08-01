"""Guarded camera-gimbal pose helpers for the Raspbot S1/S2 servos."""

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Optional


CAR_DIR = Path(__file__).resolve().parents[1]
CONTROL_UTILS_DIR = CAR_DIR / "control" / "utils"
if str(CONTROL_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_UTILS_DIR))


PAN_SERVO_ID = 1
TILT_SERVO_ID = 2
PAN_LIMITS = (0, 180)
# The bundled Raspbot driver documents a chassis-specific 100-degree ceiling
# for servo 2, so keep that mechanical protection at this boundary too.
TILT_LIMITS = (0, 100)


@dataclass(frozen=True)
class CameraGimbalPose:
    pan_angle: Optional[int] = None
    tilt_angle: Optional[int] = None
    settle_time: float = 0.8

    def __post_init__(self):
        if self.pan_angle is None and self.tilt_angle is None:
            raise ValueError("at least one of pan_angle or tilt_angle is required")
        if self.pan_angle is not None and not (
            PAN_LIMITS[0] <= self.pan_angle <= PAN_LIMITS[1]
        ):
            raise ValueError("pan_angle must be in [0, 180]")
        if self.tilt_angle is not None and not (
            TILT_LIMITS[0] <= self.tilt_angle <= TILT_LIMITS[1]
        ):
            raise ValueError("tilt_angle must be in [0, 100]")
        if self.settle_time < 0:
            raise ValueError("settle_time must not be negative")

    def apply(self, controller, sleep=time.sleep):
        commands = []
        if self.pan_angle is not None:
            controller.Ctrl_Servo(PAN_SERVO_ID, int(self.pan_angle))
            commands.append((PAN_SERVO_ID, int(self.pan_angle)))
        if self.tilt_angle is not None:
            controller.Ctrl_Servo(TILT_SERVO_ID, int(self.tilt_angle))
            commands.append((TILT_SERVO_ID, int(self.tilt_angle)))
        if self.settle_time:
            sleep(self.settle_time)
        return tuple(commands)


def startup_pose_from_mapping(camera_config):
    """Build the optional startup pose from a runtime camera configuration."""
    gimbal = (camera_config or {}).get("gimbal") or {}
    if not bool(gimbal.get("initialize_on_startup", False)):
        return None
    return CameraGimbalPose(
        pan_angle=(
            None if gimbal.get("pan_angle") is None else int(gimbal["pan_angle"])
        ),
        tilt_angle=(
            None if gimbal.get("tilt_angle") is None else int(gimbal["tilt_angle"])
        ),
        settle_time=float(gimbal.get("settle_time", 0.8)),
    )


def initialize_configured_gimbal(
    camera_config,
    controller_factory=None,
    sleep=time.sleep,
):
    """Apply a configured startup pose and close the temporary I2C handle."""
    pose = startup_pose_from_mapping(camera_config)
    if pose is None:
        return ()
    if controller_factory is None:
        from Raspbot_Lib import Raspbot

        controller_factory = Raspbot
    controller = controller_factory()
    try:
        return pose.apply(controller, sleep=sleep)
    finally:
        close = getattr(controller, "close", None)
        if close is not None:
            close()
