"""Compatibility exports for the vehicle hardware adapter layer."""

from .base import FourWheelChassis, ServoGimbal, clamp
from .factory import (
    HardwareNotCalibratedError,
    create_chassis,
    create_gimbal,
    emergency_stop,
    validate_motion_configuration,
)
from .platforms.raspbot import Raspbot, RaspbotChassis, RospbotChassis
from .platforms.rosmaster import RosmasterChassis


__all__ = [
    "FourWheelChassis",
    "ServoGimbal",
    "HardwareNotCalibratedError",
    "Raspbot",
    "RaspbotChassis",
    "RospbotChassis",
    "RosmasterChassis",
    "clamp",
    "create_chassis",
    "create_gimbal",
    "emergency_stop",
    "validate_motion_configuration",
]
