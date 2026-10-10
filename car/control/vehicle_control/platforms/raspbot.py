"""Raspbot Pi 5 I2C hardware backend."""

from pathlib import Path
import sys

from ..base import FourWheelChassis, ServoGimbal, validate_four_values


CONTROL_DIR = Path(__file__).resolve().parents[2]
UTILS_DIR = CONTROL_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from Raspbot_Lib import Raspbot


class RaspbotChassis(FourWheelChassis):
    backend = "raspbot"

    def __init__(
        self,
        controller=None,
        command_limit=255,
        motor_ids=(0, 1, 2, 3),
        wheel_signs=(1, 1, 1, 1),
    ):
        self.controller = controller if controller is not None else Raspbot()
        self.motor_ids = tuple(
            int(value) for value in validate_four_values("motor_ids", motor_ids)
        )
        if len(set(self.motor_ids)) != 4:
            raise ValueError("motor_ids must contain four unique IDs")
        super().__init__(command_limit=command_limit, wheel_signs=wheel_signs)

    def _write_native(self, signed_logical_commands):
        for motor_id, value in zip(self.motor_ids, signed_logical_commands):
            self.controller.Ctrl_Muto(motor_id, value)

    def _close_native(self):
        close = getattr(self.controller, "close", None)
        if close is not None:
            close()


# Preserve the historical misspelled public name used by existing scripts.
RospbotChassis = RaspbotChassis


class RaspbotGimbal(ServoGimbal):
    backend = "raspbot"

    def __init__(self, controller=None, **kwargs):
        self.controller = controller if controller is not None else Raspbot()
        super().__init__(**kwargs)

    def _write_servo(self, servo_id, angle):
        self.controller.Ctrl_Servo(servo_id, angle)

    def _close_native(self):
        close = getattr(self.controller, "close", None)
        if close is not None:
            close()
