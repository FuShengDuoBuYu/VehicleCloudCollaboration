"""Yahboom Rosmaster X3 serial hardware backend."""

from ..base import (FourWheelChassis, ServoGimbal, validate_four_values,
                    checked_integer, checked_number)


def _new_rosmaster(serial_port, car_type, delay, debug):
    from .rosmaster_transport import RosmasterSerialSession

    if checked_integer("car_type", car_type) not in (1, 2, 4, 5):
        raise ValueError("unsupported Rosmaster car_type")
    if type(debug) is not bool:
        raise ValueError("debug must be a boolean")
    return RosmasterSerialSession(serial_port=str(serial_port), delay=delay)


def _close_controller(controller):
    close = getattr(controller, "close", None)
    if close is not None:
        close()
        return
    serial_port = getattr(controller, "ser", None)
    if serial_port is not None and getattr(serial_port, "is_open", True):
        serial_port.close()


class RosmasterChassis(FourWheelChassis):
    backend = "rosmaster"

    def __init__(
        self,
        controller=None,
        serial_port="/dev/myserial",
        car_type=1,
        delay=0.002,
        debug=False,
        command_limit=100,
        motor_order=(0, 1, 2, 3),
        wheel_signs=(1, 1, 1, 1),
        owns_controller=True,
    ):
        self.motor_order = tuple(
            checked_integer("motor_order", value)
            for value in validate_four_values("motor_order", motor_order)
        )
        if sorted(self.motor_order) != [0, 1, 2, 3]:
            raise ValueError("motor_order must be a permutation of [0, 1, 2, 3]")
        super().__init__(command_limit=command_limit, wheel_signs=wheel_signs)
        if self.command_limit > 100:
            raise ValueError("Rosmaster command_limit must be inside [1, 100]")
        if type(owns_controller) is not bool or (not owns_controller and controller is None):
            raise ValueError("borrowed ownership requires an injected controller")
        self._owns_controller = owns_controller
        self.controller = controller if controller is not None else _new_rosmaster(serial_port, car_type, delay, debug)

    def _write_native(self, signed_logical_commands):
        native = tuple(
            signed_logical_commands[index] for index in self.motor_order
        )
        self.controller.set_motor(*native)

    def _close_native(self):
        if self._owns_controller:
            _close_controller(self.controller)


class RosmasterGimbal(ServoGimbal):
    backend = "rosmaster"

    def __init__(
        self,
        controller=None,
        serial_port="/dev/myserial",
        car_type=1,
        delay=0.002,
        debug=False,
        owns_controller=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.pan_servo_id > 4 or self.tilt_servo_id > 4:
            raise ValueError("Rosmaster PWM servo IDs must be inside [1, 4]")
        if type(owns_controller) is not bool or (not owns_controller and controller is None):
            raise ValueError("borrowed ownership requires an injected controller")
        self._owns_controller = owns_controller
        self.controller = controller if controller is not None else _new_rosmaster(serial_port, car_type, delay, debug)

    def _write_servo(self, servo_id, angle):
        self.controller.set_pwm_servo(servo_id, angle)

    def _close_native(self):
        if self._owns_controller:
            _close_controller(self.controller)
