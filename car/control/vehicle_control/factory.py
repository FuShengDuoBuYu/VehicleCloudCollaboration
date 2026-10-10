"""Factories and safety checks for configured vehicle hardware."""


class HardwareNotCalibratedError(RuntimeError):
    pass


def _status(vehicle_config, key):
    status = vehicle_config.get("status", {})
    if not isinstance(status, dict):
        raise ValueError("vehicle status must be a mapping")
    value = status.get(key, False)
    if type(value) is not bool:
        raise ValueError(f"vehicle status.{key} must be a boolean")
    return value


def _identity(vehicle_config):
    return str(
        vehicle_config.get("profile")
        or vehicle_config.get("id")
        or vehicle_config.get("backend")
        or "unknown"
    )


def validate_motion_configuration(vehicle_config, wheel_config):
    if not _status(vehicle_config, "motion_calibrated"):
        raise HardwareNotCalibratedError(
            f"vehicle profile {_identity(vehicle_config)!r} is not marked "
            "motion_calibrated; run the lifted-wheel checks and update its profile"
        )
    command_limit = int(
        (vehicle_config.get("chassis") or {}).get("command_limit", 0)
    )
    pwm_limit = int((wheel_config or {}).get("pwm_limit", 0))
    if command_limit < 1:
        raise ValueError("vehicle chassis.command_limit must be positive")
    if not 1 <= pwm_limit <= command_limit:
        raise ValueError(
            f"wheels.pwm_limit={pwm_limit} exceeds the "
            f"{_identity(vehicle_config)} command limit {command_limit}"
        )


def create_chassis(vehicle_config, require_motion_calibrated=True, controller=None):
    backend = str(vehicle_config.get("backend", ""))
    options = dict(vehicle_config.get("chassis") or {})
    if require_motion_calibrated and not _status(
        vehicle_config, "motion_calibrated"
    ):
        raise HardwareNotCalibratedError(
            f"vehicle profile {_identity(vehicle_config)!r} is not marked "
            "motion_calibrated"
        )
    if controller is not None:
        options["controller"] = controller
    if backend == "raspbot":
        from .platforms.raspbot import RaspbotChassis

        allowed = {"controller", "command_limit", "motor_ids", "wheel_signs"}
        kwargs = {key: value for key, value in options.items() if key in allowed}
        return RaspbotChassis(**kwargs)
    if backend == "rosmaster":
        from .platforms.rosmaster import RosmasterChassis

        allowed = {
            "controller",
            "serial_port",
            "car_type",
            "delay",
            "debug",
            "command_limit",
            "motor_order",
            "wheel_signs",
            "owns_controller",
        }
        kwargs = {key: value for key, value in options.items() if key in allowed}
        return RosmasterChassis(**kwargs)
    raise ValueError(f"unsupported vehicle backend: {backend!r}")


def create_gimbal(vehicle_config, require_gimbal_calibrated=True, controller=None):
    backend = str(vehicle_config.get("backend", ""))
    options = dict(vehicle_config.get("gimbal") or {})
    if require_gimbal_calibrated and not _status(
        vehicle_config, "gimbal_calibrated"
    ):
        raise HardwareNotCalibratedError(
            f"vehicle profile {_identity(vehicle_config)!r} is not marked "
            "gimbal_calibrated"
        )
    if controller is not None:
        options["controller"] = controller
    shared = {
        "controller",
        "pan_servo_id",
        "tilt_servo_id",
        "pan_limits",
        "tilt_limits",
    }
    if backend == "raspbot":
        from .platforms.raspbot import RaspbotGimbal

        kwargs = {key: value for key, value in options.items() if key in shared}
        return RaspbotGimbal(**kwargs)
    if backend == "rosmaster":
        from .platforms.rosmaster import RosmasterGimbal

        allowed = shared | {"serial_port", "car_type", "delay", "debug", "owns_controller"}
        kwargs = {key: value for key, value in options.items() if key in allowed}
        return RosmasterGimbal(**kwargs)
    raise ValueError(f"unsupported vehicle backend: {backend!r}")


def emergency_stop(vehicle_config):
    """Write a backend-native all-zero command even for an uncalibrated profile."""
    if vehicle_config.get("backend") == "rosmaster":
        from .platforms.rosmaster import RosmasterChassis

        # An all-zero native frame needs no wheel mapping, signs or calibration.
        # Preserve only the device address; invalid addresses still fail loudly.
        serial_port = (vehicle_config.get("chassis") or {}).get("serial_port", "/dev/myserial")
        chassis = RosmasterChassis(serial_port=serial_port)
    else:
        chassis = create_chassis(vehicle_config, require_motion_calibrated=False)
    chassis.close()
