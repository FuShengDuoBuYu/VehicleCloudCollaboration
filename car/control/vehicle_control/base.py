"""Common contracts and guarded mechanics for vehicle hardware backends."""

from abc import ABC, abstractmethod
import math
from numbers import Integral, Real
import threading
import time


WHEEL_NAMES = ("front-left", "rear-left", "front-right", "rear-right")


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def validate_four_values(name, values):
    values = tuple(values)
    if len(values) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    return values


def checked_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def checked_number(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


class FourWheelChassis(ABC):
    """Logical four-wheel chassis ordered FL, RL, FR, RR.

    Commands use each backend's native signed unit. The LCC produces a
    normalized steering command first; its vehicle-specific wheel mapping is
    responsible for converting that command into these native units.
    """

    backend = "unknown"

    def __init__(self, command_limit, wheel_signs=(1, 1, 1, 1)):
        command_limit = checked_integer("command_limit", command_limit)
        if command_limit < 1:
            raise ValueError("command_limit must be positive")
        signs = tuple(
            checked_integer("wheel_signs", value)
            for value in validate_four_values("wheel_signs", wheel_signs)
        )
        if any(value not in (-1, 1) for value in signs):
            raise ValueError("wheel_signs values must be -1 or 1")
        self.command_limit = command_limit
        self.wheel_signs = signs
        self.current_left = 0
        self.current_right = 0
        self.current_front_left = 0
        self.current_rear_left = 0
        self.current_front_right = 0
        self.current_rear_right = 0
        self._lock = threading.Lock()
        self._closed = False
        self._ramp_generation = 0
        self._closing = False

    def _logical_commands(self, values):
        return tuple(
            int(clamp(checked_number("wheel command", value), -self.command_limit, self.command_limit))
            for value in validate_four_values("wheel commands", values)
        )

    @abstractmethod
    def _write_native(self, signed_logical_commands):
        """Write signed logical commands to a platform-specific controller."""

    @abstractmethod
    def _close_native(self):
        """Close the platform controller without issuing a movement command."""

    def set_four_wheels(
        self,
        front_left,
        rear_left,
        front_right,
        rear_right,
    ):
        return self._set_four_wheels((front_left, rear_left, front_right, rear_right))

    def _set_four_wheels(self, values, ramp_generation=None):
        logical = self._logical_commands(values)
        signed = tuple(
            value * sign for value, sign in zip(logical, self.wheel_signs)
        )
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.backend} chassis is closed")
            if self._closing and any(logical):
                raise RuntimeError(f"{self.backend} chassis is closing")
            if ramp_generation is not None and ramp_generation != self._ramp_generation:
                return None
            self._write_native(signed)
            (
                self.current_front_left,
                self.current_rear_left,
                self.current_front_right,
                self.current_rear_right,
            ) = logical
            self.current_left = int(round((logical[0] + logical[1]) * 0.5))
            self.current_right = int(round((logical[2] + logical[3]) * 0.5))
        return logical

    def set_wheels(self, left_speed, right_speed):
        return self.set_four_wheels(
            left_speed,
            left_speed,
            right_speed,
            right_speed,
        )

    def ramp_four_to(
        self,
        front_left_target,
        rear_left_target,
        front_right_target,
        rear_right_target,
        transition_time,
        stop_event=None,
    ):
        targets = self._logical_commands(
            (
                front_left_target,
                rear_left_target,
                front_right_target,
                rear_right_target,
            )
        )
        transition_time = checked_number("transition_time", transition_time)
        if transition_time < 0:
            raise ValueError("transition_time must be nonnegative")
        if transition_time == 0:
            self.set_four_wheels(*targets)
            return

        with self._lock:
            generation = self._ramp_generation
            starts = (
                self.current_front_left,
                self.current_rear_left,
                self.current_front_right,
                self.current_rear_right,
            )

        started = time.monotonic()
        while True:
            if stop_event is not None and stop_event.is_set():
                self.stop()
                return
            elapsed = time.monotonic() - started
            if elapsed >= transition_time:
                break
            ratio = elapsed / transition_time
            current = tuple(
                start + (target - start) * ratio
                for start, target in zip(starts, targets)
            )
            if self._set_four_wheels(current, ramp_generation=generation) is None:
                return
            time.sleep(0.02)
        self._set_four_wheels(targets, ramp_generation=generation)

    def ramp_to(self, left_target, right_target, transition_time, stop_event=None):
        return self.ramp_four_to(
            left_target,
            left_target,
            right_target,
            right_target,
            transition_time,
            stop_event=stop_event,
        )

    def hold(self, duration, stop_event=None):
        duration = checked_number("duration", duration)
        if duration < 0:
            raise ValueError("duration must be nonnegative")
        started = time.monotonic()
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            if time.monotonic() - started >= duration:
                return True
            time.sleep(0.05)

    def stop(self):
        with self._lock:
            self._ramp_generation += 1
        first_error = None
        for attempt in range(3):
            try:
                self.set_four_wheels(0, 0, 0, 0)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            if attempt < 2:
                time.sleep(0.05)
        if first_error is not None:
            raise first_error

    def close(self, stop=True):
        with self._lock:
            if self._closed or self._closing:
                return
            self._closing = True
        try:
            if stop:
                self.stop()
        finally:
            with self._lock:
                if not self._closed:
                    try:
                        self._close_native()
                    finally:
                        self._closed = True


class ServoGimbal(ABC):
    """Two-axis servo adapter with profile-defined channels and limits."""

    backend = "unknown"

    def __init__(
        self,
        pan_servo_id=1,
        tilt_servo_id=2,
        pan_limits=(0, 180),
        tilt_limits=(0, 180),
    ):
        self.pan_servo_id = checked_integer("pan_servo_id", pan_servo_id)
        self.tilt_servo_id = checked_integer("tilt_servo_id", tilt_servo_id)
        if self.pan_servo_id < 1 or self.tilt_servo_id < 1 or self.pan_servo_id == self.tilt_servo_id:
            raise ValueError("gimbal servo IDs must be distinct positive integers")
        self.pan_limits = self._limits("pan_limits", pan_limits)
        self.tilt_limits = self._limits("tilt_limits", tilt_limits)
        self._closed = False

    @staticmethod
    def _limits(name, values):
        values = tuple(checked_integer(name, value) for value in values)
        if len(values) != 2 or not 0 <= values[0] <= values[1] <= 180:
            raise ValueError(f"{name} must be an ordered pair inside [0, 180]")
        return values

    @abstractmethod
    def _write_servo(self, servo_id, angle):
        """Write one bounded servo command."""

    @abstractmethod
    def _close_native(self):
        """Close the platform controller."""

    def _bounded_angle(self, axis, angle, limits):
        angle = checked_number(f"{axis}_angle", angle)
        if not limits[0] <= angle <= limits[1]:
            raise ValueError(
                f"{axis}_angle must be in [{limits[0]}, {limits[1]}]"
            )
        return int(angle)

    def set_pose(self, pan_angle=None, tilt_angle=None):
        if self._closed:
            raise RuntimeError(f"{self.backend} gimbal is closed")
        if pan_angle is None and tilt_angle is None:
            raise ValueError("at least one of pan_angle or tilt_angle is required")
        commands = []
        if pan_angle is not None:
            angle = self._bounded_angle("pan", pan_angle, self.pan_limits)
            commands.append((self.pan_servo_id, angle))
        if tilt_angle is not None:
            angle = self._bounded_angle("tilt", tilt_angle, self.tilt_limits)
            commands.append((self.tilt_servo_id, angle))
        for servo_id, angle in commands:
            self._write_servo(servo_id, angle)
        return tuple(commands)

    def close(self):
        if self._closed:
            return
        self._close_native()
        self._closed = True
