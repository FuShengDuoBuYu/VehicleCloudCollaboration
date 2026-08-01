"""Safety boundary between normalized LCC commands and the physical chassis."""

from dataclasses import asdict, dataclass
import threading
import time
from typing import Optional

import numpy as np

from .lane_centering import DifferentialDriveCommand


@dataclass(frozen=True)
class WheelMappingConfig:
    drive_mode: str = "four-wheel-trim"
    pwm_limit: int = 35
    minimum_moving_pwm: int = 10
    front_left_base_pwm: int = 16
    rear_left_base_pwm: int = 16
    front_right_base_pwm: int = 20
    rear_right_base_pwm: int = 20
    front_steering_delta_pwm: int = 20
    maximum_steering_delta_pwm: int = 10
    transition_time: float = 0.25

    def __post_init__(self):
        if self.drive_mode != "four-wheel-trim":
            raise ValueError("drive_mode must be four-wheel-trim")
        if not 1 <= self.pwm_limit <= 255:
            raise ValueError("pwm_limit must be in [1, 255]")
        if not 0 <= self.minimum_moving_pwm <= self.pwm_limit:
            raise ValueError("minimum_moving_pwm must be in [0, pwm_limit]")
        direct_values = (
            self.front_left_base_pwm,
            self.rear_left_base_pwm,
            self.front_right_base_pwm,
            self.rear_right_base_pwm,
            self.front_steering_delta_pwm,
            self.maximum_steering_delta_pwm,
        )
        if any(value < 0 or value > self.pwm_limit for value in direct_values):
            raise ValueError("per-wheel PWM values must be inside pwm_limit")
        if self.transition_time < 0:
            raise ValueError("transition_time must not be negative")


class SafeWheelDriver:
    """Map normalized commands to PWM; hardware access is opt-in."""

    def __init__(
        self,
        chassis=None,
        motors_enabled: bool = False,
        config: WheelMappingConfig = WheelMappingConfig(),
    ):
        if motors_enabled and chassis is None:
            raise ValueError("motors_enabled requires a chassis")
        self.chassis = chassis
        self.motors_enabled = bool(motors_enabled)
        self.config = config
        self._lock = threading.Lock()
        self._moving = False
        # Force the first explicit stop through to the motor controller. Later
        # stop frames are de-duplicated so a lost-boundary condition does not
        # spend 150 ms per frame repeating the same physical stop sequence.
        self._stop_written = False
        self._last_state = {
            "mode": "hardware" if self.motors_enabled else "dry-run",
            "action": "stopped",
            "left_pwm": 0,
            "right_pwm": 0,
            "front_left_pwm": 0,
            "rear_left_pwm": 0,
            "front_right_pwm": 0,
            "rear_right_pwm": 0,
            "reason": "initialized",
            "updated_at": time.monotonic(),
        }

    def command_to_four_pwm(
        self, command: DifferentialDriveCommand
    ) -> tuple[int, int, int, int]:
        if command.action == "stop":
            return 0, 0, 0, 0

        steering = float(np.clip(command.steering, -1.0, 1.0))
        delta = int(round(steering * self.config.front_steering_delta_pwm))
        if self.config.maximum_steering_delta_pwm > 0:
            delta = int(
                np.clip(
                    delta,
                    -self.config.maximum_steering_delta_pwm,
                    self.config.maximum_steering_delta_pwm,
                )
            )
        # Respect the calibrated moving floor in either turn direction. With
        # asymmetric straight trims, safe positive/negative limits differ.
        moving_floor = self.config.minimum_moving_pwm
        positive_limit = max(
            0,
            min(
                self.config.pwm_limit - self.config.front_left_base_pwm,
                self.config.pwm_limit - self.config.rear_left_base_pwm,
                self.config.front_right_base_pwm - moving_floor,
                self.config.rear_right_base_pwm - moving_floor,
            ),
        )
        negative_limit = max(
            0,
            min(
                self.config.front_left_base_pwm - moving_floor,
                self.config.rear_left_base_pwm - moving_floor,
                self.config.pwm_limit - self.config.front_right_base_pwm,
                self.config.pwm_limit - self.config.rear_right_base_pwm,
            ),
        )
        delta = int(np.clip(delta, -negative_limit, positive_limit))
        front_left = int(
            np.clip(
                self.config.front_left_base_pwm + delta,
                0,
                self.config.pwm_limit,
            )
        )
        front_right = int(
            np.clip(
                self.config.front_right_base_pwm - delta,
                0,
                self.config.pwm_limit,
            )
        )
        rear_left = int(
            np.clip(
                self.config.rear_left_base_pwm + delta,
                0,
                self.config.pwm_limit,
            )
        )
        rear_right = int(
            np.clip(
                self.config.rear_right_base_pwm - delta,
                0,
                self.config.pwm_limit,
            )
        )
        return (
            front_left,
            rear_left,
            front_right,
            rear_right,
        )

    def apply(self, command: DifferentialDriveCommand) -> dict:
        if command.action == "stop":
            return self.stop(command.reason or "controller requested stop")

        front_left, rear_left, front_right, rear_right = (
            self.command_to_four_pwm(command)
        )
        state = {
            "mode": "hardware" if self.motors_enabled else "dry-run",
            "action": command.action,
            "left_pwm": front_left,
            "right_pwm": front_right,
            "front_left_pwm": front_left,
            "rear_left_pwm": rear_left,
            "front_right_pwm": front_right,
            "rear_right_pwm": rear_right,
            "reason": command.reason,
            "updated_at": time.monotonic(),
        }
        with self._lock:
            if self.motors_enabled:
                if not self._moving and self.config.transition_time > 0:
                    # Ramp only when leaving a stopped state. Ramping every
                    # camera frame blocks the control loop for the full
                    # transition time and turns a 20 Hz camera into ~4 Hz LCC.
                    self.chassis.ramp_four_to(
                        front_left,
                        rear_left,
                        front_right,
                        rear_right,
                        self.config.transition_time,
                    )
                else:
                    previous = (
                        self._last_state["front_left_pwm"],
                        self._last_state["rear_left_pwm"],
                        self._last_state["front_right_pwm"],
                        self._last_state["rear_right_pwm"],
                    )
                    target = (front_left, rear_left, front_right, rear_right)
                    if target != previous:
                        self.chassis.set_four_wheels(*target)
            self._moving = True
            self._stop_written = False
            self._last_state = state
        return dict(state)

    def stop(self, reason: str = "stop requested") -> dict:
        state = {
            "mode": "hardware" if self.motors_enabled else "dry-run",
            "action": "stopped",
            "left_pwm": 0,
            "right_pwm": 0,
            "front_left_pwm": 0,
            "rear_left_pwm": 0,
            "front_right_pwm": 0,
            "rear_right_pwm": 0,
            "reason": str(reason),
            "updated_at": time.monotonic(),
        }
        with self._lock:
            if self.motors_enabled and not self._stop_written:
                self.chassis.stop()
            self._moving = False
            self._stop_written = True
            self._last_state = state
        return dict(state)

    def get_state(self) -> dict:
        with self._lock:
            state = dict(self._last_state)
        state["mapping"] = asdict(self.config)
        return state


class PerceptionMotionGate:
    """Require stable valid perception before starting or resuming motion.

    A stop command is still applied immediately. Unlike a process-lifetime
    latch, a transient bad frame can recover after several consecutive valid
    commands, which is necessary for a long outer-loop run while preserving a
    stopped validation window before the wheels move again.
    """

    def __init__(self, resume_valid_frames: int = 4):
        self.resume_valid_frames = int(resume_valid_frames)
        if self.resume_valid_frames < 1:
            raise ValueError("resume_valid_frames must be at least 1")
        self._ready = False
        self._consecutive_valid = 0
        self._last_stop_reason = "waiting for initial perception"

    def reset(self, reason: str = "motion gate reset") -> None:
        self._ready = False
        self._consecutive_valid = 0
        self._last_stop_reason = str(reason)

    def filter(
        self, command: DifferentialDriveCommand
    ) -> DifferentialDriveCommand:
        if command.action == "stop":
            self.reset(command.reason or "perception requested stop")
            return command
        if self._ready:
            return command

        self._consecutive_valid += 1
        if self._consecutive_valid >= self.resume_valid_frames:
            self._ready = True
            return command
        return DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            command.confidence,
            (
                "validating perception before motion "
                f"({self._consecutive_valid}/{self.resume_valid_frames})"
            ),
        )

    def get_state(self) -> dict:
        return {
            "ready": self._ready,
            "consecutive_valid": self._consecutive_valid,
            "resume_valid_frames": self.resume_valid_frames,
            "last_stop_reason": self._last_stop_reason,
        }


class CommandWatchdog:
    """Stop the vehicle when fresh perception commands stop arriving."""

    def __init__(
        self,
        wheel_driver: SafeWheelDriver,
        timeout: float = 2.5,
        check_interval: float = 0.05,
    ):
        if timeout <= 0 or check_interval <= 0:
            raise ValueError("watchdog timings must be positive")
        self.wheel_driver = wheel_driver
        self.timeout = float(timeout)
        self.check_interval = float(check_interval)
        self._lock = threading.Lock()
        self._last_heartbeat: Optional[float] = None
        self._armed = False
        self._tripped = False
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def arm(self) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()
            self._armed = True
            self._tripped = False

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()
            self._tripped = False

    def disarm(self, stop: bool = True) -> None:
        with self._lock:
            self._armed = False
            self._last_heartbeat = None
        if stop:
            self.wheel_driver.stop("watchdog disarmed")

    def get_state(self) -> dict:
        with self._lock:
            age = (
                None
                if self._last_heartbeat is None
                else time.monotonic() - self._last_heartbeat
            )
            return {
                "armed": self._armed,
                "tripped": self._tripped,
                "heartbeat_age": age,
                "timeout": self.timeout,
            }

    def _run(self) -> None:
        while not self._closed.wait(self.check_interval):
            should_stop = False
            with self._lock:
                if (
                    self._armed
                    and not self._tripped
                    and self._last_heartbeat is not None
                    and time.monotonic() - self._last_heartbeat > self.timeout
                ):
                    self._tripped = True
                    should_stop = True
            if should_stop:
                self.wheel_driver.stop("perception watchdog timeout")

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=max(1.0, self.check_interval * 3))
        self.disarm(stop=True)
