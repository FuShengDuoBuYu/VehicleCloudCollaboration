"""Safety boundary between normalized LCC commands and the physical chassis."""

from dataclasses import asdict, dataclass
import threading
import time
from typing import Optional

import numpy as np

from .lane_centering import DifferentialDriveCommand


@dataclass(frozen=True)
class WheelMappingConfig:
    pwm_limit: int = 30
    minimum_moving_pwm: int = 10
    left_scale: float = 1.0
    right_scale: float = 1.0
    left_sign: int = 1
    right_sign: int = 1
    transition_time: float = 0.15

    def __post_init__(self):
        if not 1 <= self.pwm_limit <= 255:
            raise ValueError("pwm_limit must be in [1, 255]")
        if not 0 <= self.minimum_moving_pwm <= self.pwm_limit:
            raise ValueError("minimum_moving_pwm must be in [0, pwm_limit]")
        if self.left_scale <= 0 or self.right_scale <= 0:
            raise ValueError("wheel scales must be positive")
        if self.left_sign not in (-1, 1) or self.right_sign not in (-1, 1):
            raise ValueError("wheel signs must be -1 or 1")
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
        self._last_state = {
            "mode": "hardware" if self.motors_enabled else "dry-run",
            "action": "stopped",
            "left_pwm": 0,
            "right_pwm": 0,
            "reason": "initialized",
            "updated_at": time.monotonic(),
        }

    def _map_speed(self, speed: float, scale: float, sign: int) -> int:
        speed = float(np.clip(speed, -1.0, 1.0))
        magnitude = abs(speed)
        if magnitude < 1e-4:
            return 0
        usable_range = self.config.pwm_limit - self.config.minimum_moving_pwm
        pwm = self.config.minimum_moving_pwm + magnitude * usable_range
        pwm = min(self.config.pwm_limit, int(round(pwm * scale)))
        direction = 1 if speed >= 0 else -1
        return int(pwm * direction * sign)

    def command_to_pwm(
        self, command: DifferentialDriveCommand
    ) -> tuple[int, int]:
        if command.action == "stop":
            return 0, 0
        return (
            self._map_speed(
                command.left_speed,
                self.config.left_scale,
                self.config.left_sign,
            ),
            self._map_speed(
                command.right_speed,
                self.config.right_scale,
                self.config.right_sign,
            ),
        )

    def apply(self, command: DifferentialDriveCommand) -> dict:
        if command.action == "stop":
            return self.stop(command.reason or "controller requested stop")

        left_pwm, right_pwm = self.command_to_pwm(command)
        if self.motors_enabled:
            self.chassis.ramp_to(
                left_pwm,
                right_pwm,
                self.config.transition_time,
            )
        state = {
            "mode": "hardware" if self.motors_enabled else "dry-run",
            "action": command.action,
            "left_pwm": left_pwm,
            "right_pwm": right_pwm,
            "reason": command.reason,
            "updated_at": time.monotonic(),
        }
        with self._lock:
            self._last_state = state
        return dict(state)

    def stop(self, reason: str = "stop requested") -> dict:
        if self.motors_enabled:
            self.chassis.stop()
        state = {
            "mode": "hardware" if self.motors_enabled else "dry-run",
            "action": "stopped",
            "left_pwm": 0,
            "right_pwm": 0,
            "reason": str(reason),
            "updated_at": time.monotonic(),
        }
        with self._lock:
            self._last_state = state
        return dict(state)

    def get_state(self) -> dict:
        with self._lock:
            state = dict(self._last_state)
        state["mapping"] = asdict(self.config)
        return state


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
