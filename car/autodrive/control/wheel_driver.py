"""Bounded wheel mapping and opt-in physical actuation."""
from __future__ import annotations

from .lane_centering import DifferentialDriveCommand
from dataclasses import asdict
from dataclasses import dataclass
from numbers import Real
import math
import numpy as np
import threading
import time


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
    tight_turn_start_steering: float = 0.50
    tight_turn_full_steering: float = 0.70
    tight_turn_outside_pwm: int = 0
    tight_turn_inside_pwm: int = 0
    transition_time: float = 0.25
    stationary_pivot_pwm: int = 0

    def __post_init__(self):
        if self.drive_mode != "four-wheel-trim":
            raise ValueError("drive_mode must be four-wheel-trim")
        if not 1 <= self.pwm_limit <= 255:
            raise ValueError("pwm_limit must be in [1, 255]")
        if (type(self.stationary_pivot_pwm) is not int
                or not 0 <= self.stationary_pivot_pwm <= min(30, self.pwm_limit)):
            raise ValueError("stationary_pivot_pwm must be an integer in [0, min(30, pwm_limit)]")
        if not 0 <= self.minimum_moving_pwm <= self.pwm_limit:
            raise ValueError("minimum_moving_pwm must be in [0, pwm_limit]")
        direct_values = (
            self.front_left_base_pwm,
            self.rear_left_base_pwm,
            self.front_right_base_pwm,
            self.rear_right_base_pwm,
            self.front_steering_delta_pwm,
            self.maximum_steering_delta_pwm,
            self.tight_turn_outside_pwm,
            self.tight_turn_inside_pwm,
        )
        if any(value < 0 or value > self.pwm_limit for value in direct_values):
            raise ValueError("per-wheel PWM values must be inside pwm_limit")
        if not (
            0.0
            < self.tight_turn_start_steering
            < self.tight_turn_full_steering
            <= 1.0
        ):
            raise ValueError(
                "tight-turn steering thresholds must be ordered inside (0, 1]"
            )
        if self.tight_turn_outside_pwm and self.tight_turn_outside_pwm < max(
            self.front_left_base_pwm,
            self.rear_left_base_pwm,
            self.front_right_base_pwm,
            self.rear_right_base_pwm,
        ):
            raise ValueError(
                "tight_turn_outside_pwm must be zero or at least every base PWM"
            )
        if (
            self.tight_turn_inside_pwm != 0
            and self.tight_turn_inside_pwm < self.minimum_moving_pwm
        ):
            raise ValueError(
                "tight_turn_inside_pwm must be zero or at least minimum_moving_pwm"
            )
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
        self._motion_deadline = None
        self._inhibited_reason = None
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

        if command.action in {"pivot-right", "pivot-left"}:
            if self.config.stationary_pivot_pwm == 0:
                raise ValueError("stationary pivot is disabled")
            values = (command.steering, command.left_speed, command.right_speed)
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(value) or not -1 <= value <= 1 for value in values):
                raise ValueError("stationary pivot proposals must be finite normalized numbers")
            direction = 1 if command.action == "pivot-right" else -1
            if not (direction * command.steering > 0
                    and direction * command.left_speed > 0
                    and direction * command.right_speed < 0):
                raise ValueError("stationary pivot action and proposed directions disagree")
            pwm = direction * self.config.stationary_pivot_pwm
            return pwm, pwm, -pwm, -pwm

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
        # Bound reductions by the moving floor. Saturating the outside wheels
        # at pwm_limit must still allow the inside wheels to slow for a turn.
        # Each outside wheel is independently capped below.
        moving_floor = self.config.minimum_moving_pwm
        positive_limit = max(
            0,
            min(
                self.config.front_right_base_pwm - moving_floor,
                self.config.rear_right_base_pwm - moving_floor,
            ),
        )
        negative_limit = max(
            0,
            min(
                self.config.front_left_base_pwm - moving_floor,
                self.config.rear_left_base_pwm - moving_floor,
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
        # A grounded 30/10 forward-only arc was still too wide for the field's
        # 90-degree outer corner. Near steering saturation, increase the
        # outside pair and progressively stop the inside pair. No wheel ever
        # reverses, so this is a one-sided advancing pivot rather than the old
        # counter-rotating in-place turn.
        blend = command.tight_turn_factor
        if blend is None:
            # Diagnostic/manual commands created outside the LCC do not carry
            # the near-field turn factor. Preserve their steering-only mapping.
            blend = float(
                np.clip(
                    (
                        abs(steering)
                        - self.config.tight_turn_start_steering
                    )
                    / (
                        self.config.tight_turn_full_steering
                        - self.config.tight_turn_start_steering
                    ),
                    0.0,
                    1.0,
                )
            )
        else:
            blend = float(np.clip(blend, 0.0, 1.0))
        if self.config.tight_turn_outside_pwm > 0 and blend > 0.0:
            outside_pwm = self.config.tight_turn_outside_pwm
            inside_pwm = self.config.tight_turn_inside_pwm
            if steering > 0:
                front_left = int(round(front_left * (1.0 - blend) + outside_pwm * blend))
                rear_left = int(round(rear_left * (1.0 - blend) + outside_pwm * blend))
                front_right = int(round(front_right * (1.0 - blend) + inside_pwm * blend))
                rear_right = int(round(rear_right * (1.0 - blend) + inside_pwm * blend))
            else:
                front_right = int(round(front_right * (1.0 - blend) + outside_pwm * blend))
                rear_right = int(round(rear_right * (1.0 - blend) + outside_pwm * blend))
                front_left = int(round(front_left * (1.0 - blend) + inside_pwm * blend))
                rear_left = int(round(rear_left * (1.0 - blend) + inside_pwm * blend))
        # Tight-turn interpolation used to emit PWM 1..9 even though the
        # grounded moving floor is 10. Static friction then decided whether a
        # wheel moved, making identical corners non-repeatable. Snap that dead
        # zone to the nearest intentional state: stopped (0) or reliably moving
        # (minimum_moving_pwm). Ties preserve forward progress.
        moving_floor = self.config.minimum_moving_pwm
        if moving_floor > 0:
            reliable_threshold = (moving_floor + 1) // 2

            def snap_dead_zone(value):
                if not 0 < value < moving_floor:
                    return value
                return moving_floor if value >= reliable_threshold else 0

            front_left = snap_dead_zone(front_left)
            rear_left = snap_dead_zone(rear_left)
            front_right = snap_dead_zone(front_right)
            rear_right = snap_dead_zone(rear_right)
        return (
            front_left,
            rear_left,
            front_right,
            rear_right,
        )

    def apply(self, command: DifferentialDriveCommand, *, deadline=None) -> dict:
        if command.action == "stop":
            return self.stop(command.reason or "controller requested stop")

        try:
            front_left, rear_left, front_right, rear_right = self.command_to_four_pwm(command)
        except ValueError:
            if command.action in {"pivot-right", "pivot-left"}:
                self.stop("invalid or disabled stationary pivot")
            raise
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
            if self._inhibited_reason is not None:
                return dict(self._last_state)
            if deadline is not None and (type(deadline) not in (int, float)
                    or not math.isfinite(deadline) or time.monotonic() >= deadline):
                return self._stop_locked('visual step expired before application')
            self._motion_deadline = deadline
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
            return self._stop_locked(reason, state)

    def _stop_locked(self, reason, state=None):
        if state is None:
            state = dict(self._last_state, action='stopped', left_pwm=0, right_pwm=0,
                front_left_pwm=0, rear_left_pwm=0, front_right_pwm=0, rear_right_pwm=0,
                reason=str(reason), updated_at=time.monotonic())
        if self.motors_enabled and not self._stop_written:
            self.chassis.stop()
        self._moving = False
        self._stop_written = True
        self._motion_deadline = None
        self._last_state = state
        return dict(state)

    def inhibit(self, reason):
        """An operator/lease cancellation cannot be undone in this driver instance."""
        with self._lock:
            self._inhibited_reason = str(reason)
            return self._stop_locked(reason)

    def check_motion_deadline(self):
        with self._lock:
            if self._motion_deadline is not None and time.monotonic() >= self._motion_deadline:
                return self._stop_locked('visual step time ceiling')
            return dict(self._last_state)

    def get_state(self) -> dict:
        with self._lock:
            state = dict(self._last_state)
        state["mapping"] = asdict(self.config)
        return state

