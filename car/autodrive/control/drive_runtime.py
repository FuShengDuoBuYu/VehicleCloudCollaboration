"""Safety boundary between normalized LCC commands and the physical chassis."""

from dataclasses import asdict, dataclass
import threading
import time
from typing import Optional

import numpy as np

from .lane_centering import DifferentialDriveCommand, LaneEstimate


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


@dataclass(frozen=True)
class CornerContinuationConfig:
    """Bounded continuation through a sharp corner's one-boundary blind spot."""

    enabled: bool = False
    enter_steering_threshold: float = 0.50
    exit_steering_threshold: float = 0.18
    exit_min_confidence: float = 0.35
    exit_valid_frames: int = 3
    maximum_hold_seconds: float = 2.80
    progress_timeout_seconds: float = 1.00
    minimum_heading_progress: float = 0.015
    minimum_lateral_progress: float = 0.010
    apex_trigger_factor: float = 0.50
    apex_near_heading_trigger: float = 0.43
    apex_commit_delay_seconds: float = 0.30
    minimum_apex_seconds: float = 0.40
    maximum_apex_seconds: float = 0.70
    apex_exit_near_heading: float = 0.30
    apex_exit_valid_frames: int = 2
    minimum_transverse_boundary_ratio: float = 0.01
    reacquire_steering_limit: float = 0.35
    tight_turn_factor_cap: float = 0.0

    def __post_init__(self):
        if not 0.0 < self.enter_steering_threshold <= 1.0:
            raise ValueError("enter_steering_threshold must be in (0, 1]")
        if not 0.0 <= self.exit_steering_threshold < self.enter_steering_threshold:
            raise ValueError(
                "exit_steering_threshold must be below enter_steering_threshold"
            )
        if not 0.0 <= self.exit_min_confidence <= 1.0:
            raise ValueError("exit_min_confidence must be in [0, 1]")
        if self.exit_valid_frames < 1:
            raise ValueError("exit_valid_frames must be at least 1")
        if self.maximum_hold_seconds <= 0:
            raise ValueError("maximum_hold_seconds must be positive")
        if not 0.0 < self.progress_timeout_seconds <= self.maximum_hold_seconds:
            raise ValueError(
                "progress_timeout_seconds must be positive and no greater "
                "than maximum_hold_seconds"
            )
        if not 0.0 < self.minimum_heading_progress <= 1.0:
            raise ValueError("minimum_heading_progress must be in (0, 1]")
        if not 0.0 < self.minimum_lateral_progress <= 1.0:
            raise ValueError("minimum_lateral_progress must be in (0, 1]")
        if not 0.0 < self.apex_trigger_factor <= 1.0:
            raise ValueError("apex_trigger_factor must be in (0, 1]")
        if not 0.0 < self.apex_near_heading_trigger <= 1.0:
            raise ValueError("apex_near_heading_trigger must be in (0, 1]")
        if not 0.0 < self.apex_commit_delay_seconds <= self.maximum_hold_seconds:
            raise ValueError(
                "apex_commit_delay_seconds must be positive and no greater "
                "than maximum_hold_seconds"
            )
        if not 0.0 < self.minimum_apex_seconds <= self.maximum_hold_seconds:
            raise ValueError(
                "minimum_apex_seconds must be positive and no greater than "
                "maximum_hold_seconds"
            )
        if not (
            self.minimum_apex_seconds
            <= self.maximum_apex_seconds
            <= self.maximum_hold_seconds
        ):
            raise ValueError(
                "maximum_apex_seconds must be no smaller than "
                "minimum_apex_seconds and no greater than maximum_hold_seconds"
            )
        if not 0.0 <= self.apex_exit_near_heading < self.apex_near_heading_trigger:
            raise ValueError(
                "apex_exit_near_heading must be non-negative and below "
                "apex_near_heading_trigger"
            )
        if self.apex_exit_valid_frames < 1:
            raise ValueError("apex_exit_valid_frames must be at least 1")
        if not 0.0 < self.minimum_transverse_boundary_ratio <= 1.0:
            raise ValueError(
                "minimum_transverse_boundary_ratio must be in (0, 1]"
            )
        if not (
            self.exit_steering_threshold
            < self.reacquire_steering_limit
            <= 1.0
        ):
            raise ValueError(
                "reacquire_steering_limit must be above the exit threshold "
                "and no greater than 1"
            )
        if not 0.0 <= self.tight_turn_factor_cap <= 1.0:
            raise ValueError("tight_turn_factor_cap must be in [0, 1]")


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

    _HISTORICAL_SOURCES = {"history", "visible-history"}

    def __init__(
        self,
        resume_valid_frames: int = 4,
        resume_min_confidence: float = 0.0,
        maximum_lateral_jump: float = 0.0,
        maximum_heading_jump: float = 0.0,
        require_consistent_source: bool = False,
    ):
        self.resume_valid_frames = int(resume_valid_frames)
        self.resume_min_confidence = float(resume_min_confidence)
        self.maximum_lateral_jump = float(maximum_lateral_jump)
        self.maximum_heading_jump = float(maximum_heading_jump)
        self.require_consistent_source = bool(require_consistent_source)
        if self.resume_valid_frames < 1:
            raise ValueError("resume_valid_frames must be at least 1")
        if not 0.0 <= self.resume_min_confidence <= 1.0:
            raise ValueError("resume_min_confidence must be in [0, 1]")
        if self.maximum_lateral_jump < 0.0:
            raise ValueError("maximum_lateral_jump must not be negative")
        if self.maximum_heading_jump < 0.0:
            raise ValueError("maximum_heading_jump must not be negative")
        self._ready = False
        self._consecutive_valid = 0
        self._last_stop_reason = "waiting for initial perception"
        self._candidate_source: Optional[str] = None
        self._last_lateral: Optional[float] = None
        self._last_heading: Optional[float] = None
        self._last_source: Optional[str] = None

    def reset(self, reason: str = "motion gate reset") -> None:
        self._ready = False
        self._consecutive_valid = 0
        self._last_stop_reason = str(reason)
        self._candidate_source = None
        self._last_lateral = None
        self._last_heading = None
        self._last_source = None

    @staticmethod
    def _stop_command(
        command: DifferentialDriveCommand,
        reason: str,
    ) -> DifferentialDriveCommand:
        return DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            command.confidence,
            reason,
        )

    @staticmethod
    def _source(boundary_result) -> Optional[str]:
        if boundary_result is None:
            return None
        source = getattr(boundary_result, "source", None)
        return None if source is None else str(source)

    def _continuity_issue(
        self,
        estimate: Optional[LaneEstimate],
        boundary_result,
    ) -> Optional[str]:
        if estimate is None or boundary_result is None or not estimate.valid:
            return None
        lateral = float(estimate.lateral_error)
        heading = float(estimate.heading_error)
        source = self._source(boundary_result)
        # A history-backed estimate reuses extrapolated geometry and can jump
        # for one frame when a fresh fit momentarily disappears.  Treating it
        # as a new continuity anchor caused run 20260803_180818 to stop at
        # sample 242, then reject the following fresh outer-edge estimate as a
        # second jump.  The controller's own confidence/missing-boundary stops
        # still apply; history simply cannot trip or rebase the fresh-geometry
        # continuity check by itself.
        if source in self._HISTORICAL_SOURCES:
            return None
        if (
            self.maximum_lateral_jump > 0.0
            and self._last_lateral is not None
            and abs(lateral - self._last_lateral) > self.maximum_lateral_jump
        ):
            return (
                "perception lateral estimate jumped "
                f"{abs(lateral - self._last_lateral):.3f} > "
                f"{self.maximum_lateral_jump:.3f}"
            )
        if (
            self.maximum_heading_jump > 0.0
            and self._last_heading is not None
            and abs(heading - self._last_heading) > self.maximum_heading_jump
        ):
            return (
                "perception heading estimate jumped "
                f"{abs(heading - self._last_heading):.3f} > "
                f"{self.maximum_heading_jump:.3f}"
            )
        if (
            self.require_consistent_source
            and not self._ready
            and self._candidate_source is not None
            and source is not None
            and source != self._candidate_source
        ):
            return (
                "perception boundary source changed while validating "
                f"({self._candidate_source} -> {source})"
            )
        return None

    def _remember_observation(
        self,
        estimate: Optional[LaneEstimate],
        boundary_result,
    ) -> None:
        if estimate is None or boundary_result is None or not estimate.valid:
            return
        if self._source(boundary_result) in self._HISTORICAL_SOURCES:
            return
        self._last_lateral = float(estimate.lateral_error)
        self._last_heading = float(estimate.heading_error)
        self._last_source = self._source(boundary_result)

    def filter(
        self,
        command: DifferentialDriveCommand,
        estimate: Optional[LaneEstimate] = None,
        boundary_result=None,
        allow_discontinuity: bool = False,
    ) -> DifferentialDriveCommand:
        if command.action == "stop":
            self.reset(command.reason or "perception requested stop")
            return command

        source = self._source(boundary_result)
        issue = None
        if not allow_discontinuity:
            issue = self._continuity_issue(estimate, boundary_result)
        if issue is not None:
            self._ready = False
            self._consecutive_valid = 0
            self._last_stop_reason = issue
            self._candidate_source = source
            self._remember_observation(estimate, boundary_result)
            return self._stop_command(command, issue)

        self._remember_observation(estimate, boundary_result)
        if self._ready:
            return command

        if command.confidence < self.resume_min_confidence:
            self._consecutive_valid = 0
            self._last_stop_reason = "waiting for resume confidence"
            self._candidate_source = source
            return self._stop_command(
                command,
                (
                    "waiting for resume confidence "
                    f"({command.confidence:.3f} < "
                    f"{self.resume_min_confidence:.3f})"
                ),
            )

        if self._candidate_source is None:
            self._candidate_source = source
        self._consecutive_valid += 1
        if self._consecutive_valid >= self.resume_valid_frames:
            self._ready = True
            return command
        return self._stop_command(
            command,
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
            "resume_min_confidence": self.resume_min_confidence,
            "maximum_lateral_jump": self.maximum_lateral_jump,
            "maximum_heading_jump": self.maximum_heading_jump,
            "require_consistent_source": self.require_consistent_source,
            "last_boundary_source": self._last_source,
            "last_stop_reason": self._last_stop_reason,
        }


class CornerContinuationGate:
    """Carry a committed arc briefly when a 90-degree corner leaves one edge.

    The boundary tracker represents each edge as x(y). Near a tight corner the
    remaining edge can become almost horizontal, so its near-field polynomial
    extrapolation is temporarily unreliable. This gate may bridge only that
    known one-boundary condition. It never overrides a yellow-line hazard,
    missing boundary track, watchdog condition, or inference timeout.
    """

    _ONE_BOUNDARY_SOURCES = {
        "outer+width",
        "inner+width",
        "history",
        "visible-history",
    }
    _RECOVERABLE_REASONS = {
        "boundary remains visible outside x(y) fit",
        "road confidence below safety threshold",
        "lateral error exceeds recovery limit",
        "too few valid road rows",
    }

    def __init__(
        self,
        config: CornerContinuationConfig = CornerContinuationConfig(),
    ):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._direction = 0
        self._last_command: Optional[DifferentialDriveCommand] = None
        self._committed_boundary_role: Optional[str] = None
        self._committed_at: Optional[float] = None
        self._hold_started_at: Optional[float] = None
        self._last_progress_at: Optional[float] = None
        self._best_heading_magnitude: Optional[float] = None
        self._best_lateral_magnitude: Optional[float] = None
        self._progress_boundary_role: Optional[str] = None
        self._apex_started_at: Optional[float] = None
        self._apex_trigger_reason: Optional[str] = None
        self._apex_completed = False
        self._apex_completion_reason: Optional[str] = None
        self._apex_exit_valid_count = 0
        self._exit_valid_count = 0
        self._last_reason = "inactive"

    @staticmethod
    def _steering_direction(command: DifferentialDriveCommand) -> int:
        if command.steering > 0:
            return 1
        if command.steering < 0:
            return -1
        return 0

    @staticmethod
    def _with_tight_turn_factor(
        command: DifferentialDriveCommand,
        factor: float,
    ) -> DifferentialDriveCommand:
        return DifferentialDriveCommand(
            command.action,
            command.steering,
            command.left_speed,
            command.right_speed,
            command.confidence,
            command.reason,
            float(factor),
        )

    def _reacquire_steering(self) -> float:
        if self._last_command is None:
            return 0.0
        magnitude = min(
            abs(float(self._last_command.steering)),
            self.config.reacquire_steering_limit,
        )
        return magnitude if self._direction > 0 else -magnitude

    @staticmethod
    def _boundary_role(boundary_result) -> Optional[str]:
        """Return only a directly observed single-edge role.

        History has no reliable role of its own, while ``both`` is the exit
        geometry rather than a replacement single edge.  Keeping those out of
        this value prevents one transitional frame from changing the command
        that grounded the committed corner.
        """
        if boundary_result is None:
            return None
        source = str(getattr(boundary_result, "source", ""))
        if source == "outer+width":
            return "outer"
        if source == "inner+width":
            return "inner"
        return None

    def _set_progress_baseline(
        self,
        estimate: LaneEstimate,
        boundary_result,
        current_time: float,
    ) -> None:
        """Start a progress window in one comparable boundary coordinate."""
        self._last_progress_at = current_time
        self._best_heading_magnitude = (
            abs(float(estimate.heading_error)) if estimate.valid else None
        )
        self._best_lateral_magnitude = (
            abs(float(estimate.lateral_error)) if estimate.valid else None
        )
        role = self._boundary_role(boundary_result)
        if role is not None:
            self._progress_boundary_role = role

    def _refresh_grounded_progress(
        self,
        estimate: LaneEstimate,
        boundary_result,
        current_time: float,
    ) -> None:
        """A fresh moving LCC command is positive recovery evidence.

        The former timer advanced only on raw stop frames.  In the fourth
        field corner the controller recovered for five frames, but that valid
        movement did not refresh the timer; its next recoverable stop therefore
        expired immediately.  Refreshing here grants the normal bounded
        recovery window only after a fresh, non-stop perception command.
        """
        if self._hold_started_at is None:
            return
        self._set_progress_baseline(
            estimate,
            boundary_result,
            current_time,
        )

    @staticmethod
    def _stop(reason: str, confidence: float) -> DifferentialDriveCommand:
        return DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            confidence,
            reason,
        )

    def _arm_apex(
        self,
        command: DifferentialDriveCommand,
        estimate: LaneEstimate,
        boundary_result,
        current_time: float,
    ) -> None:
        # The apex latch exists only to bridge a one-boundary blind corner.
        # If both physical edges are visible, normal LCC has enough geometry
        # to steer and recenter without a forced minimum-duration pivot.
        if not self._boundary_allows_continuation(boundary_result):
            return
        factor = float(command.tight_turn_factor or 0.0)
        near_heading = (
            abs(float(estimate.near_heading_error)) if estimate.valid else 0.0
        )
        factor_triggered = factor >= self.config.apex_trigger_factor
        near_heading_triggered = (
            near_heading >= self.config.apex_near_heading_trigger
        )
        commit_delay_elapsed = bool(
            self._committed_at is not None
            and current_time - self._committed_at
            >= self.config.apex_commit_delay_seconds
        )
        if (
            not self._apex_completed
            and self._apex_started_at is None
            and (factor_triggered or near_heading_triggered or commit_delay_elapsed)
        ):
            self._apex_started_at = current_time
            self._apex_trigger_reason = (
                "tight-turn factor"
                if factor_triggered
                else "near-heading geometry"
                if near_heading_triggered
                else "committed-corner delay"
            )

    def _arm_delayed_apex(
        self,
        boundary_result,
        current_time: float,
    ) -> None:
        """Apply the fixed apex fallback even after fresh fitting drops out.

        A control-loop pause can skip directly from the committing frame to a
        history-backed frame after ``apex_commit_delay_seconds``.  The old
        code checked the delay only while a fresh edge with the original role
        was still fitted, so that skipped interval could suppress the apex
        entirely.  Once a direction is committed, elapsed time is sufficient
        to arm the bounded apex as long as the tracked physical boundary still
        permits continuation.  Yellow or genuinely missing boundaries remain
        non-recoverable and are rejected before this method is reached.
        """
        if (
            not self._apex_completed
            and self._apex_started_at is None
            and self._committed_at is not None
            and current_time - self._committed_at
            >= self.config.apex_commit_delay_seconds
            and self._boundary_allows_continuation(boundary_result)
        ):
            self._apex_started_at = current_time
            self._apex_trigger_reason = "committed-corner delay"

    def _apex_active(
        self,
        current_time: float,
        estimate: Optional[LaneEstimate] = None,
        boundary_result=None,
    ) -> bool:
        if self._apex_started_at is None or self._apex_completed:
            return False
        apex_age = current_time - self._apex_started_at
        if apex_age >= self.config.maximum_apex_seconds:
            self._apex_completed = True
            self._apex_completion_reason = "hard-time-limit"
            self._last_reason = "apex hard time limit reached"
            return False

        # Observations from the forced minimum turn do not prove that the new
        # straight has been acquired. The failed second field corner had five
        # such observations banked before 0.40 s, so it left the 30/0 apex on
        # the very first eligible frame. Start the consecutive exit window
        # only after the grounded minimum has actually elapsed.
        if apex_age < self.config.minimum_apex_seconds:
            self._apex_exit_valid_count = 0
            return True

        # Do not decide from elapsed time alone. The latest physical run left
        # the apex while the outer yellow edge was still nearly transverse,
        # then drove a wide arc into that edge. After the bounded minimum,
        # require either two genuinely visible boundaries or a near edge that
        # has become aligned with the new straight. Consecutive observations
        # reject a single noisy fit; the hard limit above prevents a blind
        # indefinite pivot when the second boundary remains outside the FOV.
        if estimate is not None and boundary_result is not None:
            both_visible = bool(
                boundary_result.valid
                and not boundary_result.yellow_hazard
                and boundary_result.source == "both"
            )
            near_aligned = bool(
                estimate.valid
                and abs(float(estimate.near_heading_error))
                <= self.config.apex_exit_near_heading
            )
            self._apex_exit_valid_count = (
                self._apex_exit_valid_count + 1
                if both_visible or near_aligned
                else 0
            )

        if self._apex_exit_valid_count >= self.config.apex_exit_valid_frames:
            self._apex_completed = True
            self._apex_completion_reason = "alignment-restored"
            self._last_reason = "apex alignment restored"
            return False
        return True

    def _lcc_handoff_ready(
        self,
        command: DifferentialDriveCommand,
        estimate: LaneEstimate,
        boundary_result,
    ) -> bool:
        """Return control after a geometrically confirmed apex recovery.

        Waiting for two physical boundaries kept the old right arc active for
        another 2.7-3.5 seconds on the straight, even while LCC requested a
        left correction.  The apex latch already requires consecutive aligned
        observations; one fresh fitted edge is sufficient for ordinary LCC to
        resume once that bounded turn has completed.
        """
        source = (
            None
            if boundary_result is None
            else str(getattr(boundary_result, "source", ""))
        )
        return bool(
            self._apex_completed
            and self._apex_completion_reason == "alignment-restored"
            and command.action != "stop"
            and estimate.valid
            and command.confidence >= self.config.exit_min_confidence
            and source in {"both", "outer+width", "inner+width"}
            and abs(float(estimate.near_heading_error))
            <= self.config.apex_exit_near_heading
        )

    def _post_apex_lateral_stop_must_apply(
        self,
        command: DifferentialDriveCommand,
        estimate: LaneEstimate,
    ) -> bool:
        """Never steer farther past centre after the bounded apex is over."""
        if (
            not self._apex_completed
            or command.reason != "lateral error exceeds recovery limit"
            or not estimate.valid
        ):
            return False
        # A same-side error can be the conservative one-edge virtual centre
        # settling immediately after the pivot.  An opposite-side error means
        # the retained corner command has already carried the car through and
        # must not override LCC's hard safety stop.
        return float(estimate.lateral_error) * self._direction < 0.0

    def _boundary_allows_continuation(self, boundary_result) -> bool:
        if boundary_result is None or bool(boundary_result.yellow_hazard):
            return False
        source = str(getattr(boundary_result, "source", ""))
        transverse_boundary_visible = bool(
            source == "visible-history"
            and str(getattr(boundary_result, "reason", ""))
            == "boundary remains visible outside x(y) fit"
            and float(
                getattr(boundary_result, "boundary_visible_ratio", 0.0)
            )
            >= self.config.minimum_transverse_boundary_ratio
        )
        return bool(
            source in self._ONE_BOUNDARY_SOURCES
            and (bool(boundary_result.valid) or transverse_boundary_visible)
        )

    def filter(
        self,
        command: DifferentialDriveCommand,
        estimate: LaneEstimate,
        boundary_result,
        now: Optional[float] = None,
    ) -> DifferentialDriveCommand:
        if not self.config.enabled:
            return command
        current_time = time.monotonic() if now is None else float(now)
        yellow_hazard = bool(
            boundary_result is not None and boundary_result.yellow_hazard
        )
        if yellow_hazard:
            self.reset()
            self._last_reason = "yellow hazard stopped corner continuation"
            return command

        if (
            self._direction != 0
            and self._committed_at is not None
            and current_time - self._committed_at
            > self.config.maximum_hold_seconds
        ):
            self.reset()
            self._last_reason = "corner continuation hard time limit reached"
            return self._stop(self._last_reason, command.confidence)

        if self._direction != 0:
            # Check the time fallback on every safe continuation frame, not
            # only on a fresh fit with the same boundary role.  This makes a
            # committed 90-degree corner deterministic even when one control
            # interval is skipped and the next estimate comes from history.
            self._arm_delayed_apex(boundary_result, current_time)

        direction = self._steering_direction(command)
        if self._direction == 0:
            if (
                command.action != "stop"
                and abs(command.steering)
                >= self.config.enter_steering_threshold
                and self._boundary_allows_continuation(boundary_result)
            ):
                self._direction = direction
                self._last_command = command
                self._committed_boundary_role = self._boundary_role(
                    boundary_result
                )
                self._committed_at = current_time
                self._arm_apex(
                    command,
                    estimate,
                    boundary_result,
                    current_time,
                )
                self._last_reason = "sharp one-boundary corner committed"
            return (
                self._with_tight_turn_factor(command, 1.0)
                if self._apex_active(current_time, estimate, boundary_result)
                else command
            )

        if command.action != "stop":
            opposite_steering = bool(
                direction != 0
                and direction != self._direction
                and abs(command.steering) > self.config.exit_steering_threshold
            )
            if (
                not opposite_steering
                and direction == self._direction
                and abs(command.steering) > self.config.exit_steering_threshold
                and self._boundary_role(boundary_result)
                == self._committed_boundary_role
            ):
                self._last_command = command
                self._arm_apex(
                    command,
                    estimate,
                    boundary_result,
                    current_time,
                )
            self._refresh_grounded_progress(
                estimate,
                boundary_result,
                current_time,
            )
            apex_active = self._apex_active(
                current_time,
                estimate,
                boundary_result,
            )
            if self._lcc_handoff_ready(command, estimate, boundary_result):
                self.reset()
                self._last_reason = "aligned post-apex handoff to LCC"
                return command
            weak_or_opposite = bool(
                direction != self._direction
                or abs(command.steering) <= self.config.exit_steering_threshold
            )
            if apex_active and weak_or_opposite:
                return self._with_tight_turn_factor(
                    self._last_command,
                    1.0,
                )
            if opposite_steering:
                # A newly visible opposite correction is often exactly what
                # follows the sharp corner. Confirm it for the configured
                # apex-exit window instead of allowing one noisy frame to
                # reverse a still-committed pivot.
                if apex_active:
                    return self._with_tight_turn_factor(
                        self._last_command,
                        1.0,
                    )

            both_boundaries_restored = bool(
                boundary_result is not None
                and boundary_result.valid
                and not boundary_result.yellow_hazard
                and boundary_result.source == "both"
            )
            stable_exit = bool(
                estimate.valid
                and command.confidence >= self.config.exit_min_confidence
                and not apex_active
                and both_boundaries_restored
            )
            self._exit_valid_count = (
                self._exit_valid_count + 1 if stable_exit else 0
            )
            if self._exit_valid_count >= self.config.exit_valid_frames:
                self.reset()
                self._last_reason = "stable forward tracking restored"
                return command

            # A single fitted edge is exactly the ambiguous geometry this
            # state bridges. Do not let a weak/forward/opposite estimate cancel
            # the committed direction until both physical boundaries have
            # remained visible for the configured confirmation window. The
            # retained command is the latest meaningful same-direction arc,
            # with the forced 30/0 factor removed after the apex.
            if not apex_active and weak_or_opposite:
                self._last_reason = (
                    "holding corner direction until two-boundary reacquisition"
                )
                held = self._last_command
                return DifferentialDriveCommand(
                    "turn-right" if self._direction > 0 else "turn-left",
                    self._reacquire_steering(),
                    held.left_speed,
                    held.right_speed,
                    command.confidence,
                    self._last_reason,
                    min(
                        float(held.tight_turn_factor or 0.0),
                        self.config.tight_turn_factor_cap,
                    ),
                )
            return (
                self._with_tight_turn_factor(command, 1.0)
                if apex_active
                else command
            )

        if self._post_apex_lateral_stop_must_apply(command, estimate):
            self.reset()
            self._last_reason = "post-apex lateral safety stop"
            return command

        if (
            command.reason not in self._RECOVERABLE_REASONS
            or not self._boundary_allows_continuation(boundary_result)
            or self._last_command is None
        ):
            self.reset()
            self._last_reason = "non-recoverable stop"
            return command

        # The controller may issue a recoverable confidence/lateral stop on
        # the same frame that near-field geometry first proves the apex. Arm
        # from the estimate as well as the smoothed command factor so that one
        # ordering difference cannot suppress the whole tight-turn phase.
        self._arm_apex(
            self._last_command,
            estimate,
            boundary_result,
            current_time,
        )

        if self._hold_started_at is None:
            self._hold_started_at = current_time
            self._set_progress_baseline(
                estimate,
                boundary_result,
                current_time,
            )
        elif estimate.valid:
            role = self._boundary_role(boundary_result)
            if (
                role is not None
                and self._progress_boundary_role is not None
                and role != self._progress_boundary_role
            ):
                # Outer- and inner-derived virtual centrelines use different
                # coordinates.  The fourth field corner changed from outer to
                # inner after one ``both`` frame; comparing the new errors to
                # the old minima falsely reported a full second of no progress.
                self._set_progress_baseline(
                    estimate,
                    boundary_result,
                    current_time,
                )
            heading_magnitude = abs(float(estimate.heading_error))
            lateral_magnitude = abs(float(estimate.lateral_error))
            made_progress = False
            if (
                self._best_heading_magnitude is None
                or heading_magnitude
                <= self._best_heading_magnitude
                - self.config.minimum_heading_progress
            ):
                self._best_heading_magnitude = heading_magnitude
                made_progress = True
            if (
                self._best_lateral_magnitude is None
                or lateral_magnitude
                <= self._best_lateral_magnitude
                - self.config.minimum_lateral_progress
            ):
                self._best_lateral_magnitude = lateral_magnitude
                made_progress = True
            if made_progress:
                self._last_progress_at = current_time
        hold_age = current_time - self._hold_started_at
        if hold_age > self.config.maximum_hold_seconds:
            self.reset()
            self._last_reason = "corner continuation hard time limit reached"
            return command
        progress_age = (
            0.0
            if self._last_progress_at is None
            else current_time - self._last_progress_at
        )
        if progress_age > self.config.progress_timeout_seconds:
            self.reset()
            self._last_reason = "corner continuation stopped making progress"
            return command

        steering = self._reacquire_steering()
        action = "turn-right" if self._direction > 0 else "turn-left"
        tight_turn_factor = (
            1.0
            if self._apex_active(current_time, estimate, boundary_result)
            else min(
                float(self._last_command.tight_turn_factor or 0.0),
                self.config.tight_turn_factor_cap,
            )
        )
        self._last_reason = f"continuing committed corner ({command.reason})"
        return DifferentialDriveCommand(
            action,
            steering,
            self._last_command.left_speed,
            self._last_command.right_speed,
            command.confidence,
            self._last_reason,
            # Near-field geometry is unavailable during this dropout. Retain
            # only a configured, bounded part of the last grounded turn. This
            # is tighter than the ordinary 26/10 arc, without blindly keeping
            # the 30/0 apex command that previously over-rotated the chassis.
            tight_turn_factor,
        )

    def get_state(self, now: Optional[float] = None) -> dict:
        current_time = time.monotonic() if now is None else float(now)
        hold_age = (
            None
            if self._hold_started_at is None
            else max(0.0, current_time - self._hold_started_at)
        )
        progress_age = (
            None
            if self._last_progress_at is None
            else max(0.0, current_time - self._last_progress_at)
        )
        apex_age = (
            None
            if self._apex_started_at is None
            else max(0.0, current_time - self._apex_started_at)
        )
        return {
            "enabled": self.config.enabled,
            "active": self._direction != 0,
            "direction": (
                "right" if self._direction > 0 else "left" if self._direction < 0 else None
            ),
            "holding": self._hold_started_at is not None,
            "hold_age_seconds": hold_age,
            "maximum_hold_seconds": self.config.maximum_hold_seconds,
            "progress_age_seconds": progress_age,
            "progress_timeout_seconds": self.config.progress_timeout_seconds,
            "best_heading_magnitude": self._best_heading_magnitude,
            "best_lateral_magnitude": self._best_lateral_magnitude,
            "apex_active": self._apex_active(current_time),
            "apex_age_seconds": apex_age,
            "apex_commit_delay_seconds": self.config.apex_commit_delay_seconds,
            "apex_trigger_reason": self._apex_trigger_reason,
            "apex_completion_reason": self._apex_completion_reason,
            "minimum_apex_seconds": self.config.minimum_apex_seconds,
            "maximum_apex_seconds": self.config.maximum_apex_seconds,
            "apex_exit_valid_count": self._apex_exit_valid_count,
            "apex_exit_valid_frames": self.config.apex_exit_valid_frames,
            "exit_valid_count": self._exit_valid_count,
            "last_reason": self._last_reason,
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
