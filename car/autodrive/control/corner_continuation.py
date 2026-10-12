"""Historical bounded corner continuation, retained for Pi and replay compatibility."""
from __future__ import annotations

from .lane_centering import DifferentialDriveCommand
from .lane_centering import LaneEstimate
from dataclasses import dataclass
from typing import Optional
import time


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
        self._apex_both_valid_count = 0
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
            self._apex_both_valid_count = 0
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
            # Confirm the two kinds of recovery independently.  A single
            # false ``both`` frame followed by one merely aligned one-edge
            # frame must not be combined into an early apex exit.
            self._apex_both_valid_count = (
                self._apex_both_valid_count + 1 if both_visible else 0
            )
            self._apex_exit_valid_count = (
                self._apex_exit_valid_count + 1 if near_aligned else 0
            )

        if (
            self._apex_exit_valid_count >= self.config.apex_exit_valid_frames
            or self._apex_both_valid_count
            >= self.config.apex_exit_valid_frames
        ):
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

        # Enforce the bounded apex even when a recoverable LCC stop coincides
        # with one provisional ``both`` frame.  The raw yellow-hazard check at
        # the top of this method remains an unconditional immediate stop.
        provisional_both = bool(
            boundary_result is not None
            and bool(getattr(boundary_result, "valid", False))
            and not bool(getattr(boundary_result, "yellow_hazard", False))
            and str(getattr(boundary_result, "source", "")) == "both"
        )
        apex_active = bool(
            provisional_both
            and self._apex_active(
                current_time,
                estimate,
                boundary_result,
            )
        )
        if (
            apex_active
            and command.reason in self._RECOVERABLE_REASONS
            and self._last_command is not None
            and provisional_both
        ):
            held = self._last_command
            self._last_reason = (
                "continuing bounded apex through provisional recovery"
            )
            return DifferentialDriveCommand(
                held.action,
                held.steering,
                held.left_speed,
                held.right_speed,
                command.confidence,
                self._last_reason,
                1.0,
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
            "apex_both_valid_count": self._apex_both_valid_count,
            "apex_exit_valid_frames": self.config.apex_exit_valid_frames,
            "exit_valid_count": self._exit_valid_count,
            "last_reason": self._last_reason,
        }

