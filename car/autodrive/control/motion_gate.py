"""Fresh-observation motion admission and recovery gates."""
from __future__ import annotations

from .lane_centering import DifferentialDriveCommand
from .lane_centering import LaneEstimate
from typing import Optional


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
        self._last_semantic_sequence: Optional[int] = None
        self._last_lateral: Optional[float] = None
        self._last_heading: Optional[float] = None
        self._last_source: Optional[str] = None

    def reset(self, reason: str = "motion gate reset") -> None:
        self._ready = False
        self._consecutive_valid = 0
        self._last_stop_reason = str(reason)
        self._candidate_source = None
        self._last_semantic_sequence = None
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
        # A fast control loop can consume the same asynchronous mask many times.
        # Resuming requires distinct model observations, not repeated control ticks.
        sequence = getattr(boundary_result, "semantic_sequence", None)
        if sequence is not None:
            if sequence == self._last_semantic_sequence:
                return self._stop_command(command, "waiting for a new YOLOPv2 observation")
            self._last_semantic_sequence = sequence
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

