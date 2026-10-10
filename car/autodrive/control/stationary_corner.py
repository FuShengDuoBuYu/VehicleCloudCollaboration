"""Pure current-observation corner proposals; no hardware or model imports.

The caller must apply the final safety veto, map proposals to verified wheel
commands, and feed back vetoes through ``local_valid``.  ``stop`` means zero on
both wheels.  The dwell times assume that zero was actually applied; they do
not prove physical standstill.  IMU timestamps and ``now`` share a monotonic
clock.  The default yaw sign is unverified and the controller is disabled by
default.  Neither the image-space entry check nor yaw feedback proves body
clearance from a lane boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional


def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


@dataclass(frozen=True)
class StationaryCornerConfig:
    enabled: bool = False
    visual_feedback: bool = False
    continuous_cruise: bool = False
    visual_corner_guard: bool = False
    visual_path_margin_ratio: float = .06
    step_max_seconds: float = .15
    step_max_yaw_rad: float = math.radians(5)
    step_settle_seconds: float = .7
    visual_pivot_heading: float = .18
    front_near_ratio: float = .74
    front_max_ratio: float = .85
    semantic_max_age_seconds: float = .3
    enter_observations: int = 2
    settle_seconds: float = .25
    target_yaw_rad: float = math.pi / 2
    yaw_sign: int = -1
    imu_max_age_seconds: float = .2
    progress_timeout_seconds: float = .8
    max_pivot_seconds: float = 10.
    minimum_progress_rad: float = math.radians(1.)
    reverse_tolerance_rad: float = math.radians(8.)
    maximum_yaw_step_rad: float = math.radians(45.)
    exit_observations: int = 3

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be a bool')
        if type(self.visual_feedback) is not bool:
            raise ValueError('visual_feedback must be a bool')
        if type(self.continuous_cruise) is not bool or self.continuous_cruise and not self.visual_feedback:
            raise ValueError('continuous_cruise requires visual_feedback=true')
        if type(self.visual_corner_guard) is not bool or self.visual_corner_guard and not self.visual_feedback:
            raise ValueError('visual_corner_guard requires visual_feedback=true')
        if (not _finite_number(self.visual_path_margin_ratio)
                or not 0 < self.visual_path_margin_ratio <= .12):
            raise ValueError('visual path margin must be in (0,.12] of image width')
        if self.visual_feedback:
            from .visual_feedback import VisualFeedbackController
            VisualFeedbackController(self.step_max_seconds, self.step_max_yaw_rad,
                self.step_settle_seconds, self.semantic_max_age_seconds,
                self.imu_max_age_seconds, self.yaw_sign)
            if not _finite_number(self.visual_pivot_heading) or not 0 < self.visual_pivot_heading < 1:
                raise ValueError('visual_pivot_heading must be in (0,1)')
        if type(self.yaw_sign) is not int or self.yaw_sign not in (-1, 1):
            raise ValueError('yaw_sign must be -1 or 1, verified before use')
        for key, minimum in (('enter_observations', 2), ('exit_observations', 3)):
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        for key in ('front_near_ratio', 'front_max_ratio',
                    'semantic_max_age_seconds', 'settle_seconds',
                    'target_yaw_rad', 'imu_max_age_seconds',
                    'progress_timeout_seconds', 'max_pivot_seconds',
                    'minimum_progress_rad', 'reverse_tolerance_rad',
                    'maximum_yaw_step_rad'):
            value = getattr(self, key)
            if not _finite_number(value) or value <= 0:
                raise ValueError(f'{key} must be a positive finite number')
        if not self.front_near_ratio <= self.front_max_ratio <= .85:
            raise ValueError('front ratios must satisfy near <= max <= .85')
        if self.settle_seconds < .25:
            raise ValueError('settle_seconds must be >= .25')
        if self.imu_max_age_seconds > .2:
            raise ValueError('imu_max_age_seconds must be <= .2')
        if self.max_pivot_seconds > 10.:
            raise ValueError('max_pivot_seconds must be <= 10')
        if self.progress_timeout_seconds > self.max_pivot_seconds:
            raise ValueError('progress timeout must not exceed total timeout')
        if not self.minimum_progress_rad < self.target_yaw_rad <= math.pi / 2:
            raise ValueError('yaw target must exceed progress threshold and be <= pi/2')
        if not self.reverse_tolerance_rad < self.maximum_yaw_step_rad <= math.pi / 2:
            raise ValueError('yaw step bound must exceed reverse tolerance and be <= pi/2')


@dataclass(frozen=True)
class StationaryCornerDecision:
    state: str
    action: str
    reason: str
    yaw_progress_rad: float
    remaining_yaw_rad: float
    paused: bool


class StationaryCornerController:
    """One controller instance per run; ``blocked`` requires an explicit reset.

    Recreating the instance is the reset, and must only be done while stopped.
    Repeated reads of a semantic/IMU sample never create new evidence.  A short
    semantic failure pauses an active pivot, preserving its yaw and wall-clock
    deadline; only a newer fresh semantic observation can resume it.
    """

    def __init__(self, config: Optional[StationaryCornerConfig] = None):
        self.config = config if config is not None else StationaryCornerConfig()
        self._state = 'cruise'
        self._last_now = None
        self._last_sequence = None
        self._enter_count = 0
        self._exit_count = 0
        self._settle_started = None
        self._pivot_started = None
        self._last_imu_timestamp = None
        self._last_yaw = None
        self._yaw_progress = 0.
        self._best_progress = 0.
        self._progress_checkpoint = 0.
        self._active_seconds = 0.
        self._last_progress_active_seconds = 0.
        self._paused = False
        self._pause_sequence = None
        self._armed = True
        self._decision = self._result('passthrough', 'disabled' if not self.config.enabled
                                      else 'no-current-corner')

    def _result(self, action, reason):
        decision = StationaryCornerDecision(
            state=self._state, action=action, reason=reason,
            yaw_progress_rad=self._yaw_progress,
            remaining_yaw_rad=max(0., self.config.target_yaw_rad - self._yaw_progress),
            paused=self._paused)
        self._decision = decision
        return decision

    def _block(self, reason):
        self._state = 'blocked'
        self._paused = False
        return self._result('stop', reason)

    def get_state(self):
        """JSON-serializable diagnostics for the caller's existing run log."""
        result = asdict(self._decision)
        result.update(enabled=self.config.enabled, armed=self._armed,
                      entry_observations=self._enter_count,
                      exit_observations=self._exit_count,
                      last_semantic_sequence=self._last_sequence,
                      last_imu_timestamp=self._last_imu_timestamp,
                      pivot_started=self._pivot_started,
                      pivot_active_seconds=self._active_seconds)
        return result

    def _semantic_status(self, sequence, age):
        valid_sequence = type(sequence) is int and sequence >= 0
        previous = self._last_sequence
        ordered = valid_sequence and (previous is None or sequence >= previous)
        is_new = ordered and (previous is None or sequence > previous)
        # Consume even a stale/invalid observation so that refreshing its age
        # later cannot make the same source frame a new confirmation.
        if is_new:
            self._last_sequence = sequence
        fresh = (ordered and _finite_number(age)
                 and 0. <= age <= self.config.semantic_max_age_seconds)
        return fresh, is_new

    def _imu_fault(self, now, yaw, timestamp, valid):
        if valid is not True or not _finite_number(yaw) or not _finite_number(timestamp):
            return 'imu-invalid'
        age = now - timestamp
        if age < 0. or age > self.config.imu_max_age_seconds:
            return 'imu-not-current'
        if self._last_imu_timestamp is not None:
            if timestamp < self._last_imu_timestamp:
                return 'imu-time-regressed'
            if timestamp == self._last_imu_timestamp:
                delta = math.atan2(math.sin(yaw - self._last_yaw),
                                   math.cos(yaw - self._last_yaw))
                if abs(delta) > 1e-12:
                    return 'imu-duplicate-changed'
        return None

    def _pause(self, reason):
        self._enter_count = self._exit_count = 0
        if self._state == 'pivot-right':
            self._paused = True
            self._pause_sequence = self._last_sequence
        elif self._state == 'settle':
            self._state = 'approach'
            self._settle_started = None
        return self._result('stop', reason)

    def veto(self, reason):
        """Apply a recorded final execution veto without consuming a new sample.

        A paused pivot keeps its angle/deadline and requires a newer semantic
        observation to resume. The caller must actually apply zero output.
        """
        return self._pause(reason)

    def _update_pivot(self, now, yaw, timestamp, valid):
        fault = self._imu_fault(now, yaw, timestamp, valid)
        if fault:
            return self._block(fault)
        # A late sample must not renew an allowance that already expired.
        if now - self._pivot_started >= self.config.max_pivot_seconds:
            return self._block('pivot-total-timeout')
        if self._active_seconds - self._last_progress_active_seconds >= self.config.progress_timeout_seconds:
            return self._block('yaw-no-progress')
        if timestamp > self._last_imu_timestamp:
            delta = math.atan2(math.sin(yaw - self._last_yaw),
                               math.cos(yaw - self._last_yaw)) * self.config.yaw_sign
            if abs(delta) > self.config.maximum_yaw_step_rad:
                return self._block('imu-yaw-jump')
            self._yaw_progress += delta
            self._last_yaw, self._last_imu_timestamp = yaw, timestamp
            self._best_progress = max(self._best_progress, self._yaw_progress)
            if self._best_progress - self._yaw_progress > self.config.reverse_tolerance_rad:
                return self._block('yaw-reversed')
            if self._yaw_progress - self._progress_checkpoint >= self.config.minimum_progress_rad:
                self._progress_checkpoint = self._yaw_progress
                self._last_progress_active_seconds = self._active_seconds
        if self._yaw_progress >= self.config.target_yaw_rad - 1e-12:
            self._state = 'exit'
            self._armed = False
            self._paused = False
            self._exit_count = 0
            self._settle_started = now
            return self._result('stop', 'yaw-target-reached')
        return None

    def update(self, *, now, semantic_sequence=None, semantic_age_seconds=None,
               front_boundary_ratio=None, right_exit_observed=False,
               yaw_rad=None, imu_timestamp=None, imu_valid=False,
               exit_aligned=False, local_valid=True):
        """Return passthrough/straight/stop/pivot-right using supplied evidence.

        ``front_boundary_ratio`` is y/image-height of a true current semantic
        transverse boundary, not a synthetic crop/mask edge. ``exit_aligned``
        describes the next straight road in that same current semantic frame.
        ``local_valid`` is a caller veto: false always yields zero when enabled.
        It must not be replaced with truthiness from a non-boolean field.
        """
        if not self.config.enabled:
            return self._result('passthrough', 'disabled')
        if self._state == 'blocked':
            return self._decision
        if not _finite_number(now) or now < 0. or (self._last_now is not None and now < self._last_now):
            return self._block('controller-time-invalid')
        if self._last_now is not None and self._decision.action == 'pivot-right':
            self._active_seconds += now - self._last_now
        self._last_now = now
        fresh, is_new = self._semantic_status(semantic_sequence, semantic_age_seconds)

        # Track the real angle even while stopped for camera loss.  A reached
        # target latches zero immediately and cannot be undone by later images.
        if self._state == 'pivot-right':
            outcome = self._update_pivot(now, yaw_rad, imu_timestamp, imu_valid)
            if outcome is not None:
                return outcome
        if not fresh or local_valid is not True:
            return self._pause('semantic-not-current' if not fresh else 'local-veto')

        if self._state == 'pivot-right':
            if self._paused:
                if not is_new or (self._pause_sequence is not None
                                  and semantic_sequence <= self._pause_sequence):
                    return self._result('stop', 'await-new-semantic')
                self._paused = False
            return self._result('pivot-right', 'yaw-target-pending')

        if self._state == 'exit':
            if exit_aligned is not True:
                self._exit_count = 0
            elif is_new:
                self._exit_count += 1
            if (self._exit_count >= self.config.exit_observations
                    and now - self._settle_started >= self.config.settle_seconds):
                self._state = 'cruise'
                return self._result('passthrough', 'next-straight-confirmed')
            return self._result('stop', 'await-next-straight')

        if not self._armed:
            if front_boundary_ratio is None and is_new:
                self._armed = True
            else:
                return self._result('stop', 'await-corner-departure')

        if front_boundary_ratio is None:
            if self._state in ('approach', 'settle'):
                # An observed corner remains a pending maneuver. Missing its
                # cue is uncertainty, not evidence that an early arc is safe.
                self._state = 'approach'
                self._enter_count = 0
                self._settle_started = None
                return self._result('stop', 'await-current-front-boundary')
            self._state = 'cruise'
            self._enter_count = 0
            self._settle_started = None
            return self._result('passthrough', 'no-current-corner')
        if (not _finite_number(front_boundary_ratio)
                or not 0. <= front_boundary_ratio <= self.config.front_max_ratio):
            self._state = 'approach'
            self._enter_count = 0
            self._settle_started = None
            return self._result('stop', 'front-boundary-out-of-range')
        if front_boundary_ratio < self.config.front_near_ratio:
            self._state = 'approach'
            self._enter_count = 0
            self._settle_started = None
            return self._result('straight', 'approach-current-front')
        if right_exit_observed is not True:
            self._state = 'approach'
            self._enter_count = 0
            self._settle_started = None
            return self._result('stop', 'right-exit-unconfirmed')
        if self._state != 'settle':
            self._state = 'approach'
            if is_new:
                self._enter_count += 1
            if self._enter_count >= self.config.enter_observations:
                self._state = 'settle'
                self._settle_started = now
            return self._result('stop', 'settle-before-pivot' if self._state == 'settle'
                                else 'confirm-current-corner')
        if now - self._settle_started < self.config.settle_seconds:
            return self._result('stop', 'settle-before-pivot')
        fault = self._imu_fault(now, yaw_rad, imu_timestamp, imu_valid)
        if fault:
            return self._block(fault)
        self._state = 'pivot-right'
        self._pivot_started = now
        self._last_yaw, self._last_imu_timestamp = yaw_rad, imu_timestamp
        self._yaw_progress = self._best_progress = self._progress_checkpoint = 0.
        self._active_seconds = self._last_progress_active_seconds = 0.
        self._paused = False
        return self._result('pivot-right', 'current-corner-confirmed')
