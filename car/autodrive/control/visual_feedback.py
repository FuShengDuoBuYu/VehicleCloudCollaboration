"""A visual intent is revocable every cycle; there is no corner angle target.

Fresh cruise frames renew a short permit; corner steps never renew their limit.
Limits bound one proposal, never describe the route. All clocks are supplied by
the caller. A zero-output dwell is conservative and does not prove standstill.
"""
from dataclasses import asdict, dataclass
import math


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class VisualDecision:
    state: str
    action: str
    reason: str
    yaw_progress_rad: float = 0.
    remaining_yaw_rad: float = 0.
    paused: bool = False


class VisualFeedbackController:
    ACTIONS = {'forward', 'turn-left', 'turn-right', 'pivot-right'}

    def __init__(self, max_step_seconds=.15, max_step_yaw_rad=math.radians(5),
                 settle_seconds=.7, semantic_max_age_seconds=.3,
                 imu_max_age_seconds=.2, yaw_sign=-1):
        for value in (max_step_seconds, max_step_yaw_rad, settle_seconds,
                      semantic_max_age_seconds, imu_max_age_seconds):
            if not finite(value) or value <= 0:
                raise ValueError('visual limits must be positive finite numbers')
        if (max_step_seconds > .25 or max_step_yaw_rad > math.radians(15)
                or settle_seconds < .25 or imu_max_age_seconds > .2
                or type(yaw_sign) is not int or yaw_sign not in (-1, 1)):
            raise ValueError('visual step limits exceed the experimental bounds')
        self.max_step_seconds = max_step_seconds
        self.max_step_yaw_rad = max_step_yaw_rad
        self.settle_seconds = settle_seconds
        self.semantic_max_age_seconds = semantic_max_age_seconds
        self.imu_max_age_seconds = imu_max_age_seconds
        self.yaw_sign = yaw_sign
        self._now = None
        self._sequence = None
        self._capture = None
        self._intent = None
        self._continuous = False
        self._deadline = None
        self._settle_required = True
        self._started = None
        self._stopped = None
        self._last_yaw = None
        self._imu_stamp = None
        self._yaw = 0.
        self._decision = VisualDecision('observe', 'stop', 'await-current-visual-intent')

    def _result(self, action, reason):
        state = action if action != 'stop' else ('settle' if self._stopped is not None else 'observe')
        self._decision = VisualDecision(state, action, reason, self._yaw)
        return self._decision

    def _stop(self, reason, *, settle=None):
        if self._intent is not None:
            self._stopped = self._now
            self._settle_required = not self._continuous if settle is None else settle
        self._intent = None
        self._continuous = False
        self._deadline = None
        self._started = None
        return self._result('stop', reason)

    def veto(self, reason, now=None):
        if now is not None and finite(now) and (self._now is None or now>=self._now):
            self._now=now
        return self._stop(reason)

    def get_state(self):
        return dict(asdict(self._decision), enabled=True, visual_feedback=True,
                    last_semantic_sequence=self._sequence,
                    last_semantic_captured_at=self._capture, last_control_monotonic=self._now,
                    step_started=self._started,
                    motion_deadline=self._deadline, continuous=self._continuous,
                    stopped_at=self._stopped, last_imu_timestamp=self._imu_stamp)

    def update(self, *, now, semantic_sequence, captured_at, desired_action,
               local_valid, yaw_rad=None, imu_timestamp=None, imu_valid=False,
               continuous=False):
        if not finite(now) or now < 0 or (self._now is not None and now < self._now):
            return self._stop('visual-clock-invalid')
        self._now = now
        ordered = (type(semantic_sequence) is int and semantic_sequence >= 0
                   and (self._sequence is None or semantic_sequence >= self._sequence))
        is_new = ordered and (self._sequence is None or semantic_sequence > self._sequence)
        capture_valid = (finite(captured_at) and 0 <= now-captured_at <= self.semantic_max_age_seconds
                         and (self._capture is None or captured_at >= self._capture))
        if is_new:
            self._sequence = semantic_sequence
            if capture_valid:
                self._capture = captured_at
        if not ordered or not capture_valid or local_valid is not True:
            return self._stop('visual-current-evidence-veto')
        if (desired_action not in self.ACTIONS or type(continuous) is not bool
                or continuous and desired_action == 'pivot-right'):
            return self._stop('visual-stop-or-unknown-intent')
        if self._intent is not None:
            if now >= self._deadline:
                return self._stop('visual-cruise-permit-expired' if self._continuous
                                  else 'visual-step-time-ceiling')
            if continuous != self._continuous:
                return self._stop('visual-motion-mode-changed', settle=True)
            if desired_action != self._intent and not self._continuous:
                return self._stop('visual-intent-changed')
        if desired_action == 'pivot-right':
            if (imu_valid is not True or not finite(yaw_rad) or not finite(imu_timestamp)
                    or not 0 <= now-imu_timestamp <= self.imu_max_age_seconds
                    or self._intent is not None and self._imu_stamp is not None
                    and imu_timestamp < self._imu_stamp):
                return self._stop('visual-imu-not-current')
            if self._intent is not None:
                delta = math.atan2(math.sin(yaw_rad-self._last_yaw),
                                   math.cos(yaw_rad-self._last_yaw)) * self.yaw_sign
                if (abs(delta) > math.radians(45) or delta < -math.radians(8)
                        or imu_timestamp == self._imu_stamp and abs(delta) > 1e-12):
                    return self._stop('visual-imu-inconsistent')
                self._yaw += delta
                self._last_yaw, self._imu_stamp = yaw_rad, imu_timestamp
                if self._yaw >= self.max_step_yaw_rad:
                    return self._stop('visual-step-yaw-ceiling')
        if self._intent is None:
            if not is_new:
                return self._result('stop', 'visual-await-new-frame')
            if self._stopped is not None:
                # Only cruise-to-cruise recovery can omit the corner dwell.
                # A permit expiry and a turn request may arrive together.
                needs_settle = self._settle_required or not continuous
                ready_at = self._stopped + (self.settle_seconds if needs_settle else 0.)
                if now < ready_at or captured_at <= ready_at:
                    return self._result('stop', 'visual-await-post-settle-frame')
            self._intent = desired_action
            self._continuous = continuous
            self._started = now
            self._deadline = now + self.max_step_seconds
            self._stopped = None
            self._yaw = 0.
            self._last_yaw, self._imu_stamp = yaw_rad, imu_timestamp
        if self._continuous and is_new:
            # Cached observations cannot extend motion. Source expiry is an
            # independent ceiling even if the latest proposal arrived late.
            self._deadline = min(now + .25, captured_at + self.semantic_max_age_seconds)
            self._intent = desired_action
        return self._result(self._intent, 'current-YOLO-cruise' if self._continuous
                            else 'current-YOLO-local-path')
