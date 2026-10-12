"""Deterministic local motion adapter shared by live control and recorded replay."""
from __future__ import annotations

from autodrive.control.lane_centering import DifferentialDriveCommand
from autodrive.control.stationary_corner import StationaryCornerController
from dataclasses import replace
import math
import numpy as np
import time


STATIONARY_BOUNDARY_DIAGNOSTICS = (
    'semantic_front_observed', 'semantic_observed_front_ratio',
    'semantic_front_spread_ratio', 'semantic_right_exit_support_ratio',
    'semantic_corner_reject_reason', 'semantic_pivot_road_support_ratio',
    'semantic_track_surface_pixels','semantic_turn_entry_ready','semantic_exit_heading_error',
    'semantic_forward_road_valid','semantic_rotation_yellow_ratio','semantic_rotation_heading_error',
)


VISUAL_CORNER_DIAGNOSTICS = (
    'visual_corner_active', 'visual_corner_phase', 'visual_corner_exit_count',
    'visual_corner_reason', 'visual_path_margin_px', 'visual_required_margin_px',
    'visual_forward_clear',
    'visual_rotation_margin_px',
    'visual_rotation_required_margin_px','visual_rotation_origin_x_px','visual_rotation_origin_y_px',
    'visual_turn_margin_loss_px','visual_turn_margin_safe',
)


STATIONARY_LOG_FIELDS = (
    'stationary_state_before', 'stationary_phase', 'stationary_yaw_progress',
    'stationary_decision', 'stationary_reason', 'stationary_startup_ready',
    'stationary_pivot_verified', 'imu_yaw_rad', 'imu_received_monotonic',
    'imu_valid', 'imu_stale', 'imu_age_s', 'control_monotonic', 'frame_age_exact_s',
    'stationary_external_allowed', 'stationary_perception_budget_valid',
    'stationary_application_allowed',
    'stationary_continuous', 'stationary_motion_deadline',
    'semantic_front_boundary_ratio', 'semantic_right_exit_observed', 'semantic_hard_safe',
    *STATIONARY_BOUNDARY_DIAGNOSTICS,
    *VISUAL_CORNER_DIAGNOSTICS,
)


class StationaryCornerRuntime:
    """Deterministic adapter shared by live control and recorded-input replay.

    No clock or device reads occur here. ``external_motion_allowed`` feeds the
    current recording/watchdog gates into the controller, independent of cloud
    waiting. Cloud filtering remains a separate final gate. A dry-run
    replay may set pivot_verified=True; physical callers must validate evidence.
    """
    def __init__(self, config, motion_gate, *, pivot_verified=False):
        if config.visual_feedback:
            from autodrive.control.visual_feedback import VisualFeedbackController
            self.controller = VisualFeedbackController(config.step_max_seconds, config.step_max_yaw_rad,
                config.step_settle_seconds, config.semantic_max_age_seconds,
                config.imu_max_age_seconds, config.yaw_sign)
        else:
            self.controller = StationaryCornerController(config)
        self.config = config
        self.motion_gate = motion_gate
        self.pivot_verified = pivot_verified is True
        self._startup_ready = False
        self._state = {}
        self._visual_corner_active = False
        self._visual_exit_count = 0
        self._visual_exit_capture = None
        self._visual_last_turn_at = None
        self._visual_turn_start_margin = None
        self._visual_turn_margin_loss = 0.
        self._visual_exit_released_this_cycle = False

    @staticmethod
    def _finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    @staticmethod
    def _stop(reason):
        return DifferentialDriveCommand('stop', 0., 0., 0., 0., reason, 0.)

    def filter(self, command, estimate, boundary, *, now, imu_sample=None,
               frame_age_seconds=0., perception_budget_valid=True, external_motion_allowed=True):
        if not self.config.enabled:
            return self.motion_gate.filter(command, estimate, boundary)
        if self.config.visual_feedback:
            return self._filter_visual(command, estimate, boundary, now=now, imu_sample=imu_sample,
                frame_age_seconds=frame_age_seconds, perception_budget_valid=perception_budget_valid,
                external_motion_allowed=external_motion_allowed)
        controller_state = self.controller.get_state()
        before = controller_state['state']
        sequence = getattr(boundary, 'semantic_sequence', None)
        previous_sequence = controller_state['last_semantic_sequence']
        new_observation = (type(sequence) is int and sequence >= 0
                           and (previous_sequence is None or sequence > previous_sequence))
        sample = imu_sample if isinstance(imu_sample, dict) else {}
        value = sample.get('value') or {}
        angles = value.get('radians') if isinstance(value, dict) else None
        yaw = angles[2] if isinstance(angles, (list, tuple)) and len(angles) == 3 else None
        yaw = yaw if self._finite(yaw) else None
        receipt = sample.get('received_monotonic')
        receipt = receipt if self._finite(receipt) else None
        age = sample.get('age_s')
        age = age if self._finite(age) else None
        imu_valid = (sample.get('valid') is True and sample.get('stale') is False
                     and yaw is not None and receipt is not None and age is not None
                     and 0 <= age <= self.config.imu_max_age_seconds
                     and self._finite(now) and 0 <= now-receipt <= self.config.imu_max_age_seconds)
        semantic_age = getattr(boundary, 'semantic_result_age_seconds', None)
        semantic_captured_at = getattr(boundary, 'semantic_captured_at', None)
        if semantic_captured_at is not None:
            semantic_age = (now-semantic_captured_at
                if self._finite(now) and self._finite(semantic_captured_at) else None)
            # The same refreshed age is persisted for deterministic replay.
            boundary.semantic_result_age_seconds = semantic_age
        hard_safe = (getattr(boundary, 'semantic_hard_safe', False) is True
                     and self._finite(semantic_age) and 0 <= semantic_age <= self.config.semantic_max_age_seconds
                     and self._finite(frame_age_seconds) and 0 <= frame_age_seconds <= .25
                     and perception_budget_valid is True)
        ordinary_safe = (hard_safe and estimate.valid is True and boundary.valid is True
                         and command.action != 'stop')
        ordinary = command if ordinary_safe else self._stop('stationary corner: ordinary path unsafe')
        active_pivot = before == 'pivot-right' and self.pivot_verified
        if not active_pivot and (not ordinary_safe or external_motion_allowed is not True):
            self.motion_gate.reset('stationary corner: current permission vetoed')
            self._startup_ready = False
        elif before == 'exit':
            # Exit alignment and ordinary recovery are separate evidence
            # windows: three fresh exit observations precede five new ordinary
            # observations. No exit frame is banked toward ordinary recovery.
            self.motion_gate.reset('stationary corner: awaiting exit confirmation')
            self._startup_ready = False
        elif not active_pivot:
            if self.motion_gate.get_state()['ready'] or new_observation:
                ordinary = self.motion_gate.filter(ordinary, estimate, boundary)
            else:
                ordinary = self._stop('stationary corner: waiting for a new ordinary observation')
            self._startup_ready = self.motion_gate.get_state()['ready']
        local_safe = hard_safe and (imu_valid if active_pivot else
                                   ordinary_safe and (before == 'exit' or self._startup_ready))
        local_valid = bool(local_safe and external_motion_allowed is True
                           and (self.pivot_verified or before not in ('settle', 'pivot-right')))
        decision = self.controller.update(now=now,
            semantic_sequence=getattr(boundary, 'semantic_sequence', None), semantic_age_seconds=semantic_age,
            front_boundary_ratio=getattr(boundary, 'semantic_front_boundary_ratio', None),
            right_exit_observed=getattr(boundary, 'semantic_right_exit_observed', False),
            yaw_rad=yaw, imu_timestamp=receipt, imu_valid=bool(imu_valid),
            exit_aligned=bool(ordinary_safe and abs(estimate.heading_error) <= .18
                              and abs(estimate.lateral_error) <= .35), local_valid=local_valid)
        if decision.action == 'pivot-right' and self.pivot_verified and local_valid:
            self.motion_gate.reset('stationary pivot; ordinary recovery must be revalidated')
            self._startup_ready = False
            output = DifferentialDriveCommand('pivot-right', 1., .3, -.3, 1., decision.reason, 0.)
        elif decision.action == 'stop' or not local_valid:
            # Controller-generated zero during corner confirmation/settle is
            # expected. It does not revoke current ordinary-path evidence.
            # An active pivot keeps its own yaw/deadline across a short veto.
            output = self._stop('stationary corner: ' + decision.reason)
        elif before == 'exit':
            output = self._stop('stationary corner: ' + decision.reason
                                + '; ordinary recovery must be revalidated')
        elif decision.action == 'straight':
            if ordinary.action == 'stop':
                output = ordinary
            else:
                speed = max(0., min(command.left_speed, command.right_speed))
                output = DifferentialDriveCommand('forward', 0., speed, speed, command.confidence, decision.reason, 0.)
        else:
            output = ordinary
        self._state = dict(
            stationary_state_before=before, stationary_phase=decision.state,
            stationary_yaw_progress=decision.yaw_progress_rad, stationary_decision=decision.action,
            stationary_reason=decision.reason, stationary_startup_ready=self._startup_ready,
            stationary_pivot_verified=self.pivot_verified, imu_yaw_rad=yaw,
            imu_received_monotonic=receipt, imu_valid=bool(imu_valid),
            imu_stale=sample.get('stale') is not False, imu_age_s=age,
            control_monotonic=now, frame_age_exact_s=frame_age_seconds,
            stationary_external_allowed=external_motion_allowed is True,
            stationary_perception_budget_valid=perception_budget_valid is True,
            stationary_application_allowed=True,
            semantic_front_boundary_ratio=getattr(boundary, 'semantic_front_boundary_ratio', None),
            semantic_right_exit_observed=getattr(boundary, 'semantic_right_exit_observed', False),
            semantic_hard_safe=getattr(boundary, 'semantic_hard_safe', False), local_safe=bool(local_safe))
        self._state.update({field: getattr(boundary, field, None)
                            for field in STATIONARY_BOUNDARY_DIAGNOSTICS})
        return output

    def _filter_visual(self, command, estimate, boundary, *, now, imu_sample,
                       frame_age_seconds, perception_budget_valid, external_motion_allowed):
        self._visual_exit_released_this_cycle = False
        before = self.controller.get_state()
        sequence = getattr(boundary, 'semantic_sequence', None)
        captured = getattr(boundary, 'semantic_captured_at', None)
        age = now-captured if self._finite(now) and self._finite(captured) else None
        if captured is None:
            age = getattr(boundary, 'semantic_result_age_seconds', None)
            captured = now-age if self._finite(age) and self._finite(now) else None
        sample = imu_sample if isinstance(imu_sample, dict) else {}
        angles = (sample.get('value') or {}).get('radians')
        yaw = angles[2] if isinstance(angles, (list, tuple)) and len(angles)==3 else None
        stamp = sample.get('received_monotonic')
        imu_valid = (sample.get('valid') is True and sample.get('stale') is False
            and self._finite(yaw) and self._finite(stamp) and self._finite(now)
            and 0 <= now-stamp <= self.config.imu_max_age_seconds)
        evidence_safe = (getattr(boundary, 'semantic_hard_safe', False) is True
            and boundary.valid is True
            and self._finite(age) and 0 <= age <= self.config.semantic_max_age_seconds
            and self._finite(frame_age_seconds) and 0 <= frame_age_seconds <= .25
            and perception_budget_valid is True and external_motion_allowed is True
            and self._finite(estimate.heading_error) and self._finite(estimate.lateral_error)
            and self._finite(estimate.near_heading_error))
        new = (type(sequence) is int and (before['last_semantic_sequence'] is None
                                        or sequence > before['last_semantic_sequence']))
        desired = command.action
        near_aligned = (abs(estimate.near_heading_error) <= self.config.visual_pivot_heading
                        and abs(estimate.lateral_error) <= .35)
        front_observed = getattr(boundary, 'semantic_front_observed', False) is True
        front_ratio = getattr(boundary, 'semantic_observed_front_ratio', None)
        # A far preview reaching the planner lookahead is not an in-place
        # turning location. Approach using current visible front distance.
        front_near = front_observed and self._finite(front_ratio) and front_ratio >= self.config.front_near_ratio
        exit_heading=getattr(boundary,'semantic_exit_heading_error',None)
        rotation_heading=getattr(boundary,'semantic_rotation_heading_error',None)
        rotation_candidate=bool(self.config.visual_corner_guard
            and getattr(boundary,'semantic_track_surface_pixels',0)>0
            and self.pivot_verified and imu_valid and
            (front_near and getattr(boundary,'semantic_right_exit_observed',False) is True
             and self._finite(exit_heading) and exit_heading>self.config.visual_pivot_heading
             or self._visual_corner_active and self._finite(rotation_heading)
             and rotation_heading>self.config.visual_pivot_heading))
        local_safe=bool(evidence_safe and (rotation_candidate or
            estimate.valid is True and command.action!='stop'
            and getattr(boundary,'semantic_forward_road_valid',True) is True))
        cruise = bool(self.config.continuous_cruise and near_aligned and not front_near
                      and self._finite(command.steering) and abs(command.steering) <= .35
                      and command.action in ('forward', 'turn-left', 'turn-right'))
        clearance = {};rotation={};phase='cruise';corner_reason='guard disabled';exit_confirm=False
        forward_in_corner=False;turn_margin_safe=True
        if self.config.visual_corner_guard:
            from autodrive.perception.visual_clearance import assess_visible_clearance,assess_rotation_clearance
            clearance=assess_visible_clearance(getattr(boundary,'corridor_mask',None),
                getattr(boundary,'semantic_yellow_mask',None),estimate.centerline,
                self.config.visual_path_margin_ratio)
            # Exit confirmation must obey the controller's global sequence,
            # capture and clock ordering before changing corner memory.
            ordered=(type(sequence) is int and sequence>=0 and
                (before['last_semantic_sequence'] is None or sequence>=before['last_semantic_sequence']))
            capture_ordered=(self._finite(captured) and
                (before['last_semantic_captured_at'] is None or captured>=before['last_semantic_captured_at']))
            clock_ordered=(self._finite(now) and now>=0 and
                (before['last_control_monotonic'] is None or now>=before['last_control_monotonic']))
            rotation=assess_rotation_clearance(getattr(boundary,'semantic_rotation_mask',None),
                getattr(boundary,'semantic_yellow_mask',None))
            # When forward support actually ends, a verified nearby exit can
            # require rotation before the normal approach distance is reached.
            # Check a straight trace independently of a failed fitted path.
            rotation_mask=getattr(boundary,'semantic_rotation_mask',None)
            if (not rotation_candidate and self.pivot_verified and imu_valid
                    and front_observed and getattr(boundary,'semantic_turn_entry_ready',False) is True
                    and getattr(boundary,'semantic_right_exit_observed',False) is True
                    and self._finite(exit_heading) and exit_heading>self.config.visual_pivot_heading
                    and rotation['rotation_clear'] and rotation_mask is not None):
                h,w=rotation_mask.shape
                straight=assess_visible_clearance(rotation_mask,getattr(boundary,'semantic_yellow_mask',None),
                    np.array([[w//2,int(h*.98)],[w//2,int(h*.78)]]),self.config.visual_path_margin_ratio)
                if straight['minimum_margin_px'] is not None and not straight['forward_clear']:
                    rotation_candidate=True;local_safe=bool(evidence_safe)
            margin=rotation.get('minimum_margin_px');required=rotation.get('required_margin_px')
            if (new and ordered and capture_ordered and clock_ordered
                    and self._finite(captured) and self._visual_last_turn_at is not None
                    and captured>self._visual_last_turn_at and self._finite(margin)
                    and self._visual_turn_start_margin is not None):
                # Reserve at least the visible margin consumed by the last
                # small turn. This is an image-space warning, not a body model.
                # Recompute on the current frame: one transient segmentation
                # gap must not latch its worst margin after support recovers.
                self._visual_turn_margin_loss=max(self._visual_turn_start_margin-margin,0.)
            if self._finite(margin) and self._finite(required):
                turn_margin_safe=margin-self._visual_turn_margin_loss>=required
            forward_in_corner=bool(self._visual_corner_active and clearance['path_safe']
                and clearance['forward_clear'] and estimate.valid is True and command.action!='stop'
                and getattr(boundary,'semantic_forward_road_valid',True) is True
                and (near_aligned or not turn_margin_safe))
            if forward_in_corner:
                rotation_candidate=False
                local_safe=bool(evidence_safe)
            local_safe=local_safe and (rotation['rotation_clear'] if rotation_candidate else clearance['path_safe']) and ordered and capture_ordered and clock_ordered
            if rotation_candidate and not turn_margin_safe:local_safe=False
            if rotation_candidate:cruise=False
            corner_reason=clearance['reason']
            # A briefly straight-looking patch after a pivot is not a corner
            # exit. Keep this memory across missing/expired evidence and vetoes.
            if self._visual_corner_active:
                exit_confirm=bool(local_safe and near_aligned and not front_observed
                                  and clearance['forward_clear'] and
                                  (not getattr(boundary,'semantic_track_surface_pixels',0)
                                   or abs(estimate.heading_error)<=self.config.visual_pivot_heading))
                stopped=before.get('stopped_at')
                exit_ready=(exit_confirm and before['action']=='stop' and self._finite(captured)
                    and self._finite(stopped) and self._visual_last_turn_at is not None
                    and captured>max(stopped,self._visual_last_turn_at))
                if not exit_ready:
                    self._visual_exit_count=0;self._visual_exit_capture=None
                elif new:
                    if self._visual_exit_capture is None or captured>self._visual_exit_capture:
                        self._visual_exit_count+=1;self._visual_exit_capture=captured
                    else:
                        self._visual_exit_count=0;self._visual_exit_capture=None
                    if self._visual_exit_count>=2:
                        self._visual_corner_active=False;exit_confirm=False
                        forward_in_corner=False
                        self._visual_turn_start_margin=None;self._visual_turn_margin_loss=0.
                        self._visual_exit_released_this_cycle=True
            cruise=cruise and not front_observed and not self._visual_corner_active
            phase=('turn' if self._visual_corner_active else
                   'approach' if front_observed else 'cruise')
            if exit_confirm:phase='exit-confirm'
        # A far exit can be visible while the near road is still straight.
        # Use the current near path to decide this short physical step.
        approach=(self.config.visual_corner_guard and front_observed
                  and not front_near and not self._visual_corner_active)
        if exit_confirm:
            desired='stop';corner_reason='visual-await-post-stop-exit-observations'
        elif forward_in_corner:
            desired='forward';cruise=False
            corner_reason=('visual-forward-preserve-turn-margin' if not turn_margin_safe
                           else 'visual-current-near-path-forward')
        elif rotation_candidate:
            desired='pivot-right';corner_reason=('visual-current-rotation-support' if turn_margin_safe
                                                else 'visual-turn-margin-shrinking')
        elif approach:
            desired='forward';local_safe=local_safe and clearance['forward_clear']
            if not clearance['forward_clear']:corner_reason='visual-forward-approach-not-clear'
        elif (estimate.near_heading_error > self.config.visual_pivot_heading
              or self.config.visual_corner_guard and (front_near or self._visual_corner_active and self._finite(exit_heading))
              and getattr(boundary,'semantic_right_exit_observed',False) is True
              and (self._finite(exit_heading) and exit_heading>self.config.visual_pivot_heading
                   or estimate.heading_error>self.config.visual_pivot_heading)):
            desired = 'pivot-right'
            local_safe = local_safe and self.pivot_verified and imu_valid
            if self.config.visual_corner_guard:
                turn_allowed=(front_near and getattr(boundary,'semantic_right_exit_observed',False) is True
                    or self._visual_corner_active and (getattr(boundary,'semantic_right_exit_observed',False) is True
                        or getattr(boundary,'semantic_track_surface_pixels',0)>0))
                local_safe=local_safe and turn_allowed
                if not turn_allowed:corner_reason='visual-turn-await-current-front-and-exit'
        elif near_aligned and not cruise:
            desired = 'forward'
            if self.config.visual_corner_guard:
                local_safe=local_safe and clearance['forward_clear']
                if not clearance['forward_clear']:corner_reason='visual-forward-step-not-clear'
        if not local_safe:
            self.motion_gate.reset('visual current path veto')
            self._startup_ready = False
        elif not self._startup_ready and new:
            gate_command=command;gate_estimate=estimate
            if rotation_candidate:
                gate_command=DifferentialDriveCommand('pivot-right',1.,.3,-.3,
                    getattr(boundary,'confidence',0.),'current rotation support',0.)
                gate_estimate=replace(estimate,valid=True,confidence=getattr(boundary,'confidence',0.),
                    lateral_error=0.,heading_error=exit_heading if self._finite(exit_heading) else rotation_heading)
            gated = self.motion_gate.filter(gate_command, gate_estimate, boundary)
            self._startup_ready = gated.action != 'stop' and self.motion_gate.get_state()['ready']
        decision = self.controller.update(now=now, semantic_sequence=sequence, captured_at=captured,
            desired_action=desired, local_valid=bool(local_safe and self._startup_ready),
            yaw_rad=yaw, imu_timestamp=stamp, imu_valid=bool(imu_valid), continuous=cruise)
        if self.config.visual_corner_guard and decision.action=='pivot-right':
            if before['action']!='pivot-right':
                self._visual_turn_start_margin=rotation.get('minimum_margin_px')
                self._visual_turn_margin_loss=0.
            self._visual_corner_active=True;self._visual_exit_count=0;self._visual_exit_capture=None
            self._visual_last_turn_at=now;phase='turn'
        if decision.action != 'stop' and self.controller.get_state()['continuous']:
            output = replace(command, reason=decision.reason, tight_turn_factor=0.)
        elif decision.action=='pivot-right':
            output = DifferentialDriveCommand('pivot-right', 1., .3, -.3, estimate.confidence, decision.reason, 0.)
        elif decision.action=='forward':
            speed = max(0., min(command.left_speed, command.right_speed))
            output = DifferentialDriveCommand('forward', 0., speed, speed, estimate.confidence, decision.reason, 0.)
        elif decision.action=='stop':
            output = self._stop(decision.reason)
        else:
            output = command
        self._state = dict(stationary_state_before=before['state'], stationary_phase=decision.state,
            stationary_yaw_progress=decision.yaw_progress_rad, stationary_decision=decision.action,
            stationary_reason=decision.reason, stationary_startup_ready=self._startup_ready,
            stationary_pivot_verified=self.pivot_verified, imu_yaw_rad=yaw,
            imu_received_monotonic=stamp, imu_valid=bool(imu_valid), imu_stale=sample.get('stale') is not False,
            imu_age_s=sample.get('age_s'), control_monotonic=now, frame_age_exact_s=frame_age_seconds,
            stationary_external_allowed=external_motion_allowed is True,
            stationary_perception_budget_valid=perception_budget_valid is True,
            stationary_application_allowed=True, local_safe=bool(local_safe), visual_feedback=True,
            stationary_continuous=self.controller.get_state()['continuous'],
            stationary_motion_deadline=self.controller.get_state()['motion_deadline'])
        self._state.update(visual_corner_active=self._visual_corner_active,
            visual_corner_phase=phase,visual_corner_exit_count=self._visual_exit_count,
            visual_corner_reason=corner_reason,visual_path_margin_px=clearance.get('minimum_margin_px'),
            visual_required_margin_px=clearance.get('required_margin_px'),
            visual_forward_clear=clearance.get('forward_clear'))
        self._state['visual_rotation_margin_px']=rotation.get('minimum_margin_px')
        self._state['visual_rotation_required_margin_px']=rotation.get('required_margin_px')
        self._state['visual_rotation_origin_x_px']=rotation.get('origin_x_px')
        self._state['visual_rotation_origin_y_px']=rotation.get('origin_y_px')
        self._state['visual_turn_margin_loss_px']=self._visual_turn_margin_loss
        self._state['visual_turn_margin_safe']=turn_margin_safe
        self._state.update({field: getattr(boundary, field, None) for field in STATIONARY_BOUNDARY_DIAGNOSTICS})
        self._state.update({field: getattr(boundary, field, None) for field in
            ('semantic_front_boundary_ratio', 'semantic_right_exit_observed', 'semantic_hard_safe')})
        return output

    def apply_veto(self, reason, now=None):
        """Record a final zero after filtering without another timed update.

        Live and replay use this explicit event when a previously gated
        observation expires before application. The controller retains its
        angle/deadline and requires a newer semantic frame to resume a pivot.
        """
        decision = (self.controller.veto(reason,now=now) if self.config.visual_feedback
                    else self.controller.veto(reason))
        self.motion_gate.reset(reason)
        self._startup_ready = False
        self._state.update(stationary_phase=decision.state,
            stationary_yaw_progress=decision.yaw_progress_rad,
            stationary_decision=decision.action, stationary_reason=decision.reason,
            stationary_startup_ready=False, stationary_application_allowed=False, local_safe=False)
        if self.config.visual_feedback:
            self._state.update(stationary_continuous=False, stationary_motion_deadline=None)
            if self.config.visual_corner_guard:
                # A proposal is not an applied exit. Discard confirmation if
                # the final application rejects this cycle, including a cycle
                # that provisionally released corner memory.
                if self._visual_exit_released_this_cycle:
                    self._visual_corner_active=True
                self._visual_exit_released_this_cycle=False
                self._visual_exit_count=0;self._visual_exit_capture=None
                self._state.update(visual_corner_active=self._visual_corner_active,
                    visual_corner_exit_count=0,
                    visual_corner_phase='turn' if self._visual_corner_active else 'cruise',
                    visual_corner_reason=reason)
        return self._stop(reason)

    def get_state(self):
        return dict(self._state, controller=self.controller.get_state(), motion_gate=self.motion_gate.get_state())


def apply_runtime_command(driver,command,stationary_runtime=None,recovery=None):
    """Preserve a driver's final rejection for logs and recorded replay."""
    if recovery is not None and recovery.motion_deadline is not None and command.action!='stop':
        state=driver.apply(command,deadline=recovery.motion_deadline)
    elif (stationary_runtime is not None and stationary_runtime.config.visual_feedback
            and command.action!='stop'):
        state=driver.apply(command,deadline=stationary_runtime.controller.get_state()['motion_deadline'])
    else:
        state=driver.apply(command)
    if stationary_runtime is not None and command.action!='stop' and state.get('action')=='stopped':
        command=stationary_runtime.apply_veto(state.get('reason') or 'driver rejected motion',
                                              now=time.monotonic())
    if recovery is not None and command.action=='stop' and recovery.motion_deadline is not None:
        recovery.veto(command.reason or 'final application stopped',time.monotonic())
    return command,state

