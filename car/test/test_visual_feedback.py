"""Current visual intent, bounded motion, and fresh observations; no devices."""
import importlib
import math
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.control.stationary_corner import StationaryCornerConfig
from autodrive.control.drive_runtime import PerceptionMotionGate, SafeWheelDriver, WheelMappingConfig
from autodrive.control.lane_centering import DifferentialDriveCommand, LaneEstimate
from autodrive.runtime.onboard import StationaryCornerRuntime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    visual = importlib.import_module('autodrive.control.visual_feedback')
except ModuleNotFoundError as exc:
    if exc.name != 'autodrive.control.visual_feedback':
        raise
    visual = None


class VisualFeedbackTests(unittest.TestCase):
    def test_fresh_cruise_observations_continue_and_update_gentle_direction(self):
        self.assertEqual(self.tick(1., 1, 'forward', continuous=True).action, 'forward')
        self.assertEqual(self.tick(1.1, 2, 'turn-right', continuous=True).action, 'turn-right')
        self.assertEqual(self.tick(1.2, 3, 'turn-left', continuous=True).action, 'turn-left')
        self.assertGreater(self.controller.get_state()['motion_deadline'], 1.4)

    def test_cached_cruise_frame_cannot_renew_permission(self):
        self.tick(1., 1, 'forward', continuous=True)
        deadline = self.controller.get_state()['motion_deadline']
        self.assertEqual(self.tick(1.1, 1, 'forward', continuous=True).action, 'forward')
        self.assertEqual(self.controller.get_state()['motion_deadline'], deadline)
        self.assertEqual(self.tick(1.251, 1, 'forward', continuous=True).action, 'stop')
        self.assertEqual(self.tick(1.26, 1, 'forward', continuous=True).action, 'stop')

    def test_cruise_expires_at_source_age_and_fresh_frame_after_stop_can_recover(self):
        self.tick(1., 1, 'forward', continuous=True, captured_at=.82)
        self.assertAlmostEqual(self.controller.get_state()['motion_deadline'], 1.12)
        self.assertEqual(self.tick(1.121, 2, 'forward', continuous=True).action, 'stop')
        self.assertEqual(self.tick(1.2, 3, 'forward', continuous=True).action, 'forward')
        self.assertEqual(self.tick(1.21, 4, 'forward', continuous=True, local_valid=False).action, 'stop')

    def test_cruise_to_corner_stops_before_a_short_pivot(self):
        self.tick(1., 1, 'forward', continuous=True)
        self.assertEqual(self.tick(1.1, 2).action, 'stop')
        self.assertEqual(self.tick(1.2, 3).action, 'stop')
        self.assertEqual(self.tick(1.82, 4).action, 'pivot-right')
        self.assertEqual(self.tick(1.98, 5).action, 'stop')

    def test_expired_cruise_cannot_skip_corner_settle_on_the_next_frame(self):
        self.controller=visual.VisualFeedbackController(settle_seconds=.25)
        self.tick(1.,1,'forward',continuous=True)
        self.assertEqual(self.tick(1.251,2).action,'stop')
        self.assertEqual(self.tick(1.27,3).action,'stop')
        self.assertEqual(self.tick(1.52,4,captured_at=1.49).action,'stop')
        self.assertEqual(self.tick(1.53,5,captured_at=1.52).action,'pivot-right')

    def test_far_bend_with_straight_near_path_does_not_trigger_stationary_pivot(self):
        runtime=StationaryCornerRuntime(StationaryCornerConfig(enabled=True,visual_feedback=True),
            PerceptionMotionGate(resume_valid_frames=5),pivot_verified=True)
        for seq in range(1,6):
            now=seq*.05
            boundary=SimpleNamespace(valid=True,source='yolopv2',semantic_sequence=seq,
                semantic_captured_at=now-.01,semantic_hard_safe=True)
            result=runtime.filter(DifferentialDriveCommand('turn-right',.5,.3,.2,.9),
                LaneEstimate(True,.9,heading_error=.6,near_heading_error=.03),boundary,now=now,
                imu_sample={'value':{'radians':[0.,0.,0.]},'valid':True,'stale':False,
                    'age_s':0.,'received_monotonic':now},frame_age_seconds=.01)
        self.assertEqual(result.action,'forward')

    def setUp(self):
        self.assertIsNotNone(visual, 'visual feedback controller is missing')
        self.controller = visual.VisualFeedbackController()

    def tick(self, now, seq, intent='pivot-right', yaw=0., **changes):
        values = dict(now=now, semantic_sequence=seq, captured_at=now-.01,
                      desired_action=intent, local_valid=True,
                      yaw_rad=math.radians(yaw), imu_timestamp=now, imu_valid=True)
        values.update(changes)
        return self.controller.update(**values)

    def test_new_visual_alignment_stops_pivot_without_finishing_fixed_angle(self):
        self.assertEqual(self.tick(1., 1).action, 'pivot-right')
        result = self.tick(1.05, 2, 'forward', -2.)
        self.assertEqual(result.action, 'stop')
        self.assertEqual(self.tick(1.8, 3, 'forward', -2.).action, 'forward')

    def test_same_intent_never_renews_step_deadline(self):
        self.tick(1., 1)
        self.assertEqual(self.tick(1.1, 2, yaw=-2.).action, 'pivot-right')
        self.assertEqual(self.tick(1.151, 3, yaw=-3.).action, 'stop')

    def test_yaw_is_a_per_step_ceiling_not_a_mandatory_turn_target(self):
        self.tick(1., 1)
        self.assertEqual(self.tick(1.05, 2, yaw=-5.1).action, 'stop')
        # An aligned road after only five degrees permits a forward step.
        self.assertEqual(self.tick(1.8, 3, 'forward', -7.).action, 'forward')

    def test_settle_requires_frame_captured_after_settle_not_cached_inference(self):
        self.tick(1., 1)
        self.tick(1.16, 2)
        self.assertEqual(self.tick(1.9, 3, captured_at=1.75).action, 'stop')
        self.assertEqual(self.tick(1.91, 3, captured_at=1.90).action, 'stop')
        self.assertEqual(self.tick(1.92, 4, captured_at=1.91).action, 'pivot-right')

    def test_invalid_current_road_stops_and_cannot_resume_same_frame(self):
        self.tick(1., 1)
        self.assertEqual(self.tick(1.05, 2, local_valid=False).action, 'stop')
        self.assertEqual(self.tick(1.9, 2).action, 'stop')
        self.assertEqual(self.tick(1.91, 3).action, 'pivot-right')

    def test_missing_imu_cannot_start_or_continue_pivot(self):
        self.assertEqual(self.tick(1., 1, imu_valid=False).action, 'stop')
        self.assertEqual(self.tick(1.1, 2).action, 'pivot-right')
        self.assertEqual(self.tick(1.11, 3, imu_timestamp=.5).action, 'stop')

    def test_unknown_action_and_future_or_regressed_frames_never_move(self):
        self.assertEqual(self.tick(1., 1, 'reverse').action, 'stop')
        self.assertEqual(self.tick(1.1, 2, captured_at=2.).action, 'stop')
        self.assertEqual(self.tick(1.2, 1).action, 'stop')

    def test_forward_steps_also_stop_for_reobservation(self):
        self.assertEqual(self.tick(1., 1, 'forward').action, 'forward')
        self.assertEqual(self.tick(1.151, 2, 'forward').action, 'stop')

    def test_latest_yolo_revokes_pivot_even_when_imu_has_not_reached_ceiling(self):
        self.tick(1., 1)
        self.assertEqual(self.tick(1.04, 2, 'stop', -.5).action, 'stop')

    def test_veto_discards_intent_and_preserves_stop_interval(self):
        self.tick(1., 1)
        self.assertEqual(self.controller.veto('external stop').action, 'stop')
        self.assertEqual(self.tick(1.1, 2, 'forward').action, 'stop')


class VisualRuntimeTests(unittest.TestCase):
    def test_field_profile_recovers_on_two_distinct_delayed_observations(self):
        from autodrive.runtime.onboard import load_config, DEFAULT_CONFIG
        _, config = load_config(DEFAULT_CONFIG, 'rosmaster_jetson_yolopv2_visual_feedback')
        runtime = StationaryCornerRuntime(StationaryCornerConfig(**config['stationary_corner']),
            PerceptionMotionGate(**{k: config['safety'][k] for k in
                ('resume_valid_frames', 'resume_min_confidence')}), pivot_verified=True)
        def tick(now, sequence, captured, safe=True):
            import numpy as np
            road=np.zeros((240,320),np.uint8);road[100:,30:300]=1
            return runtime.filter(DifferentialDriveCommand('forward', 0., .3, .3, .9),
                LaneEstimate(True, .9, heading_error=.1, near_heading_error=.03,
                    centerline=np.array([[160,225],[160,211],[160,187]])),
                SimpleNamespace(valid=True, source='yolopv2', semantic_sequence=sequence,
                    semantic_captured_at=captured, semantic_hard_safe=safe,
                    corridor_mask=road,semantic_yellow_mask=np.zeros_like(road)),
                now=now, imu_sample=None, frame_age_seconds=.03)
        self.assertEqual(tick(10., 1, 9.6).action, 'stop')
        # Cached reads must not count as the second observation.
        self.assertEqual(tick(10.01, 1, 9.6).action, 'stop')
        self.assertEqual(tick(10.1, 2, 9.7).action, 'forward')
        deadline = runtime.controller.get_state()['motion_deadline']
        self.assertLessEqual(deadline, 10.35)
        self.assertEqual(tick(10.12, 2, 9.7).action, 'forward')
        self.assertEqual(runtime.controller.get_state()['motion_deadline'], deadline)
        self.assertEqual(tick(10.41, 2, 9.7).action, 'stop')
        self.assertEqual(tick(10.5, 3, 10.1).action, 'stop')
        # Recovery still requires an image captured after the actual stop.
        self.assertEqual(tick(10.6, 4, 10.2).action, 'stop')
        self.assertEqual(tick(10.7, 5, 10.51).action, 'forward')
        self.assertEqual(tick(10.71, 6, 10.61, safe=False).action, 'stop')

    def test_field_profile_accepts_delayed_masks_and_still_expires_them(self):
        import numpy as np
        from autodrive.runtime.onboard import load_config, DEFAULT_CONFIG
        from autodrive.perception.yolopv2_fusion import YOLOPv2FusionConfig, _MaskResult
        from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
        _, config = load_config(DEFAULT_CONFIG, 'rosmaster_jetson_yolopv2_visual_feedback')
        options = dict(config['perception']['yolopv2'], asynchronous=False)
        clock = [10.]
        detector = YOLOPv2SemanticDetector(YOLOPv2FusionConfig(**options), model=object(),
            clock=lambda: clock[0], output_width=320, output_height=240)
        self.addCleanup(detector.close)
        road = np.zeros((240, 320), np.uint8); road[100:, 60:260] = 1
        lane = np.zeros_like(road)
        detector._consumer_result = _MaskResult(1, 9.6, 9.7, .1, 'fp16', road, lane)
        self.assertTrue(detector.semantic_corridor(road, lane).valid)
        clock[0] = 10.21
        self.assertFalse(detector.semantic_corridor(road, lane).valid)

    def test_cruise_keeps_current_small_corrections_but_near_front_uses_steps(self):
        runtime=StationaryCornerRuntime(
            StationaryCornerConfig(enabled=True,visual_feedback=True,continuous_cruise=True),
            PerceptionMotionGate(resume_valid_frames=5),pivot_verified=True)
        def tick(now, seq, front=None):
            return runtime.filter(DifferentialDriveCommand('turn-right',.12,.3,.27,.9),
                LaneEstimate(True,.9,heading_error=.5,near_heading_error=.03,lateral_error=.1),
                SimpleNamespace(valid=True,source='yolopv2',semantic_sequence=seq,
                    semantic_captured_at=now-.01,semantic_hard_safe=True,
                    semantic_front_observed=front is not None,semantic_observed_front_ratio=front),
                now=now,imu_sample=None,frame_age_seconds=.01)
        for seq in range(1,6): result=tick(seq*.05,seq)
        self.assertEqual(result.action,'turn-right')
        self.assertEqual(result.steering,.12)
        self.assertEqual(tick(.35,6).action,'turn-right')
        self.assertEqual(tick(.45,7).action,'turn-right')
        self.assertEqual(tick(.5,8,.8).action,'stop')
        self.assertFalse(runtime.controller.get_state()['continuous'])

    def test_driver_deadline_or_cancel_rejection_is_recorded_as_final_veto(self):
        from autodrive.runtime import onboard
        self.assertTrue(hasattr(onboard,'apply_runtime_command'))
        from unittest.mock import patch
        for cancel in (False,True):
            runtime=StationaryCornerRuntime(StationaryCornerConfig(enabled=True,visual_feedback=True),
                PerceptionMotionGate(),pivot_verified=True)
            runtime.controller.update(now=10.,semantic_sequence=1,captured_at=9.99,
                desired_action='forward',local_valid=True)
            runtime._state={'stationary_application_allowed':True}
            driver=SafeWheelDriver()
            if cancel:driver.inhibit('operator ended this run')
            with patch('autodrive.control.drive_runtime.time.monotonic',return_value=10.16):
                command,state=onboard.apply_runtime_command(driver,
                    DifferentialDriveCommand('forward',0.,.3,.3,.9),runtime)
            self.assertEqual(command.action,'stop')
            self.assertEqual(state['front_left_pwm'],0)
            self.assertFalse(runtime.get_state()['stationary_application_allowed'])
            self.assertEqual(runtime.controller.get_state()['stopped_at'],10.16)
            self.assertIn('operator' if cancel else 'expired',command.reason)

    def test_live_adapter_selects_from_heading_without_fixed_entry_position(self):
        self.assertIn('visual_feedback', StationaryCornerConfig.__dataclass_fields__)
        runtime = StationaryCornerRuntime(StationaryCornerConfig(enabled=True, visual_feedback=True),
            PerceptionMotionGate(resume_valid_frames=5), pivot_verified=True)
        def tick(now, seq, heading):
            boundary = SimpleNamespace(valid=True, source='yolopv2', semantic_sequence=seq,
                semantic_captured_at=now-.01, semantic_hard_safe=True,
                semantic_front_boundary_ratio=None, semantic_right_exit_observed=False)
            return runtime.filter(DifferentialDriveCommand('turn-right', .5, .3, .2, .9, 'current path'),
                LaneEstimate(True, .9, heading_error=heading, near_heading_error=heading), boundary, now=now,
                imu_sample={'value':{'radians':[0.,0.,0.]},'valid':True,'stale':False,
                            'age_s':0.,'received_monotonic':now}, frame_age_seconds=.01)
        for seq in range(1,5):
            self.assertEqual(tick(seq*.05, seq, .6).action, 'stop')
        self.assertEqual(tick(.25, 5, .6).action, 'pivot-right')
        self.assertEqual(tick(.3, 6, 0.).action, 'stop')
        self.assertEqual(tick(1.02, 7, 0.).action, 'forward')

    def test_driver_deadline_stops_without_waiting_for_the_next_model_frame(self):
        driver = SafeWheelDriver(config=WheelMappingConfig())
        self.assertTrue(hasattr(driver, 'check_motion_deadline'), 'independent step deadline missing')
        import time
        driver.apply(DifferentialDriveCommand('forward', 0., .3, .3, .9), deadline=time.monotonic()+.015)
        time.sleep(.025)
        driver.check_motion_deadline()
        self.assertEqual(driver.get_state()['front_left_pwm'], 0)

    def test_operator_stop_latches_against_a_late_control_proposal(self):
        driver = SafeWheelDriver()
        self.assertTrue(hasattr(driver, 'inhibit'), 'operator stop cannot latch')
        driver.inhibit('operator ended this run')
        state = driver.apply(DifferentialDriveCommand('forward', 0., .3, .3, .9))
        self.assertEqual(state['front_left_pwm'], 0)

    def test_guard_actually_expires_motion_while_control_thread_is_idle(self):
        import time
        import tempfile
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('autodrive.control.motion_guard'))
        from autodrive.control.motion_guard import MotionGuard
        driver=SafeWheelDriver()
        with MotionGuard(driver):
            driver.apply(DifferentialDriveCommand('forward',0.,.3,.3,.9),deadline=time.monotonic()+.02)
            time.sleep(.07)
            self.assertEqual(driver.get_state()['front_left_pwm'],0)

    def test_expired_browser_lease_inhibits_future_motion(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('autodrive.control.motion_guard'))
        from autodrive.control.motion_guard import MotionGuard
        import tempfile,time,json
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'lease.json').write_text(json.dumps({'expires_monotonic':time.monotonic()-.01}))
            driver=SafeWheelDriver()
            with MotionGuard(driver,Path(temp)) as guard:
                time.sleep(.05)
                self.assertTrue(guard.cancelled.is_set())
                self.assertEqual(driver.apply(DifferentialDriveCommand('forward',0.,.3,.3,.9))['front_left_pwm'],0)


if __name__ == '__main__':
    unittest.main()
