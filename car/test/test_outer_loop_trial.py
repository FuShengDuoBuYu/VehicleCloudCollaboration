"""Bounded field-trial authorization and model-only outer-route selection."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.runtime import onboard


class OuterLoopTrialTests(unittest.TestCase):
    def test_current_visual_profile_passes_two_observation_startup_preflight(self):
        from autodrive.control.field_trial import validate_stationary_corner_config
        _, config = onboard.load_config(onboard.DEFAULT_CONFIG, 'rosmaster_jetson_yolopv2_visual_feedback')
        validate_stationary_corner_config(config)
        for value in (1, True, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config['safety']['resume_valid_frames'] = value
                validate_stationary_corner_config(config)

    def test_visual_trial_preflight_accepts_extended_age_and_rejects_unbounded_age(self):
        _, config = onboard.load_config(onboard.DEFAULT_CONFIG, 'rosmaster_jetson_yolopv2_visual_feedback')
        # Isolate disk-backed pulse evidence; no model/devices are constructed.
        # Actual evidence is checked separately by the deployment preflight.
        config['safety']['resume_valid_frames'] = 5
        with patch('autodrive.control.field_trial._validate_stationary_calibration'):
            onboard.validate_field_trial_config(config)
            for value in (.61, float('inf'), float('nan'), True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    config['perception']['yolopv2']['max_result_age_seconds'] = value
                    onboard.validate_field_trial_config(config)

    def test_legacy_trial_keeps_original_age_and_recovery_bounds(self):
        config = self.config()
        config['perception']['yolopv2']['max_result_age_seconds'] = .31
        with self.assertRaises(ValueError):
            onboard.validate_field_trial_config(config)
        _, visual = onboard.load_config(onboard.DEFAULT_CONFIG, 'rosmaster_jetson_yolopv2_visual_feedback')
        visual['stationary_corner'].update(visual_feedback=False, continuous_cruise=False)
        from autodrive.control.field_trial import validate_stationary_corner_config
        with self.assertRaises(ValueError):
            validate_stationary_corner_config(visual)

    def test_short_motion_window_starts_on_nonzero_output_and_does_not_restart(self):
        from autodrive.control.field_trial import TrialMotionWindow
        window = TrialMotionWindow(.35)
        window.record_output((0, 0, 0, 0), 10.)
        self.assertFalse(window.expired(50.))
        window.record_output((30, 30, 22, 22), 51.)
        self.assertFalse(window.expired(51.34))
        window.record_output((0, 0, 0, 0), 51.2)
        window.record_output((30, 30, 22, 22), 51.3)
        self.assertTrue(window.expired(51.36))

    def test_motion_window_rejects_nonfinite_and_negative_duration(self):
        from autodrive.control.field_trial import TrialMotionWindow
        for value in (-1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrialMotionWindow(value)
        disabled = TrialMotionWindow(0.)
        disabled.record_output((30, 30, 0, 0), 1.)
        self.assertFalse(disabled.expired(1000.))

    def test_live_trial_starts_without_blocking_camera_loop_in_a_ramp(self):
        from autodrive.control.drive_runtime import SafeWheelDriver, WheelMappingConfig
        from autodrive.control.lane_centering import DifferentialDriveCommand
        _, config = onboard.load_config(onboard.DEFAULT_CONFIG, 'rosmaster_jetson_yolopv2_outer_trial')
        class Chassis:
            command = None
            def ramp_four_to(self, *args):
                raise AssertionError('blocking ramp causes observed YOLO result expiry')
            def set_four_wheels(self, *args): self.command = args
        chassis = Chassis()
        driver = SafeWheelDriver(chassis=chassis,motors_enabled=True,
                                 config=WheelMappingConfig(**config['wheels']))
        driver.apply(DifferentialDriveCommand('forward',0,0,0,1,'test',0))
        self.assertEqual(chassis.command,(30,30,30,30))

    def config(self):
        _, config = onboard.load_config(onboard.DEFAULT_CONFIG, 'rosmaster_jetson_yolopv2')
        config['vehicle']['status'].update(wheel_mapping_verified=True,
                                          ground_forward_stop_verified=True)
        config['perception']['semantic_outer_loop'] = {
            'enabled': True, 'width_profile': [[0.5, .3], [1., 1.]],
            'preview_ratio': .62}
        return config

    def test_explicit_bounded_raw_image_trial_accepts_verified_setup(self):
        self.assertTrue(hasattr(onboard, 'validate_field_trial_config'))
        config = self.config()
        onboard.validate_field_trial_config(config)
        onboard.validate_motor_request(True, None, onboard.MOTOR_CONFIRMATION,
                                       None, config['camera'], image_space_trial=True,
                                       max_runtime_seconds=30)
        self.assertFalse(config['vehicle']['status']['motion_calibrated'])

    def test_trial_rejects_missing_evidence_or_excess_output(self):
        self.assertTrue(hasattr(onboard, 'validate_field_trial_config'))
        for section, key, value in (
            ('status', 'wheel_mapping_verified', False),
            ('status', 'ground_forward_stop_verified', 'true'),
            ('wheels', 'pwm_limit', 31),
            ('wheels', 'front_left_base_pwm', 20),
            ('camera', 'flip_horizontal', True),
            ('camera', 'index', 99),
        ):
            with self.subTest(key=key):
                config = self.config()
                target = config['vehicle']['status'] if section == 'status' else config[section]
                target[key] = value
                with self.assertRaises(ValueError):
                    onboard.validate_field_trial_config(config)

    def test_trial_never_allows_replay_missing_confirmation_or_unbounded_time(self):
        for video, confirmation, seconds in (
            ('recorded.mp4', onboard.MOTOR_CONFIRMATION, 30),
            (None, '', 30), (None, onboard.MOTOR_CONFIRMATION, 0),
            (None, onboard.MOTOR_CONFIRMATION, float('nan')),
            (None, onboard.MOTOR_CONFIRMATION, 121),
        ):
            with self.subTest(video=video, seconds=seconds), self.assertRaises(ValueError):
                onboard.validate_motor_request(True, video, confirmation, None,
                    image_space_trial=True, max_runtime_seconds=seconds)

    def test_normal_hardware_mode_still_requires_calibration(self):
        with self.assertRaises(ValueError):
            onboard.validate_motor_request(True, None, onboard.MOTOR_CONFIRMATION, None)

    def test_outer_selector_keeps_outer_left_corridor_at_inner_opening(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector = SemanticOuterRoute(width_profile=[[0, .4], [1, .4]])
        corridor = np.zeros((100, 200), np.uint8)
        corridor[:, 50:130] = 1
        corridor[35:76, 130:200] = 1  # right inner branch joins outer road
        result, valid, _ = selector.select(corridor)
        self.assertTrue(valid)
        self.assertTrue(np.all(result[60, 50:130]))
        self.assertFalse(np.any(result[60, 130:]))
        self.assertTrue(np.all(result <= corridor))

    def test_outer_selector_never_paints_over_model_exclusions(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector = SemanticOuterRoute(width_profile=[[0, .4], [1, .4]])
        corridor = np.zeros((100, 200), np.uint8)
        corridor[:, 50:130] = 1
        corridor[72, 99:103] = 0
        result, _, _ = selector.select(corridor)
        self.assertFalse(np.any(result[corridor == 0]))

    def test_tiny_enclosed_lane_fragment_does_not_cut_off_remaining_row(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector = SemanticOuterRoute(width_profile=[[0,.4],[1,.4]])
        corridor = np.zeros((100,200),np.uint8)
        corridor[:,50:130] = 1
        corridor[72,80:84] = 0
        result,valid,_ = selector.select(corridor)
        self.assertTrue(valid)
        self.assertFalse(result[72,80:84].any())
        self.assertTrue(np.all(result[72,84:130]))
        self.assertTrue(np.all(result <= corridor))

    def test_large_hole_and_branch_separator_still_constrain_outer_route(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector = SemanticOuterRoute(width_profile=[[0,.8],[1,.8]])
        for exclusion in ('large','vertical','exterior'):
            corridor = np.zeros((240,320),np.uint8)
            corridor[:,40:290] = 1
            if exclusion == 'large': corridor[155:170,130:155] = 0
            if exclusion == 'vertical': corridor[130:185,130:132] = 0
            if exclusion == 'exterior': corridor[162,0:140] = 0
            result,_,_ = selector.select(corridor)
            with self.subTest(exclusion=exclusion):
                self.assertFalse(np.any(result[corridor==0]))
                if exclusion != 'exterior':self.assertFalse(result[162,170])

    def test_current_small_exclusion_on_fitted_path_still_stops(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        from autodrive.control.lane_centering import RoadCenterlineEstimator
        selector = SemanticOuterRoute(width_profile=[[0,.8],[1,.8]])
        corridor = np.zeros((240,320),np.uint8)
        corridor[:,40:280] = 1
        # A hole straddling the path stays excluded even when row extents skip it.
        corridor[162:164,155:164] = 0
        result,_,_ = selector.select(corridor)
        estimate = RoadCenterlineEstimator().estimate(result,preserve_exclusions=True)
        self.assertFalse(estimate.valid)

    def test_missing_outer_edge_at_preview_stops(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector = SemanticOuterRoute(width_profile=[[0, .4], [1, .4]])
        corridor = np.ones((100, 200), np.uint8)
        result, valid, reason = selector.select(corridor)
        self.assertFalse(valid)
        self.assertFalse(result.any())
        self.assertIn('outer', reason)

    def test_fork_must_not_substitute_inner_branch_for_missing_outer_edge(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector = SemanticOuterRoute(width_profile=[[0, .4], [1, 1.]])
        corridor = np.zeros((240, 320), np.uint8)
        corridor[168:,64:272] = 1
        corridor[:168,:96] = 1
        corridor[:168,152:272] = 1
        _, valid, _ = selector.select(corridor)
        self.assertFalse(valid)

    def test_trial_waits_for_first_archived_frame_and_stops_on_write_error(self):
        from autodrive.control.field_trial import field_trial_recording_ready
        self.assertFalse(field_trial_recording_ready({'written_frames':0,'error':None}, {'error':None}))
        self.assertTrue(field_trial_recording_ready({'written_frames':1,'error':None}, {'error':None}))
        for archive, log in (({'written_frames':1,'dropped_frames':1},{}),
                             ({'written_frames':1},{'dropped_rows':1})):
            with self.assertRaisesRegex(RuntimeError,'dropped'):
                field_trial_recording_ready(archive,log)
        for archive, log in (({'written_frames':1,'error':'disk full'}, {'error':None}),
                             ({'written_frames':1,'error':None}, {'error':'I/O error'})):
            with self.assertRaises(RuntimeError): field_trial_recording_ready(archive, log)


if __name__ == '__main__':
    unittest.main()
