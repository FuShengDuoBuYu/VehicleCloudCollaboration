"""Clock/yaw driven corner tests; no model, camera, serial or wheel access."""
import importlib
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    corner = importlib.import_module('autodrive.control.stationary_corner')
except ModuleNotFoundError as exc:
    if exc.name != 'autodrive.control.stationary_corner':
        raise
    corner = None


class StationaryCornerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(corner, 'stationary corner controller is not implemented')
        self.controller = corner.StationaryCornerController(
            corner.StationaryCornerConfig(enabled=True))

    def observe(self, now, sequence, yaw=0., **changes):
        values = dict(now=now, semantic_sequence=sequence, semantic_age_seconds=.01,
                      front_boundary_ratio=.78, right_exit_observed=True,
                      yaw_rad=math.radians(yaw), imu_timestamp=now, imu_valid=True,
                      exit_aligned=False, local_valid=True)
        values.update(changes)
        return self.controller.update(**values)

    def pivot(self, yaw=0.):
        self.assertEqual(self.observe(0., 1, yaw).action, 'stop')
        self.assertEqual(self.observe(.05, 2, yaw).state, 'settle')
        self.assertEqual(self.observe(.299, 3, yaw).action, 'stop')
        self.assertEqual(self.observe(.31, 4, yaw).action, 'pivot-right')

    def finish(self):
        self.pivot()
        self.observe(.41, 5, -30.)
        self.observe(.51, 6, -60.)
        result = self.observe(.61, 7, -90.)
        self.assertEqual(result.action, 'stop')
        self.assertEqual(result.state, 'exit')
        return result

    def test_disabled_controller_never_replaces_an_existing_proposal(self):
        self.controller = corner.StationaryCornerController()
        self.assertEqual(self.controller.update(now=0.).action, 'passthrough')
        self.assertEqual(self.observe(1., 1).action, 'passthrough')

    def test_visible_distant_front_only_proposes_straight_when_local_valid(self):
        for sequence in range(1, 5):
            result = self.observe(sequence*.05, sequence, front_boundary_ratio=.70)
            self.assertEqual(result.action, 'straight')
            self.assertEqual(result.state, 'approach')
        self.assertEqual(self.observe(.3, 5, front_boundary_ratio=.70,
                                      local_valid=False).action, 'stop')
        self.assertEqual(self.observe(.4, 6, front_boundary_ratio=None).action, 'stop')
        self.assertEqual(self.controller.get_state()['state'], 'approach')

    def test_lost_front_cue_cannot_restore_early_arc_or_start_a_blind_pivot(self):
        self.observe(0., 1, front_boundary_ratio=.57)
        for sequence in range(2, 8):
            result = self.observe(sequence*.1, sequence, front_boundary_ratio=None)
            self.assertEqual((result.state,result.action), ('approach','stop'))
        self.assertEqual(self.observe(.8, 8, front_boundary_ratio=.70).action,'straight')
        self.assertEqual(self.observe(.9, 9, front_boundary_ratio=.78).action,'stop')
        self.assertEqual(self.observe(1., 10, front_boundary_ratio=.78).state,'settle')

    def test_entry_needs_two_distinct_current_observations_and_zero_settle(self):
        self.observe(0., 1)
        for now in (.02, .04, .06):
            self.assertEqual(self.observe(now, 1).state, 'approach')
        self.assertEqual(self.observe(.1, 2).state, 'settle')
        self.assertEqual(self.observe(.349, 3).action, 'stop')
        self.assertEqual(self.observe(.351, 4).action, 'pivot-right')

    def test_missing_right_or_too_close_front_cannot_start_pivot(self):
        for changes in ({'right_exit_observed':False}, {'right_exit_observed':1},
                        {'front_boundary_ratio':.86}, {'front_boundary_ratio':float('nan')}):
            with self.subTest(changes=changes):
                self.setUp()
                for i in range(4):
                    self.assertEqual(self.observe(i*.1, i, **changes).action, 'stop')

    def test_settle_rechecks_current_corner_before_starting(self):
        self.observe(0., 1)
        self.observe(.05, 2)
        result = self.observe(.4, 3, right_exit_observed=False)
        self.assertEqual(result.action, 'stop')
        self.assertNotEqual(result.state, 'pivot-right')

    def test_right_yaw_wraps_and_stops_exactly_at_target(self):
        self.pivot(-170.)
        self.assertEqual(self.observe(.41, 5, 179.).action, 'pivot-right')
        self.assertEqual(self.observe(.51, 6, 149.).action, 'pivot-right')
        self.assertEqual(self.observe(.61, 7, 119.).action, 'pivot-right')
        result = self.observe(.71, 8, 100.)
        self.assertEqual(result.action, 'stop')
        self.assertEqual(result.state, 'exit')
        self.assertAlmostEqual(result.yaw_progress_rad, math.pi/2)

    def test_configured_positive_yaw_sign_is_used_consistently(self):
        self.controller = corner.StationaryCornerController(
            corner.StationaryCornerConfig(enabled=True, yaw_sign=1))
        self.pivot()
        self.observe(.41, 5, 30.)
        self.observe(.51, 6, 60.)
        self.assertEqual(self.observe(.61, 7, 90.).action, 'stop')

    def test_cached_imu_timestamp_cannot_accumulate_rotation(self):
        self.pivot()
        self.observe(.4, 5, -20.)
        for now in (.42, .44, .46):
            result = self.observe(now, 6, -20., imu_timestamp=.4)
            self.assertAlmostEqual(result.yaw_progress_rad, math.radians(20))
        self.assertEqual(result.action, 'pivot-right')

    def test_changed_yaw_with_duplicate_timestamp_is_rejected(self):
        self.pivot()
        result = self.observe(.4, 5, -20., imu_timestamp=.31)
        self.assertEqual((result.state, result.action), ('blocked', 'stop'))

    def test_imu_stale_future_invalid_or_backward_timestamp_blocks(self):
        for changes in ({'imu_timestamp':0.}, {'imu_timestamp':1.},
                        {'imu_valid':False}, {'yaw_rad':float('nan')},
                        {'imu_timestamp':.30}):
            with self.subTest(changes=changes):
                self.setUp(); self.pivot()
                result = self.observe(.4, 5, **changes)
                self.assertEqual((result.state, result.action), ('blocked', 'stop'))
                self.assertEqual(self.observe(.5, 6).state, 'blocked')

    def test_large_feedback_jump_cannot_fake_a_completed_turn(self):
        self.pivot()
        result = self.observe(.4, 5, -150.)
        self.assertEqual((result.state, result.action), ('blocked', 'stop'))

    def test_reverse_yaw_after_progress_blocks_instead_of_counting_absolute_motion(self):
        self.pivot()
        self.observe(.4, 5, -12.)
        result = self.observe(.5, 6, 0.)
        self.assertEqual((result.state, result.action), ('blocked', 'stop'))

    def test_no_progress_timeout_and_total_timeout_are_independent(self):
        self.pivot()
        for now in (.5, .7, .9, 1.1):
            self.assertEqual(self.observe(now, int(now*100)).action, 'pivot-right')
        self.assertEqual(self.observe(1.12, 120).state, 'blocked')
        self.controller = corner.StationaryCornerController(
            corner.StationaryCornerConfig(enabled=True, max_pivot_seconds=1.))
        self.pivot()
        for i, now in enumerate((.51, .71, .91, 1.11), 1):
            self.assertEqual(self.observe(now, 10+i, -i*3.).action, 'pivot-right')
        self.assertEqual(self.observe(1.32, 20, -15.).state, 'blocked')

    def test_camera_pause_preserves_yaw_and_requires_a_new_sequence_to_resume(self):
        self.pivot()
        self.observe(.4, 5, -20.)
        paused = self.observe(.5, 5, -30., semantic_age_seconds=.31)
        self.assertEqual(paused.action, 'stop')
        self.assertTrue(paused.paused)
        self.assertAlmostEqual(paused.yaw_progress_rad, math.radians(30.))
        self.assertEqual(self.observe(.6, 5, -30.).action, 'stop')
        resumed = self.observe(.7, 6, -30.)
        self.assertEqual(resumed.action, 'pivot-right')
        self.assertAlmostEqual(resumed.yaw_progress_rad, math.radians(30.))
        self.observe(.8, 7, -60.)
        self.assertEqual(self.observe(.9, 8, -90.).state, 'exit')

    def test_progress_arriving_after_deadline_cannot_extend_allowance(self):
        self.pivot()
        for now in (.5, .7, .9, 1.1):
            self.observe(now, int(now*100))
        result = self.observe(1.12, 120, -2.)
        self.assertEqual((result.state, result.action), ('blocked', 'stop'))

    def test_settle_yaw_drift_is_not_part_of_commanded_turn(self):
        self.observe(0., 1, -10.)
        self.observe(.05, 2, -15.)
        started = self.observe(.31, 3, -20.)
        self.assertEqual(started.action, 'pivot-right')
        self.assertAlmostEqual(started.yaw_progress_rad, 0.)
        self.assertAlmostEqual(self.observe(.41, 4, -40.).yaw_progress_rad,
                               math.radians(20.))

    def test_camera_pause_does_not_grant_more_total_pivot_time(self):
        self.controller = corner.StationaryCornerController(
            corner.StationaryCornerConfig(enabled=True, max_pivot_seconds=1.))
        self.pivot()
        for i, now in enumerate((.4, .6, .8, 1., 1.2), 10):
            result = self.observe(now, i, semantic_age_seconds=.4)
            self.assertEqual(result.action, 'stop')
        self.assertEqual(self.observe(1.32, 20).state, 'blocked')

    def test_local_veto_pauses_pivot_and_cannot_be_overridden(self):
        self.pivot()
        self.assertEqual(self.observe(.4, 5, -10., local_valid=False).action, 'stop')
        self.assertEqual(self.observe(.5, 6, -10.).action, 'pivot-right')

    def test_target_reached_while_camera_invalid_is_still_latched(self):
        self.pivot()
        self.observe(.4, 5, -30.)
        self.observe(.5, 6, -60.)
        done = self.observe(.6, 7, -90., semantic_age_seconds=.4)
        self.assertEqual((done.state, done.action), ('exit', 'stop'))
        self.assertEqual(self.observe(.7, 8, -90.).action, 'stop')

    def test_exit_needs_new_aligned_observations_and_settle_time(self):
        self.finish()
        self.observe(.66, 8, -90., exit_aligned=True)
        for now in (.68, .70, .72):
            self.assertEqual(self.observe(now, 8, -90., exit_aligned=True).action, 'stop')
        self.observe(.74, 9, -90., exit_aligned=True)
        self.assertEqual(self.observe(.8, 10, -90., exit_aligned=True).action, 'stop')
        result = self.observe(.9, 11, -90., exit_aligned=True)
        self.assertEqual((result.state, result.action), ('cruise', 'passthrough'))

    def test_unaligned_or_stale_exit_resets_confirmation_count(self):
        self.finish()
        self.observe(.7, 8, -90., exit_aligned=True)
        self.observe(.8, 9, -90., exit_aligned=True)
        self.observe(.9, 10, -90., exit_aligned=False)
        self.assertEqual(self.observe(1., 11, -90., exit_aligned=True).action, 'stop')
        self.observe(1.1, 12, -90., exit_aligned=True, semantic_age_seconds=.4)
        self.assertEqual(self.observe(1.2, 13, -90., exit_aligned=True).action, 'stop')

    def test_completed_corner_cannot_repeat_without_observing_departure(self):
        self.finish()
        for i, now in enumerate((.71, .81, .91), 8):
            self.observe(now, i, -90., exit_aligned=True)
        for i, now in enumerate((1., 1.1, 1.2, 1.3), 11):
            self.assertEqual(self.observe(now, i, -90.).action, 'stop')
        self.observe(1.4, 15, -90., front_boundary_ratio=None)
        self.assertEqual(self.observe(1.5, 16, -90.).action, 'stop')
        self.assertEqual(self.observe(1.6, 17, -90.).state, 'settle')

    def test_old_semantic_sequence_cannot_resume_or_trigger(self):
        self.observe(0., 10)
        self.assertEqual(self.observe(.1, 9).action, 'stop')
        self.assertEqual(self.observe(.2, 11).state, 'approach')
        self.assertEqual(self.observe(.3, 12).state, 'settle')

    def test_bad_or_backward_controller_time_fails_closed(self):
        for now in (-1., float('nan'), float('inf')):
            with self.subTest(now=now):
                self.setUp(); self.observe(1., 1)
                self.assertEqual(self.observe(now, 2).state, 'blocked')

    def test_configuration_rejects_unsafe_numeric_and_boolean_settings(self):
        for options in ({'enabled':'false'}, {'yaw_sign':0}, {'yaw_sign':True},
                        {'front_near_ratio':.9}, {'front_max_ratio':.9},
                        {'settle_seconds':.1}, {'imu_max_age_seconds':.3},
                        {'semantic_max_age_seconds':float('nan')},
                        {'enter_observations':True}, {'exit_observations':0},
                        {'progress_timeout_seconds':0}, {'max_pivot_seconds':11.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                corner.StationaryCornerConfig(**options)


if __name__ == '__main__':
    unittest.main()
