"""Explicit stationary pivots use a separate opt-in; all chassis calls are fake."""
from dataclasses import replace
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.control.drive_runtime import CommandWatchdog, SafeWheelDriver, WheelMappingConfig
from autodrive.control.lane_centering import DifferentialDriveCommand


class RecordingChassis:
    def __init__(self):
        self.commands = []
        self.ramps = []
        self.stopped = threading.Event()

    def set_four_wheels(self, *values):
        self.commands.append(values)

    def ramp_four_to(self, front_left, rear_left, front_right, rear_right, duration):
        self.commands.append((front_left, rear_left, front_right, rear_right))
        self.ramps.append(duration)

    def stop(self):
        self.commands.append((0, 0, 0, 0))
        self.stopped.set()


def pivot(direction='right'):
    sign = 1 if direction == 'right' else -1
    return DifferentialDriveCommand('pivot-' + direction, sign, .4 * sign, -.4 * sign,
                                    .9, 'test stationary turn')


class StationaryPivotMappingTests(unittest.TestCase):
    def test_default_pivot_rejection_stops_existing_motion(self):
        chassis = RecordingChassis()
        driver = SafeWheelDriver(chassis, True, WheelMappingConfig(transition_time=0))
        driver.apply(DifferentialDriveCommand('forward', 0, .4, .4, .9))
        chassis.commands.clear()
        with self.assertRaises(ValueError):
            driver.apply(pivot())
        self.assertEqual(chassis.commands, [(0, 0, 0, 0)])
        self.assertEqual(driver.get_state()['action'], 'stopped')

    def test_default_configuration_reports_pivot_disabled(self):
        self.assertEqual(SafeWheelDriver().get_state()['mapping'].get('stationary_pivot_pwm'), 0)

    def test_enabled_pivots_map_equal_and_opposite_bounded_wheels(self):
        for pwm in (1, 17, 30):
            driver = SafeWheelDriver(config=WheelMappingConfig(stationary_pivot_pwm=pwm))
            with self.subTest(pwm=pwm):
                self.assertEqual(driver.command_to_four_pwm(pivot()), (pwm, pwm, -pwm, -pwm))
                self.assertEqual(driver.command_to_four_pwm(pivot('left')), (-pwm, -pwm, pwm, pwm))

    def test_pivot_configuration_requires_integer_and_both_output_limits(self):
        for value in (-1, 31, True, False, 1., '20', None, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                WheelMappingConfig(stationary_pivot_pwm=value)
        with self.assertRaises(ValueError):
            WheelMappingConfig(pwm_limit=20, stationary_pivot_pwm=21)
        self.assertEqual(WheelMappingConfig(pwm_limit=20, stationary_pivot_pwm=20).stationary_pivot_pwm, 20)

    def test_invalid_direction_or_nonfinite_proposal_stops_without_nonzero_write(self):
        wrong = [replace(pivot(), steering=-1), replace(pivot(), steering=0),
                 replace(pivot(), left_speed=-.4), replace(pivot(), left_speed=0),
                 replace(pivot(), right_speed=.4), replace(pivot(), right_speed=0),
                 replace(pivot('left'), steering=1), replace(pivot('left'), left_speed=.4),
                 replace(pivot('left'), right_speed=-.4)]
        for name in ('steering', 'left_speed', 'right_speed'):
            for value in (float('nan'), float('inf'), float('-inf'), True, '1', 1.1, -1.1):
                wrong.append(replace(pivot(), **{name: value}))
        for command in wrong:
            with self.subTest(command=command):
                chassis = RecordingChassis()
                driver = SafeWheelDriver(chassis, True,
                    WheelMappingConfig(stationary_pivot_pwm=30, transition_time=0))
                driver.apply(pivot())
                chassis.commands.clear()
                with self.assertRaises(ValueError):
                    driver.apply(command)
                self.assertEqual(chassis.commands, [(0, 0, 0, 0)])
                self.assertEqual(driver.get_state()['left_pwm'], 0)
                self.assertEqual(driver.get_state()['right_pwm'], 0)

    def test_ordinary_trim_mapping_stays_forward_only_when_pivot_enabled(self):
        driver = SafeWheelDriver(config=WheelMappingConfig(stationary_pivot_pwm=30))
        for action, steering, left, right, expected in (
            ('forward', 0, .4, .4, (16, 16, 20, 20)),
            ('turn-right', .5, .6, .2, (26, 26, 10, 10)),
            ('turn-left', -.5, .2, .6, (10, 10, 26, 26)),
            ('turn-right', 1, .4, -.4, (26, 26, 10, 10)),
            ('turn-left', -1, -.4, .4, (10, 10, 26, 26)),
            ('stop', 0, 0, 0, (0, 0, 0, 0)),
        ):
            with self.subTest(action=action, steering=steering):
                self.assertEqual(driver.command_to_four_pwm(
                    DifferentialDriveCommand(action, steering, left, right, .9)), expected)

    def test_apply_deduplicates_pivot_and_stop_uses_existing_state(self):
        chassis = RecordingChassis()
        driver = SafeWheelDriver(chassis, True,
            WheelMappingConfig(stationary_pivot_pwm=30, transition_time=.1))
        state = driver.apply(pivot())
        driver.apply(pivot())
        self.assertEqual(chassis.commands, [(30, 30, -30, -30)])
        self.assertEqual(chassis.ramps, [.1])
        self.assertEqual((state['action'], state['left_pwm'], state['right_pwm']),
                         ('pivot-right', 30, -30))
        driver.apply(pivot('left'))
        self.assertEqual(chassis.commands[-1], (-30, -30, 30, 30))
        driver.stop('test stop')
        driver.stop('test repeated stop')
        self.assertEqual(chassis.commands.count((0, 0, 0, 0)), 1)
        self.assertEqual(driver.get_state()['action'], 'stopped')

    def test_dry_run_pivot_does_not_touch_chassis(self):
        chassis = RecordingChassis()
        driver = SafeWheelDriver(chassis, False, WheelMappingConfig(stationary_pivot_pwm=30))
        state = driver.apply(pivot())
        driver.stop()
        self.assertEqual((state['left_pwm'], state['right_pwm']), (30, -30))
        self.assertEqual(chassis.commands, [])

    def test_watchdog_stops_stationary_pivot(self):
        chassis = RecordingChassis()
        driver = SafeWheelDriver(chassis, True,
            WheelMappingConfig(stationary_pivot_pwm=30, transition_time=0))
        watchdog = CommandWatchdog(driver, timeout=.02, check_interval=.005)
        try:
            driver.apply(pivot())
            watchdog.arm()
            self.assertTrue(chassis.stopped.wait(1))
            self.assertTrue(watchdog.get_state()['tripped'])
            self.assertEqual(chassis.commands, [(30, 30, -30, -30), (0, 0, 0, 0)])
            self.assertEqual(driver.get_state()['action'], 'stopped')
        finally:
            watchdog.close()


if __name__ == '__main__':
    unittest.main()
