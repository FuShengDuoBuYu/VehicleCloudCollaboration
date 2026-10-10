"""Operator-selected straight PWM must retain bounded differential steering."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
from autodrive.control.drive_runtime import SafeWheelDriver, WheelMappingConfig
from autodrive.control.lane_centering import DifferentialDriveCommand
from autodrive.tools.check_wheel_directions import build_parser
from vehicle_control.profile import load_vehicle_profile


class Pwm30DefaultsTest(unittest.TestCase):
    def test_field_profile_straight_turn_and_stop_mapping(self):
        _, profile = load_vehicle_profile("rosmaster_jetson_yolopv2")
        driver = SafeWheelDriver(motors_enabled=False, config=WheelMappingConfig(
            **profile["runtime_overrides"]["wheels"]))
        for action, steering, factor, expected in (
            ("forward", 0.0, 0.0, (30, 30, 30, 30)),
            ("turn-right", 0.5, 0.0, (30, 30, 25, 25)),
            ("turn-left", -0.5, 0.0, (25, 25, 30, 30)),
            ("turn-right", 1.0, 1.0, (30, 30, 0, 0)),
            ("turn-left", -1.0, 1.0, (0, 0, 30, 30)),
            ("stop", 0.0, 0.0, (0, 0, 0, 0)),
        ):
            with self.subTest(action=action, steering=steering):
                command = DifferentialDriveCommand(action, steering, 0, 0, 1,
                                                   "mock calibration", factor)
                self.assertEqual(driver.command_to_four_pwm(command), expected)
        self.assertFalse(profile["status"]["motion_calibrated"])

    def test_default_wheel_test_uses_operator_selected_pwm(self):
        self.assertEqual(build_parser().parse_args([]).pwm, 30)

    def test_upper_limit_does_not_cancel_inner_wheel_reduction(self):
        driver = SafeWheelDriver(motors_enabled=False, config=WheelMappingConfig(
            pwm_limit=30, minimum_moving_pwm=10,
            front_left_base_pwm=30, rear_left_base_pwm=30,
            front_right_base_pwm=30, rear_right_base_pwm=30,
            front_steering_delta_pwm=20, maximum_steering_delta_pwm=10))
        for steering, expected in ((0.5, (30, 30, 20, 20)),
                                   (-0.5, (20, 20, 30, 30))):
            command = DifferentialDriveCommand("turn", steering, 0, 0, 1, "test", 0)
            self.assertEqual(driver.command_to_four_pwm(command), expected)


if __name__ == "__main__":
    unittest.main()
