#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


CAR_DIR = Path(__file__).resolve().parents[1]
CONTROL_DIR = CAR_DIR / "control"
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

from autodrive.tools.check_wheel_directions import individual_targets
from vehicle_control.factory import (
    HardwareNotCalibratedError,
    create_chassis,
    validate_motion_configuration,
)
from vehicle_control.platforms.raspbot import RaspbotChassis
from vehicle_control.platforms.rosmaster import RosmasterChassis
from vehicle_control.profile import load_runtime_config, load_vehicle_profile


class FakeMotorController:
    def __init__(self):
        self.commands = []

    def Ctrl_Muto(self, motor_id, pwm):
        self.commands.append((motor_id, pwm))


class FakeRosmasterController:
    def __init__(self):
        self.commands = []

    def set_motor(self, *commands):
        self.commands.append(tuple(commands))


class HardwareMappingTests(unittest.TestCase):
    def test_motor_id_targets_follow_verified_physical_positions(self):
        self.assertEqual(individual_targets(0, 18, "forward"), (18, 0, 0, 0))
        self.assertEqual(individual_targets(1, 18, "forward"), (0, 18, 0, 0))
        self.assertEqual(individual_targets(2, 18, "forward"), (0, 0, 18, 0))
        self.assertEqual(individual_targets(3, 18, "reverse"), (0, 0, 0, -18))

    def test_logical_wheel_order_uses_verified_motor_ids(self):
        controller = FakeMotorController()
        chassis = RaspbotChassis(controller=controller)

        chassis.set_four_wheels(11, 12, 13, 14)

        self.assertEqual(
            controller.commands,
            [
                (0, 11),  # front-left
                (1, 12),  # rear-left
                (2, 13),  # front-right
                (3, 14),  # rear-right
            ],
        )

    def test_raspbot_signs_are_applied_after_logical_clamping(self):
        controller = FakeMotorController()
        chassis = RaspbotChassis(
            controller=controller,
            command_limit=20,
            wheel_signs=(1, -1, 1, -1),
        )

        logical = chassis.set_four_wheels(25, 12, -25, -9)

        self.assertEqual(logical, (20, 12, -20, -9))
        self.assertEqual(
            controller.commands,
            [(0, 20), (1, -12), (2, -20), (3, 9)],
        )

    def test_rosmaster_reorders_logical_wheels_into_one_serial_command(self):
        controller = FakeRosmasterController()
        chassis = RosmasterChassis(
            controller=controller,
            command_limit=100,
            motor_order=(1, 0, 3, 2),
            wheel_signs=(1, -1, 1, -1),
        )

        chassis.set_four_wheels(11, 12, 13, 14)

        self.assertEqual(controller.commands, [(-12, 11, -14, 13)])

    def test_uncalibrated_rosmaster_profile_is_blocked_in_normal_runtime(self):
        _, profile = load_vehicle_profile("rosmaster_jetson")

        with self.assertRaises(HardwareNotCalibratedError):
            validate_motion_configuration(profile, {"pwm_limit": 20})
        with self.assertRaises(HardwareNotCalibratedError):
            create_chassis(profile, controller=FakeRosmasterController())

    def test_factory_can_construct_uncalibrated_backend_for_guarded_checks(self):
        _, profile = load_vehicle_profile("rosmaster_jetson")
        controller = FakeRosmasterController()

        chassis = create_chassis(
            profile,
            require_motion_calibrated=False,
            controller=controller,
        )
        chassis.set_four_wheels(1, 2, 3, 4)

        self.assertEqual(controller.commands, [(1, 2, 3, 4)])

    def test_runtime_profile_deep_merge_preserves_common_algorithm_config(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "runtime.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "vehicle": {"profile": "raspbot_pi5"},
                        "camera": {"index": 9, "gimbal": {"settle_time": 2.0}},
                        "lcc": {"base_speed": 0.12},
                        "runtime": {"output_dir": "shared-output"},
                    }
                ),
                encoding="utf-8",
            )

            _, resolved = load_runtime_config(config, "rosmaster_jetson")

        self.assertEqual(resolved["vehicle"]["profile"], "rosmaster_jetson")
        self.assertEqual(resolved["camera"]["index"], 0)
        self.assertEqual(resolved["camera"]["gimbal"]["settle_time"], 0.8)
        self.assertEqual(resolved["lcc"]["base_speed"], 0.12)
        self.assertEqual(
            resolved["runtime"]["output_dir"],
            "outputs/onboard_runtime/rosmaster_jetson",
        )


if __name__ == "__main__":
    unittest.main()
