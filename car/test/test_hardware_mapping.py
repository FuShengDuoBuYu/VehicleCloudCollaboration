#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


CAR_DIR = Path(__file__).resolve().parents[1]
CONTROL_DIR = CAR_DIR / "control"
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

from autodrive.check_wheel_directions import individual_targets
from vehicle_control.hardware import RospbotChassis


class FakeMotorController:
    def __init__(self):
        self.commands = []

    def Ctrl_Muto(self, motor_id, pwm):
        self.commands.append((motor_id, pwm))


class HardwareMappingTests(unittest.TestCase):
    def test_motor_id_targets_follow_verified_physical_positions(self):
        self.assertEqual(individual_targets(0, 18, "forward"), (18, 0, 0, 0))
        self.assertEqual(individual_targets(1, 18, "forward"), (0, 18, 0, 0))
        self.assertEqual(individual_targets(2, 18, "forward"), (0, 0, 18, 0))
        self.assertEqual(individual_targets(3, 18, "reverse"), (0, 0, 0, -18))

    def test_logical_wheel_order_uses_verified_motor_ids(self):
        controller = FakeMotorController()
        with patch("vehicle_control.hardware.Raspbot", return_value=controller):
            chassis = RospbotChassis()

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


if __name__ == "__main__":
    unittest.main()
