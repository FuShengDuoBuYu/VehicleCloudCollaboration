#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


CAR_DIR = Path(__file__).resolve().parents[1]
CONTROL_UTILS_DIR = CAR_DIR / "control" / "utils"
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))
if str(CONTROL_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_UTILS_DIR))

from autodrive.camera.gimbal import (
    CameraGimbalPose,
    initialize_configured_gimbal,
    startup_pose_from_mapping,
)
from Raspbot_Lib import Raspbot
from vehicle_control.factory import create_gimbal
from vehicle_control.profile import load_vehicle_profile


class FakeServoController:
    def __init__(self):
        self.commands = []
        self.closed = False

    def Ctrl_Servo(self, servo_id, angle):
        self.commands.append((servo_id, angle))

    def close(self):
        self.closed = True


class FakeRosmasterServoController:
    def __init__(self):
        self.commands = []
        self.closed = False

    def set_pwm_servo(self, servo_id, angle):
        self.commands.append((servo_id, angle))

    def close(self):
        self.closed = True


class CameraGimbalTests(unittest.TestCase):
    def test_pan_and_tilt_use_s1_and_s2(self):
        controller = FakeServoController()
        sleeps = []
        pose = CameraGimbalPose(pan_angle=90, tilt_angle=70, settle_time=0.5)

        commands = pose.apply(controller, sleep=sleeps.append)

        self.assertEqual(commands, ((1, 90), (2, 70)))
        self.assertEqual(controller.commands, [(1, 90), (2, 70)])
        self.assertEqual(sleeps, [0.5])

    def test_pose_requires_at_least_one_axis(self):
        with self.assertRaises(ValueError):
            CameraGimbalPose()

    def test_chassis_tilt_limit_is_enforced(self):
        with self.assertRaises(ValueError):
            CameraGimbalPose(tilt_angle=101)

    def test_low_level_driver_clamps_before_building_i2c_payload(self):
        controller = Raspbot.__new__(Raspbot)
        writes = []
        controller.write_array = lambda register, data: writes.append((register, data))

        controller.Ctrl_Servo(2, 150)

        self.assertEqual(writes, [(0x02, [2, 100])])

    def test_disabled_startup_pose_does_not_touch_i2c(self):
        self.assertIsNone(startup_pose_from_mapping({"gimbal": {}}))

    def test_configured_startup_pose_centers_pan_and_closes_i2c(self):
        controller = FakeServoController()
        sleeps = []
        camera_config = {
            "gimbal": {
                "initialize_on_startup": True,
                "pan_angle": 90,
                "tilt_angle": None,
                "settle_time": 0.8,
            }
        }

        commands = initialize_configured_gimbal(
            camera_config,
            controller_factory=lambda: controller,
            sleep=sleeps.append,
        )

        self.assertEqual(commands, ((1, 90),))
        self.assertEqual(controller.commands, [(1, 90)])
        self.assertEqual(sleeps, [0.8])
        self.assertTrue(controller.closed)

    def test_rosmaster_gimbal_uses_pwm_servo_backend(self):
        _, profile = load_vehicle_profile("rosmaster_jetson")
        low_level = FakeRosmasterServoController()
        gimbal = create_gimbal(
            profile,
            require_gimbal_calibrated=False,
            controller=low_level,
        )
        pose = CameraGimbalPose(pan_angle=30, tilt_angle=40, settle_time=0)

        commands = pose.apply(gimbal)
        gimbal.close()

        self.assertEqual(commands, ((1, 30), (2, 40)))
        self.assertEqual(low_level.commands, [(1, 30), (2, 40)])
        self.assertTrue(low_level.closed)


if __name__ == "__main__":
    unittest.main()
