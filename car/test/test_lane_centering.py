#!/usr/bin/env python3

import sys
import time
import unittest
from pathlib import Path

import cv2
import numpy as np


CAR_DIR = Path(__file__).resolve().parents[1]
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.lane_centering import LaneCenteringController, RoadCenterlineEstimator
from autodrive.perspective import PerspectiveMapper
from autodrive.visualization import render_debug_frame
from autodrive.temporal import TemporalMaskPropagator
from autodrive.run_onboard import MOTOR_CONFIRMATION, validate_motor_request
from autodrive.drive_runtime import (
    CommandWatchdog,
    SafeWheelDriver,
    WheelMappingConfig,
)


class FakeChassis:
    def __init__(self):
        self.commands = []

    def ramp_to(self, left, right, duration):
        self.commands.append(("ramp", int(left), int(right), float(duration)))

    def stop(self):
        self.commands.append(("stop", 0, 0))


def make_road(shift_at_top=0):
    height, width = 180, 320
    mask = np.zeros((height, width), dtype=np.uint8)
    near_center = width // 2
    far_center = near_center + shift_at_top
    polygon = np.array(
        [
            [near_center - 125, height - 1],
            [near_center + 125, height - 1],
            [far_center + 42, int(height * 0.42)],
            [far_center - 42, int(height * 0.42)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 1)
    return mask


class RoadCenterlineEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.estimator = RoadCenterlineEstimator()

    def test_straight_road_produces_forward_command(self):
        estimate = self.estimator.estimate(make_road())
        command = LaneCenteringController().update(estimate, dt=0.2)
        self.assertTrue(estimate.valid)
        self.assertGreater(estimate.confidence, 0.5)
        self.assertAlmostEqual(estimate.lateral_error, 0.0, delta=0.04)
        self.assertAlmostEqual(estimate.heading_error, 0.0, delta=0.04)
        self.assertEqual(command.action, "forward")
        self.assertAlmostEqual(command.left_speed, command.right_speed, delta=0.03)

    def test_right_curve_produces_right_turn(self):
        estimate = self.estimator.estimate(make_road(shift_at_top=65))
        command = LaneCenteringController().update(estimate, dt=0.2)
        self.assertTrue(estimate.valid)
        self.assertGreater(estimate.heading_error, 0.1)
        self.assertEqual(command.action, "turn-right")
        self.assertGreater(command.left_speed, command.right_speed)

    def test_left_curve_produces_left_turn(self):
        estimate = self.estimator.estimate(make_road(shift_at_top=-65))
        command = LaneCenteringController().update(estimate, dt=0.2)
        self.assertTrue(estimate.valid)
        self.assertLess(estimate.heading_error, -0.1)
        self.assertEqual(command.action, "turn-left")
        self.assertLess(command.left_speed, command.right_speed)

    def test_missing_road_stops(self):
        estimate = self.estimator.estimate(np.zeros((180, 320), dtype=np.uint8))
        command = LaneCenteringController().update(estimate, dt=0.2)
        self.assertFalse(estimate.valid)
        self.assertEqual(command.action, "stop")
        self.assertEqual((command.left_speed, command.right_speed), (0.0, 0.0))

    def test_identity_perspective_preserves_mask_and_points(self):
        points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        mapper = PerspectiveMapper(points, points)
        mask = make_road()
        warped = mapper.warp_mask(mask)
        self.assertTrue(np.array_equal(mask, warped))
        original = np.array([[20, 30], [160, 120]], dtype=np.int32)
        restored = mapper.camera_points(original, mask.shape)
        self.assertTrue(np.array_equal(original, restored))

    def test_dry_run_does_not_touch_chassis(self):
        chassis = FakeChassis()
        driver = SafeWheelDriver(chassis=chassis, motors_enabled=False)
        estimate = self.estimator.estimate(make_road(shift_at_top=50))
        command = LaneCenteringController().update(estimate, dt=0.2)
        state = driver.apply(command)
        self.assertEqual(chassis.commands, [])
        self.assertEqual(state["mode"], "dry-run")
        self.assertNotEqual((state["left_pwm"], state["right_pwm"]), (0, 0))

    def test_enabled_driver_maps_and_stops(self):
        chassis = FakeChassis()
        mapping = WheelMappingConfig(pwm_limit=30, minimum_moving_pwm=10)
        driver = SafeWheelDriver(
            chassis=chassis, motors_enabled=True, config=mapping
        )
        estimate = self.estimator.estimate(make_road(shift_at_top=50))
        command = LaneCenteringController().update(estimate, dt=0.2)
        state = driver.apply(command)
        self.assertEqual(chassis.commands[0][0], "ramp")
        self.assertLessEqual(abs(state["left_pwm"]), 30)
        self.assertLessEqual(abs(state["right_pwm"]), 30)
        driver.stop("test")
        self.assertEqual(chassis.commands[-1][0], "stop")

    def test_watchdog_stops_stale_command(self):
        chassis = FakeChassis()
        driver = SafeWheelDriver(chassis=chassis, motors_enabled=True)
        watchdog = CommandWatchdog(driver, timeout=0.04, check_interval=0.005)
        try:
            watchdog.arm()
            time.sleep(0.08)
            self.assertTrue(watchdog.get_state()["tripped"])
            self.assertEqual(chassis.commands[-1][0], "stop")
        finally:
            watchdog.close()

    def test_visualization_accepts_empty_masks(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        mask = np.zeros((90, 160), dtype=np.uint8)
        estimate = self.estimator.estimate(mask)
        command = LaneCenteringController().update(estimate, dt=0.2)
        rendered = render_debug_frame(
            frame, mask, mask, estimate, command, inference_ms=10.0
        )
        self.assertEqual(rendered.shape, frame.shape)

    def test_temporal_mask_propagates_horizontal_motion(self):
        rng = np.random.default_rng(7)
        first_gray = rng.integers(0, 256, (90, 160), dtype=np.uint8)
        first = cv2.cvtColor(first_gray, cv2.COLOR_GRAY2BGR)
        transform = np.float32([[1, 0, 5], [0, 1, 0]])
        second = cv2.warpAffine(
            first,
            transform,
            (160, 90),
            borderMode=cv2.BORDER_REFLECT,
        )
        mask = np.zeros((90, 160), dtype=np.uint8)
        mask[30:70, 45:95] = 1
        expected = cv2.warpAffine(
            mask, transform, (160, 90), flags=cv2.INTER_NEAREST
        )
        propagator = TemporalMaskPropagator(max_steps=2)
        propagator.reset(first, mask, mask)
        propagated, _, confidence = propagator.propagate(second)
        intersection = np.count_nonzero((propagated > 0) & (expected > 0))
        union = np.count_nonzero((propagated > 0) | (expected > 0))
        self.assertGreater(intersection / union, 0.80)
        self.assertLess(confidence, 1.0)

    def test_motor_enable_guard_rejects_video_and_missing_calibration(self):
        with self.assertRaises(ValueError):
            validate_motor_request(
                True,
                "recorded.mp4",
                MOTOR_CONFIRMATION,
                Path("calibration.yaml"),
            )
        with self.assertRaises(ValueError):
            validate_motor_request(True, None, MOTOR_CONFIRMATION, None)
        with self.assertRaises(ValueError):
            validate_motor_request(
                True, None, "wrong confirmation", Path("calibration.yaml")
            )


if __name__ == "__main__":
    unittest.main()
