#!/usr/bin/env python3

import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml


CAR_DIR = Path(__file__).resolve().parents[1]
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.lane_centering import (
    DifferentialDriveCommand,
    LCCConfig,
    LaneEstimate,
    LaneCenteringController,
    RoadCenterlineEstimator,
)
from autodrive.outer_loop import OuterLoopBoundaryConfig, OuterLoopBoundaryTracker
from autodrive.perspective import (
    PerspectiveMapper,
    camera_pose_from_mapping,
    validate_calibration_camera_pose,
)
from autodrive.run_onboard import (
    SurfaceOnlyDetector,
    analyze,
)
from autodrive.visualization import render_debug_frame
from autodrive.run_onboard import MOTOR_CONFIRMATION, validate_motor_request
from autodrive.drive_runtime import (
    CommandWatchdog,
    PerceptionMotionGate,
    SafeWheelDriver,
    WheelMappingConfig,
)


class FakeChassis:
    def __init__(self):
        self.commands = []

    def ramp_to(self, left, right, duration):
        self.commands.append(("ramp", int(left), int(right), float(duration)))

    def ramp_four_to(
        self,
        front_left,
        rear_left,
        front_right,
        rear_right,
        duration,
    ):
        self.commands.append(
            (
                "ramp-four",
                int(front_left),
                int(rear_left),
                int(front_right),
                int(rear_right),
                float(duration),
            )
        )

    def set_wheels(self, left, right):
        self.commands.append(("set", int(left), int(right)))

    def set_four_wheels(
        self,
        front_left,
        rear_left,
        front_right,
        rear_right,
    ):
        self.commands.append(
            (
                "set-four",
                int(front_left),
                int(rear_left),
                int(front_right),
                int(rear_right),
            )
        )

    def stop(self):
        self.commands.append(("stop", 0, 0))


class ZeroWarpMapper:
    """Erase warped masks to distinguish raw-image and bird's-eye hazards."""

    @staticmethod
    def warp_mask(mask):
        return np.zeros_like(mask)

    @staticmethod
    def camera_mask(mask):
        return mask

    @staticmethod
    def camera_estimate(estimate, _shape):
        return estimate


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


def make_outer_loop_masks(shift_at_top=0, right_gap=True, right_branch=True):
    height, width = 180, 320
    drivable = np.zeros((height, width), dtype=np.uint8)
    lane = np.zeros_like(drivable)
    bottom_y, top_y = height - 1, 40
    left_bottom, right_bottom = 72, 250
    left_top = 92 + shift_at_top
    right_top = 228 + shift_at_top
    polygon = np.array(
        [
            [left_bottom, bottom_y],
            [right_bottom, bottom_y],
            [right_top, top_y],
            [left_top, top_y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(drivable, [polygon], 1)
    if right_branch:
        cv2.rectangle(drivable, (right_top, 72), (width - 1, 125), 1, -1)

    cv2.line(lane, (left_bottom, bottom_y), (left_top, top_y), 1, 5)
    if right_gap:
        cv2.line(lane, (right_bottom, bottom_y), (242 + shift_at_top, 118), 1, 5)
        cv2.line(lane, (238 + shift_at_top, 92), (right_top, top_y), 1, 5)
    else:
        cv2.line(lane, (right_bottom, bottom_y), (right_top, top_y), 1, 5)
    return drivable, lane


class RoadCenterlineEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.estimator = RoadCenterlineEstimator()

    def test_surface_only_detector_returns_configured_empty_masks(self):
        detector = SurfaceOnlyDetector(width=160, height=90)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        drivable, lane = detector.predict_masks(frame)
        self.assertEqual(drivable.shape, (90, 160))
        self.assertFalse(np.any(drivable))
        self.assertFalse(np.any(lane))

    def test_route_hint_does_not_override_local_continuity(self):
        chosen = self.estimator._choose_segment(
            [(48, 67), (82, 252)],
            previous_center=166.5,
            width=320,
            route_hint="left",
        )
        self.assertEqual(chosen, (82, 252))

    def test_route_hint_breaks_equal_distance_toward_requested_side(self):
        chosen = self.estimator._choose_segment(
            [(90, 150), (170, 230)],
            previous_center=160.0,
            width=320,
            route_hint="left",
        )
        self.assertEqual(chosen, (90, 150))

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

    def test_excessive_lateral_error_stops_instead_of_recovering(self):
        estimate = self.estimator.estimate(make_road())
        estimate.lateral_error = 0.25
        controller = LaneCenteringController(
            LCCConfig(maximum_lateral_error=0.20)
        )
        command = controller.update(estimate, dt=0.2)
        self.assertEqual(command.action, "stop")
        self.assertIn("lateral error", command.reason)

    def test_steering_smoothing_prevents_one_frame_direction_flip(self):
        controller = LaneCenteringController(
            LCCConfig(
                lateral_gain=0.0,
                heading_gain=1.0,
                derivative_gain=0.0,
                steering_smoothing=0.75,
            )
        )
        left = LaneEstimate(True, 0.9, heading_error=-0.8)
        right = LaneEstimate(True, 0.9, heading_error=0.8)
        first = controller.update(left, dt=0.2)
        second = controller.update(right, dt=0.2)
        self.assertLess(first.steering, 0.0)
        self.assertLess(second.steering, 0.1)

    def test_identity_perspective_preserves_mask_and_points(self):
        points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        mapper = PerspectiveMapper(points, points)
        mask = make_road()
        warped = mapper.warp_mask(mask)
        self.assertTrue(np.array_equal(mask, warped))
        original = np.array([[20, 30], [160, 120]], dtype=np.int32)
        restored = mapper.camera_points(original, mask.shape)
        self.assertTrue(np.array_equal(original, restored))
        restored_mask = mapper.camera_mask(mask)
        self.assertTrue(np.array_equal(mask, restored_mask))

    def test_camera_pose_signature_tracks_gimbal_and_rotated_image_size(self):
        pose = camera_pose_from_mapping(
            {
                "width": 640,
                "height": 480,
                "rotation_degrees": 90,
                "flip_horizontal": True,
                "gimbal": {"pan_angle": 25, "tilt_angle": None},
            }
        )
        self.assertEqual((pose["image_width"], pose["image_height"]), (480, 640))
        self.assertEqual(pose["pan_angle"], 25)
        self.assertTrue(pose["flip_horizontal"])

    def test_calibration_camera_pose_must_match_runtime(self):
        camera = {
            "width": 640,
            "height": 480,
            "rotation_degrees": 0,
            "flip_horizontal": False,
            "flip_vertical": False,
            "gimbal": {"pan_angle": 25, "tilt_angle": None},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "calibrated": True,
                        "camera_pose": camera_pose_from_mapping(camera),
                        "source_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "destination_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ),
                encoding="utf-8",
            )
            validate_calibration_camera_pose(path, camera)
            changed = dict(camera)
            changed["gimbal"] = {"pan_angle": 30, "tilt_angle": None}
            with self.assertRaisesRegex(ValueError, "pan_angle"):
                validate_calibration_camera_pose(path, changed)

    def test_motor_enable_rejects_legacy_calibration_without_camera_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "calibrated": True,
                        "source_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "destination_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "camera_pose"):
                validate_motor_request(
                    True,
                    None,
                    MOTOR_CONFIRMATION,
                    path,
                    {"gimbal": {"pan_angle": 25}},
                )

    def test_outer_loop_tracker_bridges_inner_boundary_gap(self):
        drivable, lane = make_outer_loop_masks(right_gap=True, right_branch=True)
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(enabled=True, minimum_observations=8)
        )
        result = tracker.update(drivable, lane)
        estimate = self.estimator.estimate(result.corridor_mask)
        command = LaneCenteringController().update(estimate, dt=0.2)
        self.assertTrue(result.valid)
        self.assertEqual(result.source, "both")
        self.assertLess(np.mean(result.corridor_mask[:, 285:] > 0), 0.02)
        self.assertAlmostEqual(estimate.heading_error, 0.0, delta=0.08)
        self.assertEqual(command.action, "forward")

    def test_outer_loop_tracker_rejects_oversized_inner_branch(self):
        drivable, lane = make_outer_loop_masks(right_gap=False, right_branch=True)
        lane[:, 160:] = 0
        cv2.line(lane, (315, 179), (300, 40), 1, 5)
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(
                enabled=True,
                expected_lane_width_ratio=0.50,
                maximum_lane_width_ratio=0.60,
                minimum_observations=8,
            )
        )
        result = tracker.update(drivable, lane)
        self.assertTrue(result.valid)
        self.assertLessEqual(result.lane_width_ratio, 0.60)
        self.assertLess(float(np.median(result.right_curve)), 285.0)

    def test_outer_loop_surface_mask_excludes_green_island(self):
        frame = np.full((180, 320, 3), (150, 150, 150), dtype=np.uint8)
        frame[:, 210:] = (70, 150, 80)
        tracker = OuterLoopBoundaryTracker(OuterLoopBoundaryConfig(enabled=True))
        surface = tracker.road_surface_mask(frame, (90, 160))
        self.assertGreater(float(np.mean(surface[:, :90])), 0.95)
        self.assertLess(float(np.mean(surface[:, 115:])), 0.05)

    def test_outer_loop_surface_mask_excludes_desaturated_green(self):
        frame = np.full((180, 320, 3), (175, 163, 151), dtype=np.uint8)
        frame[:, 210:] = (153, 178, 110)
        tracker = OuterLoopBoundaryTracker(OuterLoopBoundaryConfig(enabled=True))
        surface = tracker.road_surface_mask(frame, (90, 160))
        self.assertGreater(float(np.mean(surface[:, :90])), 0.95)
        self.assertLess(float(np.mean(surface[:, 115:])), 0.05)

    def test_outer_loop_detects_yellow_boundary_under_vehicle(self):
        yellow = np.zeros((180, 320), dtype=np.uint8)
        cv2.line(yellow, (80, 165), (240, 150), 1, 7)
        tracker = OuterLoopBoundaryTracker(OuterLoopBoundaryConfig(enabled=True))
        hazard, ratio = tracker.yellow_under_ego(yellow)
        self.assertTrue(hazard)
        self.assertGreater(ratio, 0.020)

    def test_outer_loop_normal_curved_boundaries_stay_outside_ego_footprint(self):
        tracker = OuterLoopBoundaryTracker(OuterLoopBoundaryConfig(enabled=True))
        for shift in (-60, -30, 0, 30, 60):
            _, yellow = make_outer_loop_masks(
                shift_at_top=shift,
                right_gap=False,
                right_branch=True,
            )
            hazard, _ = tracker.yellow_under_ego(yellow)
            self.assertFalse(hazard, f"normal curve shift={shift} was hazardous")

    def test_analyze_checks_vehicle_footprint_before_perspective_warp(self):
        frame = np.full((180, 320, 3), (150, 150, 150), dtype=np.uint8)
        cv2.line(frame, (80, 165), (240, 150), (0, 255, 255), 7)
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(enabled=True, include_lane_mask=False)
        )
        result = analyze(
            SurfaceOnlyDetector(width=320, height=180),
            RoadCenterlineEstimator(),
            LaneCenteringController(),
            ZeroWarpMapper(),
            frame,
            "left",
            None,
            2.5,
            tracker,
        )
        estimate, command, *_, boundary = result
        self.assertTrue(boundary.yellow_hazard)
        self.assertFalse(estimate.valid)
        self.assertEqual(command.action, "stop")
        self.assertIn("yellow boundary", command.reason)

    def test_outer_loop_tracker_uses_outer_boundary_when_inner_is_missing(self):
        drivable, lane = make_outer_loop_masks(right_gap=False, right_branch=True)
        lane[:, 160:] = 0
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(
                enabled=True,
                expected_lane_width_ratio=0.50,
                minimum_observations=8,
            )
        )
        result = tracker.update(drivable, lane)
        self.assertTrue(result.valid)
        self.assertEqual(result.source, "outer+width")
        self.assertGreater(result.confidence, 0.3)

    def test_outer_loop_tracker_expires_missing_history(self):
        drivable, lane = make_outer_loop_masks(right_gap=False, right_branch=False)
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(
                enabled=True,
                minimum_observations=8,
                maximum_missing_frames=2,
            )
        )
        self.assertTrue(tracker.update(drivable, lane).valid)
        missing = np.zeros_like(lane)
        self.assertTrue(tracker.update(drivable, missing).valid)
        self.assertTrue(tracker.update(drivable, missing).valid)
        self.assertFalse(tracker.update(drivable, missing).valid)

    def test_outer_loop_tracker_preserves_curve_direction(self):
        drivable, lane = make_outer_loop_masks(
            shift_at_top=48,
            right_gap=True,
            right_branch=True,
        )
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(enabled=True, minimum_observations=8)
        )
        result = tracker.update(drivable, lane)
        estimate = self.estimator.estimate(result.corridor_mask)
        command = LaneCenteringController().update(estimate, dt=0.2)
        self.assertTrue(result.valid)
        self.assertGreater(estimate.heading_error, 0.08)
        self.assertEqual(command.action, "turn-right")

    def test_outer_loop_sequence_tracks_both_curves_and_bridges_one_lost_frame(self):
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(
                enabled=True,
                include_lane_mask=False,
                minimum_observations=8,
                maximum_lane_width_ratio=0.60,
            )
        )
        estimator = RoadCenterlineEstimator(lookahead_ratio=0.58)
        controller = LaneCenteringController(
            LCCConfig(
                min_confidence=0.35,
                derivative_gain=0.0,
                steering_smoothing=0.65,
            )
        )
        gate = PerceptionMotionGate(resume_valid_frames=4)
        actions = []
        sources = []
        for index in range(80):
            shift = int(round(55 * np.sin(2 * np.pi * index / 80)))
            drivable, yellow = make_outer_loop_masks(
                shift,
                right_gap=index % 7 != 0,
                right_branch=True,
            )
            if index == 30:
                yellow[:] = 0
            result = tracker.update(
                drivable,
                np.zeros_like(yellow),
                yellow,
            )
            estimate = estimator.estimate(result.corridor_mask)
            estimate.confidence *= result.confidence
            command = gate.filter(controller.update(estimate, dt=0.05))
            actions.append(command.action)
            sources.append(result.source)

        self.assertEqual(actions[:3], ["stop", "stop", "stop"])
        self.assertNotIn("stop", actions[3:])
        self.assertIn("turn-left", actions)
        self.assertIn("turn-right", actions)
        self.assertIn("history", sources)

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
        self.assertEqual(chassis.commands[0][0], "ramp-four")
        self.assertLessEqual(abs(state["left_pwm"]), 30)
        self.assertLessEqual(abs(state["right_pwm"]), 30)
        driver.stop("test")
        self.assertEqual(chassis.commands[-1][0], "stop")

    def test_current_four_wheel_mapping_matches_grounded_calibration(self):
        driver = SafeWheelDriver(
            motors_enabled=False,
            config=WheelMappingConfig(
                pwm_limit=35,
                minimum_moving_pwm=10,
                front_left_base_pwm=16,
                rear_left_base_pwm=16,
                front_right_base_pwm=20,
                rear_right_base_pwm=20,
                front_steering_delta_pwm=20,
                maximum_steering_delta_pwm=10,
            ),
        )
        self.assertEqual(
            driver.command_to_four_pwm(
                DifferentialDriveCommand("forward", 0.0, 0, 0, 1, "test")
            ),
            (16, 16, 20, 20),
        )
        self.assertEqual(
            driver.command_to_four_pwm(
                DifferentialDriveCommand("turn-right", 0.5, 0, 0, 1, "test")
            ),
            (26, 26, 10, 10),
        )
        self.assertEqual(
            driver.command_to_four_pwm(
                DifferentialDriveCommand("turn-left", -0.5, 0, 0, 1, "test")
            ),
            (10, 10, 26, 26),
        )

    def test_running_updates_do_not_repeat_blocking_ramp(self):
        chassis = FakeChassis()
        driver = SafeWheelDriver(
            chassis=chassis,
            motors_enabled=True,
            config=WheelMappingConfig(
                pwm_limit=35,
                front_left_base_pwm=16,
                rear_left_base_pwm=16,
                front_right_base_pwm=20,
                rear_right_base_pwm=20,
                front_steering_delta_pwm=20,
                maximum_steering_delta_pwm=10,
                transition_time=0.25,
            ),
        )
        straight = DifferentialDriveCommand(
            "forward", 0.0, 0.0, 0.0, 0.9, "test"
        )
        right = DifferentialDriveCommand(
            "turn-right", 0.5, 0.0, 0.0, 0.9, "test"
        )

        driver.apply(straight)
        driver.apply(right)
        driver.apply(right)

        self.assertEqual(chassis.commands[0][0], "ramp-four")
        self.assertEqual(chassis.commands[1][0], "set-four")
        self.assertEqual(len(chassis.commands), 2)

    def test_repeated_stop_frames_write_hardware_stop_only_once(self):
        chassis = FakeChassis()
        driver = SafeWheelDriver(chassis=chassis, motors_enabled=True)

        driver.stop("first")
        driver.stop("same stopped state")

        self.assertEqual([item[0] for item in chassis.commands], ["stop"])

    def test_motion_gate_stops_immediately_and_recovers_after_stable_frames(self):
        gate = PerceptionMotionGate(resume_valid_frames=3)
        forward = DifferentialDriveCommand(
            "forward", 0.1, 0.0, 0.0, 0.9, "valid"
        )
        lost = DifferentialDriveCommand(
            "stop", 0.0, 0.0, 0.0, 0.0, "boundary missing"
        )

        self.assertEqual(gate.filter(forward).action, "stop")
        self.assertEqual(gate.filter(forward).action, "stop")
        self.assertEqual(gate.filter(forward).action, "forward")
        self.assertEqual(gate.filter(lost).action, "stop")
        self.assertFalse(gate.get_state()["ready"])
        self.assertEqual(gate.filter(forward).action, "stop")
        self.assertEqual(gate.filter(forward).action, "stop")
        self.assertEqual(gate.filter(forward).action, "forward")

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
