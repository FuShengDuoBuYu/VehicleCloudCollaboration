#!/usr/bin/env python3

import sys
import tempfile
import time
import queue
from types import SimpleNamespace
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml


CAR_DIR = Path(__file__).resolve().parents[1]
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.control.lane_centering import (
    DifferentialDriveCommand,
    LCCConfig,
    LaneEstimate,
    LaneCenteringController,
    RoadCenterlineEstimator,
)
from autodrive.perception.outer_loop import (
    OuterLoopBoundaryConfig,
    OuterLoopBoundaryTracker,
)
from autodrive.perception.perspective import (
    PerspectiveMapper,
    camera_pose_from_mapping,
    validate_calibration_camera_pose,
)
from autodrive.perception.yolopv2_fusion import (
    YOLOPv2FusionConfig,
    YOLOPv2FusionDetector,
)
from autodrive.perception.yolopv2_onnx import YOLOPv2ONNXDetector
from autodrive.runtime.onboard import (
    SurfaceOnlyDetector,
    VideoSource,
    _offer_latest,
    analyze,
    fuse_tracking_confidence,
)
from autodrive.perception.visualization import render_debug_frame
from autodrive.runtime.onboard import MOTOR_CONFIRMATION, validate_motor_request
from autodrive.control.drive_runtime import (
    CommandWatchdog,
    CornerContinuationConfig,
    CornerContinuationGate,
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
    def test_adaptive_yolopv2_uses_int8_only_after_stable_straight(self):
        class FakeModel:
            def __init__(self, fill):
                self.fill = fill
                self.calls = 0

            def predict_masks(self, _frame):
                self.calls += 1
                mask = np.full((18, 32), self.fill, dtype=np.uint8)
                return mask, np.zeros_like(mask)

        fp32 = FakeModel(1)
        int8 = FakeModel(0)
        detector = YOLOPv2FusionDetector(
            YOLOPv2FusionConfig(
                enabled=True,
                backend="onnxruntime",
                adaptive_precision=True,
                straight_enter_frames=2,
                asynchronous=False,
                drivable_dilate_kernel=1,
            ),
            output_width=32,
            output_height=18,
            model=fp32,
            int8_model=int8,
        )
        straight = SimpleNamespace(
            valid=True,
            source="both",
            confidence=0.8,
            yellow_hazard=False,
        )
        forward = SimpleNamespace(action="forward", steering=0.04)

        first, _ = detector.predict_masks(
            np.zeros((36, 64, 3), dtype=np.uint8)
        )
        detector.update_route_state(straight, forward)
        detector.update_route_state(straight, forward)
        quantized, _ = detector.predict_masks(
            np.zeros((36, 64, 3), dtype=np.uint8)
        )

        self.assertTrue(np.all(first == 1))
        self.assertFalse(np.any(quantized))
        self.assertEqual(fp32.calls, 1)
        self.assertEqual(int8.calls, 1)
        self.assertEqual(detector.get_state()["requested_precision"], "int8")

        turn = SimpleNamespace(
            valid=True,
            source="outer+width",
            confidence=0.8,
            yellow_hazard=False,
        )
        detector.observe_boundary(turn)
        classical = np.ones((18, 32), dtype=np.uint8)
        fused, state = detector.fuse_corridor(classical, quantized)

        self.assertEqual(state["source"], "fallback-precision-transition")
        self.assertEqual(state["precision"], "int8")
        self.assertEqual(state["requested_precision"], "fp32")
        self.assertTrue(np.array_equal(fused, classical))

        restored, _ = detector.predict_masks(
            np.zeros((36, 64, 3), dtype=np.uint8)
        )
        self.assertTrue(np.all(restored == 1))
        self.assertEqual(fp32.calls, 2)
        self.assertEqual(detector.get_state()["precision_switches"], 2)

    def test_onnx_detector_crops_padding_and_returns_drivable_mask(self):
        class TensorInfo:
            def __init__(self, name):
                self.name = name

        class FakeSession:
            def __init__(self):
                self.tensor = None

            def get_inputs(self):
                return [TensorInfo("images")]

            def get_outputs(self):
                return [TensorInfo("drivable_logits")]

            def run(self, outputs, inputs):
                self.tensor = inputs["images"]
                logits = np.zeros((1, 2, 192, 320), dtype=np.float32)
                logits[:, 1, 26:166, 80:240] = 1.0
                return [logits]

        session = FakeSession()
        detector = YOLOPv2ONNXDetector(
            weights="unused.onnx",
            session=session,
        )
        drivable, lane = detector.predict_masks(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )

        self.assertEqual(session.tensor.shape, (1, 3, 192, 320))
        self.assertEqual(drivable.shape, (180, 320))
        self.assertTrue(np.any(drivable))
        self.assertFalse(np.any(lane))

    def test_yolopv2_intersection_fusion_can_only_shrink_classical_corridor(self):
        class FakeModel:
            def predict_masks(self, _frame):
                drivable = np.zeros((18, 32), dtype=np.uint8)
                drivable[:, :16] = 1
                return drivable, np.zeros_like(drivable)

        detector = YOLOPv2FusionDetector(
            YOLOPv2FusionConfig(
                enabled=True,
                asynchronous=False,
                drivable_dilate_kernel=1,
                minimum_overlap_ratio=0.30,
            ),
            output_width=32,
            output_height=18,
            model=FakeModel(),
        )
        semantic, _ = detector.predict_masks(
            np.zeros((36, 64, 3), dtype=np.uint8)
        )
        classical = np.ones((18, 32), dtype=np.uint8)
        fused, state = detector.fuse_corridor(classical, semantic)

        self.assertEqual(state["source"], "fused-intersection")
        self.assertAlmostEqual(state["overlap_ratio"], 0.5)
        self.assertTrue(np.all(fused <= classical))
        self.assertEqual(np.count_nonzero(fused), 18 * 16)

    def test_yolopv2_low_overlap_falls_back_without_expanding_corridor(self):
        class FakeModel:
            def predict_masks(self, _frame):
                drivable = np.zeros((18, 32), dtype=np.uint8)
                drivable[:, :2] = 1
                return drivable, np.zeros_like(drivable)

        detector = YOLOPv2FusionDetector(
            YOLOPv2FusionConfig(
                enabled=True,
                asynchronous=False,
                drivable_dilate_kernel=1,
                minimum_overlap_ratio=0.30,
                required_for_motion=False,
            ),
            output_width=32,
            output_height=18,
            model=FakeModel(),
        )
        semantic, _ = detector.predict_masks(
            np.zeros((36, 64, 3), dtype=np.uint8)
        )
        classical = np.ones((18, 32), dtype=np.uint8)
        fused, state = detector.fuse_corridor(classical, semantic)

        self.assertEqual(state["source"], "fallback-low-overlap")
        self.assertTrue(state["motion_allowed"])
        self.assertTrue(np.array_equal(fused, classical))

    def test_required_yolopv2_low_overlap_blocks_motion(self):
        class FakeModel:
            def predict_masks(self, _frame):
                empty = np.zeros((18, 32), dtype=np.uint8)
                return empty, empty

        detector = YOLOPv2FusionDetector(
            YOLOPv2FusionConfig(
                enabled=True,
                asynchronous=False,
                drivable_dilate_kernel=1,
                minimum_overlap_ratio=0.30,
                required_for_motion=True,
            ),
            output_width=32,
            output_height=18,
            model=FakeModel(),
        )
        semantic, _ = detector.predict_masks(
            np.zeros((36, 64, 3), dtype=np.uint8)
        )
        fused, state = detector.fuse_corridor(
            np.ones((18, 32), dtype=np.uint8), semantic
        )

        self.assertEqual(state["source"], "required-low-overlap")
        self.assertFalse(state["motion_allowed"])
        self.assertFalse(np.any(fused))

    def test_stale_yolopv2_result_falls_back_to_classical_corridor(self):
        class FakeModel:
            def predict_masks(self, _frame):
                full = np.ones((18, 32), dtype=np.uint8)
                return full, np.zeros_like(full)

        now = [10.0]
        detector = YOLOPv2FusionDetector(
            YOLOPv2FusionConfig(
                enabled=True,
                asynchronous=False,
                max_result_age_seconds=1.0,
                drivable_dilate_kernel=1,
            ),
            output_width=32,
            output_height=18,
            model=FakeModel(),
            clock=lambda: now[0],
        )
        semantic, _ = detector.predict_masks(
            np.zeros((36, 64, 3), dtype=np.uint8)
        )
        now[0] = 11.5
        classical = np.ones((18, 32), dtype=np.uint8)
        fused, state = detector.fuse_corridor(classical, semantic)

        self.assertEqual(state["source"], "fallback-stale")
        self.assertTrue(np.array_equal(fused, classical))

    def test_diagnostic_queue_replaces_oldest_without_waiting(self):
        work_queue = queue.Queue(maxsize=1)
        work_queue.put_nowait("old")

        accepted, dropped = _offer_latest(work_queue, "latest")

        self.assertTrue(accepted)
        self.assertEqual(dropped, 1)
        self.assertEqual(work_queue.get_nowait(), "latest")
        work_queue.task_done()

    def test_video_source_uses_archived_per_frame_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "onboard_log.csv").write_text(
                "timestamp_s,action\n0.331,stop\n0.398,forward\n0.466,forward\n",
                encoding="utf-8",
            )
            source = VideoSource.__new__(VideoSource)
            source.path = root / "raw.mp4"
            source.capture = SimpleNamespace(get=lambda _property: 3)

            self.assertEqual(
                source._load_archive_timestamps(),
                [0.331, 0.398, 0.466],
            )

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

    def test_tracked_boundaries_survive_a_fragmented_surface_corridor(self):
        height, width = 180, 320
        ys = np.arange(height, dtype=np.float32)
        left = 52.0 + 0.10 * (height - 1 - ys)
        right = left + 160.0
        fragmented = np.zeros((height, width), dtype=np.uint8)
        fragmented[40:52, 70:230] = 1
        fragmented[150:158, 55:95] = 1

        surface_estimate = self.estimator.estimate(fragmented)
        boundary_estimate = self.estimator.estimate_from_boundaries(
            left, right, width
        )

        self.assertFalse(surface_estimate.valid)
        self.assertTrue(boundary_estimate.valid)
        self.assertGreater(boundary_estimate.confidence, 0.9)
        self.assertEqual(
            LaneCenteringController().update(boundary_estimate, dt=0.05).action,
            "turn-left",
        )

    def test_runtime_lookahead_keeps_early_corner_anticipation(self):
        road = make_road(shift_at_top=100)
        runtime_heading = RoadCenterlineEstimator(
            lookahead_ratio=0.50
        ).estimate(road).heading_error
        previous_heading = RoadCenterlineEstimator(
            lookahead_ratio=0.60
        ).estimate(road).heading_error
        delayed_heading = RoadCenterlineEstimator(
            lookahead_ratio=0.68
        ).estimate(road).heading_error

        self.assertGreater(runtime_heading, 0.15)
        self.assertGreater(runtime_heading, previous_heading)
        self.assertGreater(previous_heading, delayed_heading)

        runtime_estimate = RoadCenterlineEstimator(
            lookahead_ratio=0.50,
            tight_turn_lookahead_ratio=0.72,
        ).estimate(road)
        self.assertGreater(
            runtime_estimate.heading_error,
            runtime_estimate.near_heading_error,
        )

    def test_near_heading_gates_stopped_inside_wheel_mode(self):
        controller = LaneCenteringController(
            LCCConfig(
                lateral_gain=0.0,
                heading_gain=1.0,
                derivative_gain=0.0,
                steering_limit=0.70,
                steering_smoothing=0.0,
                tight_turn_near_heading_start=0.40,
                tight_turn_near_heading_full=0.65,
            )
        )
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
                tight_turn_outside_pwm=30,
                tight_turn_inside_pwm=0,
            ),
        )

        far_only = controller.update(
            LaneEstimate(
                True,
                0.8,
                heading_error=0.9,
                near_heading_error=0.2,
            ),
            dt=0.05,
        )
        self.assertEqual(far_only.tight_turn_factor, 0.0)
        self.assertEqual(driver.command_to_four_pwm(far_only), (26, 26, 10, 10))

        controller.reset()
        approaching = controller.update(
            LaneEstimate(
                True,
                0.8,
                heading_error=0.9,
                near_heading_error=0.525,
            ),
            dt=0.05,
        )
        self.assertAlmostEqual(approaching.tight_turn_factor, 0.5)
        self.assertEqual(driver.command_to_four_pwm(approaching), (28, 28, 10, 10))

        controller.reset()
        at_apex = controller.update(
            LaneEstimate(
                True,
                0.8,
                heading_error=0.9,
                near_heading_error=0.7,
            ),
            dt=0.05,
        )
        self.assertEqual(at_apex.tight_turn_factor, 1.0)
        self.assertEqual(driver.command_to_four_pwm(at_apex), (30, 30, 0, 0))

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

    def test_runtime_sharp_corner_keeps_turning_at_heading_limit(self):
        controller = LaneCenteringController(
            LCCConfig(
                min_confidence=0.35,
                lateral_gain=0.72,
                heading_gain=0.92,
                derivative_gain=0.0,
                steering_limit=0.70,
                maximum_lateral_error=0.35,
                maximum_heading_error=1.0,
                steering_smoothing=0.35,
            )
        )
        command = controller.update(
            LaneEstimate(
                True,
                0.44,
                lateral_error=-0.12,
                heading_error=1.0,
            ),
            dt=0.05,
        )

        self.assertEqual(command.action, "turn-right")
        self.assertAlmostEqual(command.steering, 0.70)

    def test_custom_heading_recovery_limit_still_stops(self):
        controller = LaneCenteringController(
            LCCConfig(maximum_heading_error=0.80)
        )
        command = controller.update(
            LaneEstimate(True, 0.9, heading_error=0.89),
            dt=0.05,
        )

        self.assertEqual(command.action, "stop")
        self.assertIn("heading error", command.reason)

    def test_corner_confidence_fusion_uses_weaker_stage_not_product(self):
        confidence = fuse_tracking_confidence(0.63, 0.46)

        self.assertAlmostEqual(confidence, 0.46)
        command = LaneCenteringController(
            LCCConfig(min_confidence=0.35, maximum_heading_error=1.0)
        ).update(
            LaneEstimate(True, confidence, heading_error=1.0),
            dt=0.05,
        )
        self.assertEqual(command.action, "turn-right")

    def test_low_boundary_confidence_still_stops_after_fusion(self):
        confidence = fuse_tracking_confidence(0.90, 0.20)
        command = LaneCenteringController(
            LCCConfig(min_confidence=0.35, maximum_heading_error=1.0)
        ).update(
            LaneEstimate(True, confidence, heading_error=1.0),
            dt=0.05,
        )

        self.assertEqual(command.action, "stop")
        self.assertIn("confidence", command.reason)

    def test_runtime_smoothing_reaches_grounded_corner_pwm_quickly(self):
        controller = LaneCenteringController(
            LCCConfig(
                base_speed=0.12,
                min_confidence=0.35,
                lateral_gain=0.0,
                heading_gain=0.92,
                derivative_gain=0.0,
                steering_limit=0.70,
                maximum_heading_error=1.0,
                steering_smoothing=0.35,
            )
        )
        command = None
        for heading in (0.16, 0.21, 0.38, 0.52, 0.73):
            command = controller.update(
                LaneEstimate(True, 0.50, heading_error=heading),
                dt=0.05,
            )
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

        self.assertIsNotNone(command)
        self.assertGreaterEqual(command.steering, 0.50)
        self.assertEqual(
            driver.command_to_four_pwm(command),
            (26, 26, 10, 10),
        )

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

    def test_runtime_width_prior_matches_calibrated_birdeye_width(self):
        """Keep the one-edge fallback coupled to measured bird's-eye width."""
        config_dir = CAR_DIR / "autodrive" / "config"
        runtime = yaml.safe_load(
            (config_dir / "onboard_runtime.yaml").read_text(encoding="utf-8")
        )
        calibration = yaml.safe_load(
            (config_dir / "onboard_calibration.yaml").read_text(encoding="utf-8")
        )
        source = np.asarray(calibration["source_points"], dtype=np.float32)
        destination = np.asarray(
            calibration["destination_points"], dtype=np.float32
        )

        self.assertGreater(source[1, 0] - source[0, 0], 0.45)
        self.assertAlmostEqual(float(destination[0, 0]), 0.28, places=6)
        self.assertAlmostEqual(float(destination[1, 0]), 0.72, places=6)
        self.assertAlmostEqual(
            runtime["outer_loop"]["expected_lane_width_ratio"],
            calibration["birdseye_lane_width_ratio"],
            places=6,
        )

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

    def test_real_pair_is_used_before_fixed_width_fallback(self):
        drivable, lane = make_outer_loop_masks(
            right_gap=False,
            right_branch=True,
        )
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(
                enabled=True,
                expected_lane_width_ratio=0.85,
                minimum_lane_width_ratio=0.24,
                maximum_lane_width_ratio=0.95,
                adaptive_lane_width=False,
                minimum_observations=8,
            )
        )

        measured = tracker.update(drivable, lane)
        estimate = self.estimator.estimate_from_boundaries(
            measured.left_curve,
            measured.right_curve,
            drivable.shape[1],
        )
        self.assertEqual(measured.source, "both")
        self.assertLess(measured.lane_width_ratio, 0.65)
        self.assertAlmostEqual(estimate.lateral_error, 0.0, delta=0.08)

        # Once the inner/right edge disappears, and only then, preserve the
        # fixed calibrated fallback over repeated temporal updates.
        outer_only = lane.copy()
        outer_only[:, 160:] = 0
        results = [tracker.update(drivable, outer_only) for _ in range(6)]
        self.assertTrue(all(result.valid for result in results))
        self.assertTrue(all(result.source == "outer+width" for result in results))
        for result in results:
            self.assertAlmostEqual(result.lane_width_ratio, 0.85, delta=0.01)

    def test_visualization_marks_inferred_right_boundary_magenta(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        mask = np.zeros((180, 320), dtype=np.uint8)
        ys = np.arange(120, 171, 5, dtype=np.int32)
        estimate = LaneEstimate(
            True,
            0.8,
            left_boundary=np.column_stack([np.full_like(ys, 70), ys]),
            right_boundary=np.column_stack([np.full_like(ys, 250), ys]),
        )
        command = LaneCenteringController().update(estimate, dt=0.2)
        rendered = render_debug_frame(
            frame,
            mask,
            mask,
            estimate,
            command,
            inference_ms=1.0,
            boundary_source="outer+width",
        )
        cyan = np.all(rendered == np.array([255, 160, 0]), axis=2)
        magenta = (
            (rendered[..., 0] > 200)
            & (rendered[..., 1] < 80)
            & (rendered[..., 2] > 200)
        )
        self.assertGreater(np.count_nonzero(cyan), 10)
        self.assertGreater(np.count_nonzero(magenta), 10)

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

    def test_visible_transverse_boundary_keeps_bounded_history_available(self):
        drivable, lane = make_outer_loop_masks(
            right_gap=False,
            right_branch=False,
        )
        tracker = OuterLoopBoundaryTracker(
            OuterLoopBoundaryConfig(
                enabled=True,
                minimum_observations=8,
                maximum_missing_frames=2,
                minimum_visible_boundary_ratio=0.01,
            )
        )
        self.assertTrue(tracker.update(drivable, lane).valid)

        full_surface = np.ones_like(drivable)
        transverse = np.zeros_like(lane)
        cv2.line(transverse, (20, 90), (300, 90), 1, 5)
        result = None
        for _ in range(5):
            result = tracker.update(full_surface, transverse)
            self.assertTrue(result.valid)
            self.assertEqual(result.source, "visible-history")
        self.assertIsNotNone(result)
        self.assertGreater(result.boundary_visible_ratio, 0.01)

        # Pixel visibility is what extends the bounded turn. Once the yellow
        # edge genuinely disappears, the already-expired normal history does
        # not permit motion.
        missing = np.zeros_like(lane)
        self.assertFalse(tracker.update(full_surface, missing).valid)

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
        estimator = RoadCenterlineEstimator(lookahead_ratio=0.50)
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
                tight_turn_start_steering=0.50,
                tight_turn_full_steering=0.70,
                tight_turn_outside_pwm=30,
                tight_turn_inside_pwm=0,
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
                DifferentialDriveCommand("turn-right", 0.6, 0, 0, 1, "test")
            ),
            (28, 28, 10, 10),
        )
        self.assertEqual(
            driver.command_to_four_pwm(
                DifferentialDriveCommand("turn-left", -0.5, 0, 0, 1, "test")
            ),
            (10, 10, 26, 26),
        )

    def test_saturated_right_turn_stops_inside_pair_without_reversing(self):
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
                tight_turn_start_steering=0.50,
                tight_turn_full_steering=0.70,
                tight_turn_outside_pwm=30,
                tight_turn_inside_pwm=0,
            ),
        )

        self.assertEqual(
            driver.command_to_four_pwm(
                DifferentialDriveCommand(
                    "turn-right", 1.0, 0.0, 0.0, 0.5, "saturated corner"
                )
            ),
            (30, 30, 0, 0),
        )

    def test_saturated_left_turn_stops_inside_pair_without_reversing(self):
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
                tight_turn_start_steering=0.50,
                tight_turn_full_steering=0.70,
                tight_turn_outside_pwm=30,
                tight_turn_inside_pwm=0,
            ),
        )

        self.assertEqual(
            driver.command_to_four_pwm(
                DifferentialDriveCommand(
                    "turn-left", -1.0, 0.0, 0.0, 0.5, "saturated corner"
                )
            ),
            (0, 0, 30, 30),
        )

    def test_tight_turn_inside_pwm_rejects_unreliable_subfloor_motion(self):
        with self.assertRaisesRegex(ValueError, "tight_turn_inside_pwm"):
            WheelMappingConfig(
                minimum_moving_pwm=10,
                tight_turn_inside_pwm=5,
            )

    def test_tight_turn_interpolation_never_emits_subfloor_pwm(self):
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
                tight_turn_outside_pwm=30,
                tight_turn_inside_pwm=0,
            ),
        )

        results = []
        for factor in np.linspace(0.0, 1.0, 101):
            pwm = driver.command_to_four_pwm(
                DifferentialDriveCommand(
                    "turn-right",
                    0.70,
                    0.0,
                    0.0,
                    0.8,
                    "tight turn",
                    float(factor),
                )
            )
            results.extend(pwm)

        self.assertFalse(any(0 < value < 10 for value in results))
        self.assertIn(0, results)
        self.assertIn(10, results)

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

    def test_motion_gate_uses_higher_confidence_only_for_resume(self):
        gate = PerceptionMotionGate(
            resume_valid_frames=3,
            resume_min_confidence=0.35,
        )
        strong = DifferentialDriveCommand(
            "turn-right", 0.7, 0.0, 0.0, 0.40, "valid corner"
        )
        weak_but_trackable = DifferentialDriveCommand(
            "turn-right", 0.7, 0.0, 0.0, 0.26, "valid single boundary"
        )

        self.assertEqual(gate.filter(strong).action, "stop")
        self.assertEqual(gate.filter(weak_but_trackable).action, "stop")
        self.assertEqual(gate.get_state()["consecutive_valid"], 0)
        self.assertEqual(gate.filter(strong).action, "stop")
        self.assertEqual(gate.filter(strong).action, "stop")
        self.assertEqual(gate.filter(strong).action, "turn-right")
        # Once moving, the LCC hard floor decides whether tracking may
        # continue; the higher confidence threshold is only for re-starting.
        self.assertEqual(
            gate.filter(weak_but_trackable).action,
            "turn-right",
        )

    def test_corner_continuation_bridges_bounded_one_boundary_fit_dropout(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                maximum_hold_seconds=1.60,
                tight_turn_factor_cap=0.40,
            )
        )
        boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        transverse_boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="visible-history",
        )
        estimate = LaneEstimate(
            True,
            0.40,
            lateral_error=-0.25,
            heading_error=0.85,
        )
        sharp = DifferentialDriveCommand(
            "turn-right", 0.68, 0.46, -0.29, 0.40, "ok", 1.0
        )
        fit_dropout = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.20,
            "road confidence below safety threshold",
        )

        self.assertEqual(gate.filter(sharp, estimate, boundary, now=0.0), sharp)
        held = gate.filter(
            fit_dropout,
            estimate,
            transverse_boundary,
            now=1.0,
        )
        self.assertEqual(held.action, "turn-right")
        self.assertAlmostEqual(held.steering, 0.35)
        self.assertEqual(held.tight_turn_factor, 0.40)
        self.assertTrue(gate.get_state(now=1.0)["holding"])

        expired = gate.filter(
            fit_dropout,
            estimate,
            transverse_boundary,
            now=2.61,
        )
        self.assertEqual(expired.action, "stop")
        self.assertFalse(gate.get_state(now=2.61)["active"])

    def test_corner_continuation_hard_limit_applies_while_controller_moves(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                maximum_hold_seconds=1.0,
                progress_timeout_seconds=0.5,
                maximum_apex_seconds=0.7,
                reacquire_steering_limit=0.35,
            )
        )
        boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        estimate = LaneEstimate(True, 0.50, heading_error=0.60)
        sharp = DifferentialDriveCommand(
            "turn-right", 0.60, 0.4, -0.2, 0.50, "ok", 0.0
        )
        weak_forward = DifferentialDriveCommand(
            "forward", 0.02, 0.1, 0.1, 0.50, "ok", 0.0
        )

        gate.filter(sharp, estimate, boundary, now=0.0)
        self.assertEqual(
            gate.filter(weak_forward, estimate, boundary, now=0.9).action,
            "turn-right",
        )
        expired = gate.filter(
            weak_forward, estimate, boundary, now=1.01
        )
        self.assertEqual(expired.action, "stop")
        self.assertEqual(
            expired.reason,
            "corner continuation hard time limit reached",
        )
        self.assertFalse(gate.get_state(now=1.01)["active"])

    def test_corner_continuation_never_overrides_yellow_hazard(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(enabled=True)
        )
        clear_boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        yellow_boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=True,
            source="outer+width",
        )
        estimate = LaneEstimate(True, 0.40, heading_error=0.90)
        sharp = DifferentialDriveCommand(
            "turn-right", 0.68, 0.46, -0.29, 0.40, "ok"
        )
        yellow_stop = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.0,
            "yellow boundary entered the vehicle safety zone",
        )

        gate.filter(sharp, estimate, clear_boundary, now=0.0)
        applied = gate.filter(yellow_stop, estimate, yellow_boundary, now=0.1)
        self.assertEqual(applied.action, "stop")
        self.assertFalse(gate.get_state(now=0.1)["active"])

    def test_corner_continuation_bridges_visible_transverse_boundary(self):
        """Regression for run 20260803_151551 sample 258."""
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                minimum_transverse_boundary_ratio=0.01,
            )
        )
        fitted_boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
            reason="ok",
            boundary_visible_ratio=0.18,
        )
        transverse_boundary = SimpleNamespace(
            valid=False,
            yellow_hazard=False,
            source="visible-history",
            reason="boundary remains visible outside x(y) fit",
            # The physical frame measured 0.23771.
            boundary_visible_ratio=0.23771,
        )
        estimate = LaneEstimate(
            True,
            0.35,
            lateral_error=0.42,
            heading_error=0.57,
            near_heading_error=0.55,
        )
        sharp = DifferentialDriveCommand(
            "turn-right", 0.663, 0.45, -0.28, 0.35, "ok", 1.0
        )
        fit_failure = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.27,
            "boundary remains visible outside x(y) fit",
        )

        gate.filter(sharp, estimate, fitted_boundary, now=0.0)
        bridged = gate.filter(
            fit_failure,
            LaneEstimate(
                False,
                0.27,
                reason="boundary remains visible outside x(y) fit",
            ),
            transverse_boundary,
            now=0.07,
        )
        self.assertEqual(bridged.action, "turn-right")
        self.assertAlmostEqual(bridged.steering, 0.35)
        self.assertEqual(bridged.tight_turn_factor, 1.0)
        self.assertTrue(gate.get_state(now=0.07)["active"])

    def test_corner_continuation_rejects_unsubstantiated_transverse_history(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                minimum_transverse_boundary_ratio=0.01,
            )
        )
        fitted_boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        weak_history = SimpleNamespace(
            valid=False,
            yellow_hazard=False,
            source="visible-history",
            reason="boundary remains visible outside x(y) fit",
            boundary_visible_ratio=0.005,
        )
        estimate = LaneEstimate(True, 0.40, heading_error=0.70)
        sharp = DifferentialDriveCommand(
            "turn-right", 0.65, 0.4, -0.2, 0.40, "ok"
        )
        failed = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.0,
            "boundary remains visible outside x(y) fit",
        )

        gate.filter(sharp, estimate, fitted_boundary, now=0.0)
        applied = gate.filter(failed, estimate, weak_history, now=0.1)
        self.assertEqual(applied.action, "stop")
        self.assertFalse(gate.get_state(now=0.1)["active"])

    def test_corner_continuation_extends_only_while_heading_improves(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                maximum_hold_seconds=2.80,
                progress_timeout_seconds=0.50,
                minimum_heading_progress=0.05,
            )
        )
        boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        sharp = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 1.0
        )
        stopped = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.40,
            "lateral error exceeds recovery limit",
        )

        gate.filter(
            sharp,
            LaneEstimate(True, 0.50, heading_error=0.90),
            boundary,
            now=0.0,
        )
        for now, heading in ((0.1, 0.80), (0.5, 0.70), (0.9, 0.60), (1.3, 0.50)):
            held = gate.filter(
                stopped,
                LaneEstimate(True, 0.40, heading_error=heading),
                boundary,
                now=now,
            )
            self.assertEqual(held.action, "turn-right")

        # Still below the 2.8 s hard cap, but no meaningful heading progress
        # has occurred for longer than the configured 0.5 s window.
        no_progress = gate.filter(
            stopped,
            LaneEstimate(True, 0.40, heading_error=0.49),
            boundary,
            now=1.81,
        )
        self.assertEqual(no_progress.action, "stop")
        self.assertEqual(
            gate.get_state(now=1.81)["last_reason"],
            "corner continuation stopped making progress",
        )

    def test_corner_continuation_accepts_lateral_recovery_progress(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                maximum_hold_seconds=3.0,
                progress_timeout_seconds=0.50,
                minimum_heading_progress=0.05,
                minimum_lateral_progress=0.03,
            )
        )
        boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        sharp = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 1.0
        )
        stopped = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.40,
            "lateral error exceeds recovery limit",
        )

        gate.filter(
            sharp,
            LaneEstimate(
                True, 0.50, lateral_error=0.60, heading_error=0.30
            ),
            boundary,
            now=0.0,
        )
        # Heading has stopped improving, but lateral error is converging by
        # more than the configured increment on every observation.
        for now, lateral in ((0.1, 0.55), (0.5, 0.50), (0.9, 0.45), (1.3, 0.40)):
            held = gate.filter(
                stopped,
                LaneEstimate(
                    True,
                    0.40,
                    lateral_error=lateral,
                    heading_error=0.30,
                ),
                boundary,
                now=now,
            )
            self.assertEqual(held.action, "turn-right")

        no_progress = gate.filter(
            stopped,
            LaneEstimate(
                True, 0.40, lateral_error=0.39, heading_error=0.30
            ),
            boundary,
            now=1.81,
        )
        self.assertEqual(no_progress.action, "stop")
        self.assertEqual(
            gate.get_state(now=1.81)["last_reason"],
            "corner continuation stopped making progress",
        )

    def test_corner_apex_is_latched_for_a_bounded_minimum_duration(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                maximum_hold_seconds=2.80,
                progress_timeout_seconds=1.00,
                apex_trigger_factor=0.50,
                apex_near_heading_trigger=0.43,
                minimum_apex_seconds=0.32,
                tight_turn_factor_cap=0.0,
            )
        )
        boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        estimate = LaneEstimate(True, 0.50, heading_error=0.80)
        trigger = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 0.56
        )
        recoverable_stop = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.40,
            "lateral error exceeds recovery limit",
        )

        armed = gate.filter(trigger, estimate, boundary, now=0.0)
        self.assertEqual(armed.tight_turn_factor, 1.0)
        self.assertEqual(
            gate.filter(
                recoverable_stop, estimate, boundary, now=0.31
            ).tight_turn_factor,
            1.0,
        )
        advancing = gate.filter(
            recoverable_stop, estimate, boundary, now=0.33
        )
        self.assertEqual(advancing.action, "turn-right")
        self.assertEqual(advancing.tight_turn_factor, 1.0)
        advancing = gate.filter(
            recoverable_stop, estimate, boundary, now=0.40
        )
        self.assertEqual(advancing.action, "turn-right")
        self.assertEqual(advancing.tight_turn_factor, 0.0)

    def test_corner_apex_rejects_single_provisional_both_lateral_stop(self):
        """Regression for run 20260803_193400 samples 792 through 796."""
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                enter_steering_threshold=0.40,
                apex_commit_delay_seconds=0.30,
                minimum_apex_seconds=0.60,
                maximum_apex_seconds=0.70,
                apex_exit_near_heading=0.30,
                apex_exit_valid_frames=2,
            )
        )
        outer = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        both = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="both",
        )
        approach = DifferentialDriveCommand(
            "turn-right", 0.462, 0.35, -0.16, 0.42, "ok", 0.0
        )
        stop = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.37,
            "lateral error exceeds recovery limit",
        )
        turning = LaneEstimate(
            True,
            0.42,
            lateral_error=0.20,
            heading_error=0.39,
            near_heading_error=0.15,
        )
        misfit = LaneEstimate(
            True,
            0.37,
            lateral_error=0.63,
            heading_error=-0.71,
            near_heading_error=-0.12,
        )

        gate.filter(approach, turning, outer, now=0.0)
        armed = gate.filter(stop, misfit, outer, now=0.31)
        self.assertEqual(armed.action, "turn-right")
        self.assertEqual(armed.tight_turn_factor, 1.0)

        # The physical failure contained exactly this early, one-frame both.
        provisional = gate.filter(stop, misfit, both, now=0.50)
        self.assertEqual(provisional.action, "turn-right")
        self.assertEqual(provisional.tight_turn_factor, 1.0)
        self.assertTrue(gate.get_state(now=0.50)["apex_active"])
        self.assertEqual(
            gate.get_state(now=0.50)["apex_both_valid_count"],
            0,
        )

        # After the minimum, measured-pair recovery must be consecutive.
        first_both = gate.filter(stop, misfit, both, now=0.92)
        self.assertEqual(first_both.action, "turn-right")
        self.assertEqual(
            gate.get_state(now=0.92)["apex_both_valid_count"],
            1,
        )
        reset_by_outer = gate.filter(stop, misfit, outer, now=0.95)
        self.assertEqual(reset_by_outer.action, "turn-right")
        self.assertEqual(
            gate.get_state(now=0.95)["apex_both_valid_count"],
            0,
        )
        gate.filter(stop, misfit, both, now=0.97)
        confirmed = gate.filter(stop, misfit, both, now=1.00)
        self.assertEqual(confirmed.action, "stop")
        self.assertFalse(gate.get_state(now=1.00)["active"])

    def test_corner_apex_can_arm_on_recoverable_stop_geometry(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                apex_trigger_factor=0.50,
                apex_near_heading_trigger=0.43,
                minimum_apex_seconds=0.32,
                tight_turn_factor_cap=0.0,
            )
        )
        boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        approach = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 0.19
        )
        gate.filter(
            approach,
            LaneEstimate(
                True, 0.50, heading_error=0.85, near_heading_error=0.40
            ),
            boundary,
            now=0.0,
        )
        recoverable_stop = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.30,
            "road confidence below safety threshold",
        )

        latched = gate.filter(
            recoverable_stop,
            LaneEstimate(
                True, 0.30, heading_error=0.75, near_heading_error=0.50
            ),
            boundary,
            now=0.1,
        )
        self.assertEqual(latched.action, "turn-right")
        self.assertEqual(latched.tight_turn_factor, 1.0)

    def test_corner_apex_does_not_arm_when_both_boundaries_are_visible(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(enabled=True)
        )
        outer_only = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        both = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="both",
        )
        approach = DifferentialDriveCommand(
            "turn-right", 0.55, 0.0, 0.0, 0.50, "one-edge approach", 0.0
        )
        visible_corner = DifferentialDriveCommand(
            "turn-right", 0.60, 0.0, 0.0, 0.60, "two-edge corner", 0.0
        )

        gate.filter(
            approach,
            LaneEstimate(
                True, 0.50, heading_error=0.50, near_heading_error=0.20
            ),
            outer_only,
            now=0.0,
        )
        applied = gate.filter(
            visible_corner,
            LaneEstimate(
                True, 0.60, heading_error=0.80, near_heading_error=0.55
            ),
            both,
            now=0.1,
        )
        self.assertEqual(applied.tight_turn_factor, 0.0)
        self.assertFalse(gate.get_state(now=0.1)["apex_active"])

        # Sustained two-boundary tracking also retires the one-boundary
        # continuation state, so a later role change cannot reuse stale turn
        # commitment.
        gate.filter(visible_corner, LaneEstimate(True, 0.60), both, now=0.2)
        gate.filter(visible_corner, LaneEstimate(True, 0.60), both, now=0.3)
        self.assertFalse(gate.get_state(now=0.3)["active"])

    def test_corner_apex_waits_for_alignment_after_minimum_duration(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                apex_trigger_factor=0.50,
                apex_near_heading_trigger=0.43,
                minimum_apex_seconds=0.40,
                maximum_apex_seconds=0.70,
                apex_exit_near_heading=0.30,
                apex_exit_valid_frames=2,
                tight_turn_factor_cap=0.0,
            )
        )
        boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        trigger = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 0.56
        )
        advancing = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 0.0
        )

        armed = gate.filter(
            trigger,
            LaneEstimate(
                True, 0.50, heading_error=0.85, near_heading_error=0.52
            ),
            boundary,
            now=0.0,
        )
        self.assertEqual(armed.tight_turn_factor, 1.0)
        # Elapsed minimum alone is not enough while the visible edge remains
        # transverse to the new straight.
        still_turning = gate.filter(
            advancing,
            LaneEstimate(
                True, 0.50, heading_error=0.65, near_heading_error=0.38
            ),
            boundary,
            now=0.41,
        )
        self.assertEqual(still_turning.tight_turn_factor, 1.0)

        first_aligned = gate.filter(
            advancing,
            LaneEstimate(
                True, 0.50, heading_error=0.45, near_heading_error=0.29
            ),
            boundary,
            now=0.50,
        )
        self.assertEqual(first_aligned.tight_turn_factor, 1.0)
        exited = gate.filter(
            advancing,
            LaneEstimate(
                True, 0.50, heading_error=0.40, near_heading_error=0.28
            ),
            boundary,
            now=0.57,
        )
        self.assertEqual(exited.tight_turn_factor, 0.0)
        self.assertFalse(gate.get_state(now=0.57)["apex_active"])

    def test_corner_apex_has_a_hard_time_limit_without_alignment(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                minimum_apex_seconds=0.40,
                maximum_apex_seconds=0.70,
                apex_exit_near_heading=0.30,
                apex_exit_valid_frames=2,
                tight_turn_factor_cap=0.0,
            )
        )
        boundary = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        estimate = LaneEstimate(
            True, 0.50, heading_error=0.80, near_heading_error=0.55
        )
        trigger = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 0.60
        )
        advancing = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 0.0
        )

        self.assertEqual(
            gate.filter(trigger, estimate, boundary, now=0.0).tight_turn_factor,
            1.0,
        )
        self.assertEqual(
            gate.filter(advancing, estimate, boundary, now=0.69).tight_turn_factor,
            1.0,
        )
        self.assertEqual(
            gate.filter(advancing, estimate, boundary, now=0.71).tight_turn_factor,
            0.0,
        )

    def test_single_fresh_boundary_releases_aligned_apex_to_lcc(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                exit_valid_frames=2,
                minimum_apex_seconds=0.40,
                maximum_apex_seconds=0.70,
                apex_exit_near_heading=0.30,
                apex_exit_valid_frames=2,
            )
        )
        outer_only = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        turning_estimate = LaneEstimate(
            True, 0.50, heading_error=0.80, near_heading_error=0.55
        )
        trigger = DifferentialDriveCommand(
            "turn-right", 0.70, 0.46, -0.30, 0.50, "ok", 0.60
        )
        left_correction = DifferentialDriveCommand(
            "turn-left", -0.30, 0.0, 0.0, 0.70, "two-edge recenter", 0.0
        )

        gate.filter(trigger, turning_estimate, outer_only, now=0.0)
        debounced = gate.filter(
            left_correction,
            LaneEstimate(
                True, 0.70, heading_error=-0.20, near_heading_error=-0.15
            ),
            outer_only,
            now=0.41,
        )
        self.assertEqual(debounced.action, "turn-right")
        self.assertEqual(debounced.tight_turn_factor, 1.0)

        released = gate.filter(
            left_correction,
            LaneEstimate(
                True, 0.72, heading_error=-0.18, near_heading_error=-0.12
            ),
            outer_only,
            now=0.48,
        )
        self.assertEqual(released.action, "turn-left")
        self.assertEqual(released.tight_turn_factor, 0.0)
        self.assertFalse(gate.get_state(now=0.48)["active"])

    def test_second_corner_trace_hands_back_after_apex_alignment(self):
        """A fresh one-edge straight must not retain the old corner arc."""
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                enter_steering_threshold=0.50,
                exit_steering_threshold=0.18,
                exit_min_confidence=0.35,
                exit_valid_frames=3,
                maximum_hold_seconds=6.20,
                minimum_apex_seconds=0.40,
                maximum_apex_seconds=0.70,
                apex_exit_near_heading=0.30,
                apex_exit_valid_frames=2,
                tight_turn_factor_cap=0.0,
            )
        )
        outer = SimpleNamespace(
            valid=True, yellow_hazard=False, source="outer+width"
        )
        both = SimpleNamespace(valid=True, yellow_hazard=False, source="both")
        inner = SimpleNamespace(
            valid=True, yellow_hazard=False, source="inner+width"
        )

        trace = (
            # now, steering, lateral, heading, near heading, boundary
            (18.4019, +0.525, -0.141, +0.708, +0.452, outer),
            (18.4661, +0.527, +0.028, +0.552, +0.340, outer),
            (18.5333, +0.498, +0.162, +0.397, +0.233, outer),
            (18.6025, +0.444, +0.185, +0.305, +0.186, outer),
            (18.6655, +0.356, +0.201, +0.178, +0.126, outer),
            (18.7337, +0.236, +0.202, +0.029, +0.053, outer),
            (18.8020, +0.025, +0.181, -0.237, -0.016, outer),
            (18.8666, -0.046, +0.126, -0.190, -0.138, outer),
            (18.9336, -0.108, +0.095, -0.228, -0.192, outer),
            (19.0016, -0.135, +0.074, -0.221, -0.214, both),
            (19.0664, +0.046, -0.322, +0.407, +0.130, inner),
        )
        applied = []
        for now, steering, lateral, heading, near_heading, boundary in trace:
            action = (
                "turn-right"
                if steering > 0.07
                else "turn-left"
                if steering < -0.07
                else "forward"
            )
            command = DifferentialDriveCommand(
                action,
                steering,
                0.0,
                0.0,
                0.53,
                "ok",
                1.0 if now == trace[0][0] else 0.0,
            )
            estimate = LaneEstimate(
                True,
                0.53,
                lateral_error=lateral,
                heading_error=heading,
                near_heading_error=near_heading,
            )
            applied.append(gate.filter(command, estimate, boundary, now=now))

        self.assertTrue(all(item.action == "turn-right" for item in applied[:7]))
        self.assertEqual(applied[7].action, "forward")
        self.assertLess(applied[7].steering, 0.0)
        self.assertEqual(applied[8].action, "turn-left")
        self.assertFalse(gate.get_state(now=19.0664)["active"])

        lateral_stop = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.45,
            "lateral error exceeds recovery limit",
        )
        bridged = gate.filter(
            lateral_stop,
            LaneEstimate(
                True,
                0.45,
                lateral_error=-0.380,
                heading_error=+0.512,
                near_heading_error=+0.188,
            ),
            inner,
            now=19.1337,
        )
        self.assertEqual(bridged.action, "stop")

    def test_fourth_corner_role_swap_honors_opposite_lateral_stop(self):
        """Post-apex continuation cannot drive farther across centre."""
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                maximum_hold_seconds=8.0,
                progress_timeout_seconds=1.0,
                apex_near_heading_trigger=0.40,
                minimum_apex_seconds=0.50,
                maximum_apex_seconds=0.70,
                reacquire_steering_limit=0.35,
                tight_turn_factor_cap=0.0,
            )
        )
        outer = SimpleNamespace(
            valid=True, yellow_hazard=False, source="outer+width"
        )
        both = SimpleNamespace(valid=True, yellow_hazard=False, source="both")
        inner = SimpleNamespace(
            valid=True, yellow_hazard=False, source="inner+width"
        )
        sharp = DifferentialDriveCommand(
            "turn-right", 0.54, 0.39, -0.20, 0.48, "ok", 0.0
        )
        recoverable_stop = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.24,
            "road confidence below safety threshold",
        )

        gate.filter(
            sharp,
            LaneEstimate(
                True,
                0.48,
                lateral_error=0.15,
                heading_error=0.52,
                near_heading_error=0.12,
            ),
            outer,
            now=0.0,
        )
        # A misleading one-edge heading minimum used to start a timeout that
        # kept running even after the raw controller recovered.
        gate.filter(
            recoverable_stop,
            LaneEstimate(
                True,
                0.24,
                lateral_error=0.21,
                heading_error=0.64,
                near_heading_error=0.25,
            ),
            outer,
            now=0.20,
        )
        gate.filter(
            recoverable_stop,
            LaneEstimate(
                True,
                0.24,
                lateral_error=0.46,
                heading_error=0.01,
                near_heading_error=0.02,
            ),
            outer,
            now=0.60,
        )

        recovered_outer = DifferentialDriveCommand(
            "turn-right", 0.55, 0.39, -0.21, 0.51, "ok", 0.0
        )
        gate.filter(
            recovered_outer,
            LaneEstimate(
                True,
                0.51,
                lateral_error=-0.12,
                heading_error=0.71,
                near_heading_error=0.42,
            ),
            outer,
            now=1.00,
        )
        # The weak transitional commands use a second physical edge. They
        # must not replace the strong outer-edge command that grounded the
        # committed right turn.
        for now, command, estimate, boundary in (
            (
                1.40,
                DifferentialDriveCommand(
                    "turn-right", 0.23, 0.23, -0.02, 0.56, "ok", 0.0
                ),
                LaneEstimate(
                    True,
                    0.56,
                    lateral_error=0.14,
                    heading_error=-0.05,
                    near_heading_error=0.02,
                ),
                both,
            ),
            (
                1.60,
                DifferentialDriveCommand(
                    "turn-right", 0.20, 0.22, 0.0, 0.39, "ok", 0.0
                ),
                LaneEstimate(
                    True,
                    0.39,
                    lateral_error=-0.30,
                    heading_error=0.43,
                    near_heading_error=0.14,
                ),
                inner,
            ),
        ):
            self.assertEqual(
                gate.filter(command, estimate, boundary, now=now).action,
                "turn-right",
            )

        bridged = gate.filter(
            DifferentialDriveCommand(
                "stop",
                0.0,
                0.0,
                0.0,
                0.55,
                "lateral error exceeds recovery limit",
            ),
            LaneEstimate(
                True,
                0.55,
                lateral_error=-0.42,
                heading_error=0.37,
                near_heading_error=0.14,
            ),
            inner,
            now=1.67,
        )
        self.assertEqual(bridged.action, "stop")
        self.assertFalse(gate.get_state(now=1.67)["active"])

    def test_committed_corner_uses_fixed_apex_fallback(self):
        """Regression for run 20260803_165757 second physical corner."""
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                apex_near_heading_trigger=0.40,
                apex_commit_delay_seconds=0.30,
                minimum_apex_seconds=0.55,
                maximum_apex_seconds=0.70,
            )
        )
        outer = SimpleNamespace(
            valid=True, yellow_hazard=False, source="outer+width"
        )
        detected_corner = DifferentialDriveCommand(
            "turn-right", 0.52, 0.38, -0.19, 0.32, "ok", 0.0
        )
        estimate = LaneEstimate(
            True,
            0.32,
            lateral_error=0.23,
            heading_error=0.40,
            near_heading_error=0.24,
        )

        entered = gate.filter(detected_corner, estimate, outer, now=0.0)
        self.assertEqual(entered.tight_turn_factor, 0.0)
        before_delay = gate.filter(
            detected_corner,
            estimate,
            outer,
            now=0.29,
        )
        self.assertEqual(before_delay.tight_turn_factor, 0.0)

        recovery_stop = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.24,
            "road confidence below safety threshold",
        )
        standardized = gate.filter(
            recovery_stop,
            LaneEstimate(
                True,
                0.24,
                lateral_error=0.37,
                heading_error=0.09,
                near_heading_error=0.06,
            ),
            outer,
            now=0.31,
        )
        self.assertEqual(standardized.action, "turn-right")
        self.assertEqual(standardized.tight_turn_factor, 1.0)
        state = gate.get_state(now=0.31)
        self.assertTrue(state["apex_active"])
        self.assertEqual(
            state["apex_trigger_reason"],
            "committed-corner delay",
        )

    def test_delayed_apex_survives_control_gap_and_history_frame(self):
        """Regression for run 20260803_174558 samples 91 through 92."""
        gate = CornerContinuationGate(
            CornerContinuationConfig(
                enabled=True,
                apex_commit_delay_seconds=0.30,
                minimum_apex_seconds=0.55,
                maximum_apex_seconds=0.70,
            )
        )
        fresh_outer = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="outer+width",
        )
        history = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="history",
        )
        committed = DifferentialDriveCommand(
            "turn-right", 0.59632, 0.42, -0.24, 0.46687, "ok", 0.0
        )
        entered = gate.filter(
            committed,
            LaneEstimate(
                True,
                0.46687,
                lateral_error=0.24258,
                heading_error=0.59710,
                near_heading_error=0.14485,
            ),
            fresh_outer,
            now=7.0622,
        )
        self.assertEqual(entered.tight_turn_factor, 0.0)

        # The next processed camera frame arrived 0.801 s later and had only
        # history geometry. It must still execute the standardized apex turn.
        after_gap = gate.filter(
            DifferentialDriveCommand(
                "turn-right", 0.66371, 0.45, -0.28, 0.36415, "ok", 0.0
            ),
            LaneEstimate(
                True,
                0.36415,
                lateral_error=0.16364,
                heading_error=0.64048,
                near_heading_error=0.22302,
            ),
            history,
            now=7.8632,
        )
        self.assertEqual(after_gap.action, "turn-right")
        self.assertEqual(after_gap.tight_turn_factor, 1.0)
        self.assertEqual(
            gate.get_state(now=7.8632)["apex_trigger_reason"],
            "committed-corner delay",
        )

    def test_motion_gate_rejects_role_swap_estimate_jump(self):
        gate = PerceptionMotionGate(
            resume_valid_frames=3,
            resume_min_confidence=0.35,
            maximum_lateral_jump=0.32,
            maximum_heading_jump=0.45,
            require_consistent_source=True,
        )
        both = SimpleNamespace(valid=True, source="both")
        inner = SimpleNamespace(valid=True, source="inner+width")
        command = DifferentialDriveCommand(
            "forward", 0.04, 0.0, 0.0, 0.55, "ok"
        )

        for lateral in (0.07, 0.08):
            self.assertEqual(
                gate.filter(
                    command,
                    LaneEstimate(True, 0.55, lateral_error=lateral),
                    both,
                ).action,
                "stop",
            )
        self.assertEqual(
            gate.filter(
                command,
                LaneEstimate(True, 0.55, lateral_error=0.09),
                both,
            ).action,
            "forward",
        )

        rejected = gate.filter(
            command,
            LaneEstimate(
                True,
                0.45,
                lateral_error=-0.322,
                heading_error=0.407,
            ),
            inner,
        )
        self.assertEqual(rejected.action, "stop")
        self.assertIn("estimate jumped", rejected.reason)
        self.assertFalse(gate.get_state()["ready"])

        # The new source becomes a candidate, but still needs a complete stable
        # validation window before motion resumes.
        for lateral in (-0.31, -0.30):
            self.assertEqual(
                gate.filter(
                    command,
                    LaneEstimate(True, 0.45, lateral_error=lateral),
                    inner,
                ).action,
                "stop",
            )
        self.assertEqual(
            gate.filter(
                command,
                LaneEstimate(True, 0.45, lateral_error=-0.29),
                inner,
            ).action,
            "forward",
        )

    def test_motion_gate_history_jump_does_not_poison_fresh_baseline(self):
        """Regression for run 20260803_180818 samples 241 through 243."""
        gate = PerceptionMotionGate(
            resume_valid_frames=2,
            resume_min_confidence=0.35,
            maximum_lateral_jump=0.32,
            maximum_heading_jump=0.45,
            require_consistent_source=True,
        )
        outer = SimpleNamespace(valid=True, source="outer+width")
        history = SimpleNamespace(valid=True, source="history")
        command = DifferentialDriveCommand(
            "turn-left", -0.15, 0.0, 0.0, 0.55, "ok"
        )

        self.assertEqual(
            gate.filter(
                command,
                LaneEstimate(True, 0.55, heading_error=-0.18),
                outer,
            ).action,
            "stop",
        )
        self.assertEqual(
            gate.filter(
                command,
                LaneEstimate(True, 0.55, heading_error=-0.17),
                outer,
            ).action,
            "turn-left",
        )

        # The extrapolated frame jumps by more than the normal fresh-geometry
        # threshold, but neither stops motion nor replaces the outer baseline.
        self.assertEqual(
            gate.filter(
                command,
                LaneEstimate(True, 0.40, heading_error=-0.68),
                history,
            ).action,
            "turn-left",
        )
        fresh = gate.filter(
            command,
            LaneEstimate(True, 0.55, heading_error=-0.16),
            outer,
        )
        self.assertEqual(fresh.action, "turn-left")
        self.assertTrue(gate.get_state()["ready"])

    def test_corner_continuation_does_not_mask_two_boundary_lateral_stop(self):
        gate = CornerContinuationGate(
            CornerContinuationConfig(enabled=True)
        )
        both = SimpleNamespace(
            valid=True,
            yellow_hazard=False,
            source="both",
        )
        estimate = LaneEstimate(True, 0.60, lateral_error=0.40)
        sharp = DifferentialDriveCommand(
            "turn-right", 0.68, 0.46, -0.29, 0.60, "ok"
        )
        lateral_stop = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            0.60,
            "lateral error exceeds recovery limit",
        )

        gate.filter(sharp, estimate, both, now=0.0)
        applied = gate.filter(lateral_stop, estimate, both, now=0.1)
        self.assertEqual(applied.action, "stop")
        self.assertFalse(gate.get_state(now=0.1)["active"])

    def test_runtime_corner_confidence_does_not_stop_controller(self):
        controller = LaneCenteringController(
            LCCConfig(
                min_confidence=0.25,
                maximum_lateral_error=0.35,
                maximum_heading_error=1.0,
            )
        )

        command = controller.update(
            LaneEstimate(
                True,
                0.2505,
                lateral_error=-0.28,
                heading_error=1.0,
            ),
            dt=0.05,
        )
        self.assertEqual(command.action, "turn-right")

        stopped = controller.update(
            LaneEstimate(
                True,
                0.20,
                lateral_error=-0.28,
                heading_error=1.0,
            ),
            dt=0.05,
        )
        self.assertEqual(stopped.action, "stop")
        self.assertIn("confidence", stopped.reason)

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
