import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

CAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAR))
from autodrive.control.lane_centering import RoadCenterlineEstimator, LaneCenteringController
from autodrive.perception.yolopv2_fusion import YOLOPv2FusionConfig
from autodrive.runtime.onboard import analyze, build_components


class MaskModel:
    def __init__(self):
        self.road = np.zeros((180, 320), np.uint8)
        self.road[70:, 90:230] = 1
        self.lane = np.zeros_like(self.road)
        self.fail = False
        self.last_detections = []

    def predict_masks(self, frame):
        if self.fail:
            raise RuntimeError('injected inference failure')
        return self.road.copy(), self.lane.copy()


class YOLOPv2PrimaryTests(unittest.TestCase):
    def detector(self, model=None):
        module = importlib.import_module('autodrive.perception.yolopv2_semantic')
        self.now = 0.0
        detector = module.YOLOPv2SemanticDetector(
            YOLOPv2FusionConfig(asynchronous=False, required_for_motion=True,
                                drivable_only=False, detect_objects=True, max_result_age_seconds=.3),
            model=model or MaskModel(), clock=lambda: self.now)
        self.addCleanup(detector.close)
        return detector

    def test_yolo_road_drives_virtual_control_without_classical_color_boundary(self):
        detector = self.detector()
        for color in (0, 255):
            result = analyze(detector, RoadCenterlineEstimator(), LaneCenteringController(),
                             None, np.full((480, 640, 3), color, np.uint8),
                             'center', .05, 1, boundary_tracker=None)
            self.assertTrue(result[0].valid)
            self.assertNotEqual(result[1].action, 'stop')
            self.assertEqual(result[5].source, 'yolopv2')

    def test_cloud_observation_binds_consumed_masks_to_original_frame(self):
        detector = self.detector()
        frame = np.full((32,32,3),7,np.uint8)
        detector.predict_masks(frame, captured_at=0.)
        frame[:] = 99
        # A worker may publish a newer result before the controller reads it.
        newer = detector._infer(1,frame,0.,'fp32')
        detector._publish_result(newer)
        observation = detector.get_consumed_observation()
        self.assertEqual(observation['sequence'],0)
        self.assertTrue(np.all(observation['frame']==7))
        observation['frame'][:] = 1
        self.assertTrue(np.all(detector.get_consumed_observation()['frame']==7))

    def test_near_road_without_control_preview_support_stops(self):
        model = MaskModel(); model.road[:] = 0; model.road[140:,90:230] = 1
        detector = self.detector(model)
        result = analyze(detector, RoadCenterlineEstimator(), LaneCenteringController(),
                         None, np.zeros((480,640,3), np.uint8), 'center', .05, 1)
        self.assertFalse(result[0].valid)
        self.assertEqual(result[1].action, 'stop')

    def test_centerline_cannot_rejoin_across_predicted_lane(self):
        model = MaskModel(); model.road[70:,60:260] = 1; model.lane[70:,159:161] = 1
        detector = self.detector(model)
        result = analyze(detector, RoadCenterlineEstimator(), LaneCenteringController(),
                         None, np.zeros((480,640,3), np.uint8), 'center', .05, 1)
        for x, y in result[0].centerline:
            self.assertEqual(model.lane[int(y),int(x)], 0)
        self.assertTrue(len(result[0].centerline))

    def test_perspective_warp_cannot_hide_all_image_road(self):
        from autodrive.perception.perspective import PerspectiveMapper
        model = MaskModel(); model.road[:] = 1
        detector = self.detector(model)
        mapper = PerspectiveMapper([[0,0],[1,0],[1,1],[0,1]], [[.1,0],[.9,0],[.9,1],[.1,1]])
        result = analyze(detector, RoadCenterlineEstimator(), LaneCenteringController(),
                         mapper, np.zeros((480,640,3), np.uint8), 'center', .05, 1)
        self.assertFalse(result[0].valid)
        self.assertEqual(result[1].action, 'stop')

    def test_primary_gate_configuration_rejects_nonfinite_or_string_flags(self):
        for invalid in ({'max_result_age_seconds': float('nan')}, {'required_for_motion': 'false'}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                YOLOPv2FusionConfig(**invalid)

    def test_primary_requires_object_head(self):
        from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
        with self.assertRaises(ValueError):
            YOLOPv2SemanticDetector(YOLOPv2FusionConfig(required_for_motion=True, drivable_only=False), model=MaskModel())

    def test_repeated_cached_mask_cannot_complete_resume_gate(self):
        from autodrive.control.drive_runtime import PerceptionMotionGate
        detector = self.detector()
        result = analyze(detector, RoadCenterlineEstimator(), LaneCenteringController(),
                         None, np.zeros((480, 640, 3), np.uint8), 'center', .05, 1)
        estimate, command, track = result[0], result[1], result[5]
        gate = PerceptionMotionGate(resume_valid_frames=3)
        for _ in range(5):
            self.assertEqual(gate.filter(command, estimate, track).action, 'stop')
        for _ in range(2):
            result = analyze(detector, RoadCenterlineEstimator(), LaneCenteringController(),
                             None, np.zeros((480, 640, 3), np.uint8), 'center', .05, 1)
            accepted = gate.filter(result[1], result[0], result[5])
        self.assertNotEqual(accepted.action, 'stop')

    def test_expired_result_cannot_produce_motion(self):
        detector = self.detector()
        road, lane = detector.predict_masks(np.zeros((480, 640, 3), np.uint8))
        self.now = .31
        track = detector.semantic_corridor(road, lane)
        self.assertFalse(track.valid)
        self.assertFalse(np.any(track.corridor_mask))
        self.assertIn('stale', track.reason)

    def test_capture_delay_counts_towards_mask_freshness(self):
        detector = self.detector()
        road, lane = detector.predict_masks(np.zeros((480, 640, 3), np.uint8), captured_at=-.4)
        self.assertFalse(detector.semantic_corridor(road, lane).valid)

    def test_half_precision_rejects_cpu_or_adaptive_precision(self):
        for settings in ({'device': 'cpu'}, {'device': 'cuda:0', 'adaptive_precision': True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                YOLOPv2FusionConfig(half=True, **settings)

    def test_half_precision_state_reports_actual_precision(self):
        from autodrive.perception.yolopv2_fusion import YOLOPv2FusionDetector
        d = YOLOPv2FusionDetector(YOLOPv2FusionConfig(half=True, device='cuda:0', asynchronous=False),
                                  model=MaskModel())
        self.addCleanup(d.close)
        d.predict_masks(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(d.get_state()['latest_precision'], 'fp16')

    def test_missing_ego_anchor_and_all_road_are_rejected(self):
        model = MaskModel(); detector = self.detector(model)
        for invalid in ('empty', 'distant', 'all-road'):
            model.road[:] = 0
            if invalid == 'distant': model.road[:80, 90:230] = 1
            if invalid == 'all-road': model.road[:] = 1
            road, lane = detector.predict_masks(np.zeros((480, 640, 3), np.uint8))
            with self.subTest(invalid=invalid):
                track = detector.semantic_corridor(road, lane)
                self.assertFalse(track.valid)
                self.assertFalse(np.any(track.corridor_mask))

    def test_lane_pixels_are_excluded_from_model_corridor(self):
        model = MaskModel(); model.lane[70:, 110:113] = 1
        detector = self.detector(model)
        road, lane = detector.predict_masks(np.zeros((480, 640, 3), np.uint8))
        track = detector.semantic_corridor(road, lane)
        self.assertFalse(np.any(track.corridor_mask[:, 110:113]))

    def test_worker_error_invalidates_even_recent_cached_mask(self):
        model = MaskModel(); detector = self.detector(model)
        frame = np.zeros((480, 640, 3), np.uint8)
        road, lane = detector.predict_masks(frame)
        model.fail = True
        detector._run_job((2, frame, self.now, 'fp32'))
        track = detector.semantic_corridor(road, lane)
        self.assertFalse(track.valid)
        self.assertIn('error', track.reason)

    def test_detected_near_obstacle_blocks_motion(self):
        model = MaskModel()
        model.last_detections = [{'box': [.40, .40, .60, .90], 'confidence': .9, 'class_id': 0}]
        detector = self.detector(model)
        road, lane = detector.predict_masks(np.zeros((480, 640, 3), np.uint8))
        track = detector.semantic_corridor(road, lane)
        self.assertFalse(track.valid)
        self.assertIn('obstacle', track.reason)

    def test_cloud_arbitration_rejects_classical_mode_before_hardware(self):
        config = {'cloud_arbitration': {'enabled': True}, 'outer_loop': {'enabled': True}}
        with patch('autodrive.runtime.onboard.create_chassis') as chassis:
            with self.assertRaisesRegex(ValueError, 'YOLOPv2'):
                build_components(config, True, None)
            chassis.assert_not_called()

    def test_yolo_mode_without_model_is_rejected_before_hardware_open(self):
        config = {'perception': {'mode': 'yolopv2', 'yolopv2': {'enabled': False}},
                  'outer_loop': {'enabled': False}}
        with patch('autodrive.runtime.onboard.create_chassis') as chassis:
            with self.assertRaisesRegex(ValueError, 'YOLOPv2'):
                build_components(config, True, None)
            chassis.assert_not_called()

    def test_primary_mode_cannot_fall_back_to_classical_or_corner_history(self):
        config = {'perception': {'mode': 'yolopv2', 'yolopv2': {'enabled': True}},
                  'outer_loop': {'enabled': True},
                  'safety': {'corner_continuation': {'enabled': True}}}
        with self.assertRaisesRegex(ValueError, 'corner|outer_loop'):
            build_components(config, False, None)

    def test_optional_detector_package_does_not_import_unselected_models(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys; from longtail.detectors import BaseDetector; "
            "assert not any(x in sys.modules for x in ('torch', 'transformers', 'ultralytics'))"],
            cwd=str(CAR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
