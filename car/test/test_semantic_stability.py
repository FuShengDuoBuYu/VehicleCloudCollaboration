"""Temporal perception checks, including immediate rejection of current hazards."""
import sys
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.perception.semantic_stability import SemanticStabilizer
from car.test import test_yolopv2_primary as primary_tests
from autodrive.runtime.onboard import analyze
from autodrive.control.lane_centering import RoadCenterlineEstimator, LaneCenteringController


class SemanticStabilityTests(unittest.TestCase):
    def road(self, right=75):
        mask = np.zeros((80,100), np.uint8)
        mask[20:,25:right] = 1
        return mask

    def test_edge_flicker_reduced_without_adding_unobserved_road(self):
        filter_ = SemanticStabilizer()
        raw, smooth = [], []
        for i in range(40):
            mask = self.road(75 + (3 if i%2 else 0))
            out = filter_.update(mask, i, i*.06)
            self.assertFalse(np.any(out & (mask == 0)))
            raw.append(mask); smooth.append(out)
        variation = lambda masks: sum(np.count_nonzero(a != b) for a,b in zip(masks[5:],masks[6:]))
        self.assertLess(variation(smooth), variation(raw)*.75)

    def test_new_excluded_hole_is_never_hidden(self):
        filter_ = SemanticStabilizer()
        for i in range(5): filter_.update(self.road(), i, i*.07)
        mask = self.road(); mask[55:75,40:60] = 0
        self.assertFalse(np.any(filter_.update(mask, 5, .35)[55:75,40:60]))

    def test_cached_sequence_cannot_confirm_expansion(self):
        filter_ = SemanticStabilizer()
        filter_.update(self.road(), 1, .1)
        expanded = self.road(78)
        first = filter_.update(expanded, 2, .16)
        for _ in range(20):
            np.testing.assert_array_equal(filter_.update(expanded, 2, .16), first)
        # A persistent new edge is accepted promptly on genuinely new frames.
        for seq in range(3,7): out = filter_.update(expanded, seq, .16+(seq-2)*.06)
        self.assertTrue(np.all(out[20:,75:78]))

    def test_old_history_and_scene_change_do_not_create_a_ghost_corridor(self):
        filter_ = SemanticStabilizer()
        filter_.update(self.road(), 1, .1)
        moved = np.roll(self.road(),25,axis=1)
        np.testing.assert_array_equal(filter_.update(moved, 2, .16), moved)
        np.testing.assert_array_equal(filter_.update(self.road(80), 3, 1.), self.road(80))
        np.testing.assert_array_equal(filter_.update(self.road(), 0, 0.), self.road())

    def test_invalid_filter_parameters_fail(self):
        for config in ({'time_constant_seconds':0}, {'max_gap_seconds':float('nan')},
                       {'threshold':1}, {'enabled':'false'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                SemanticStabilizer(**config)


class SemanticDisplayTests(unittest.TestCase):
    detector = primary_tests.YOLOPv2PrimaryTests.detector

    def test_overlay_uses_consumed_frame_instead_of_newer_camera_frame(self):
        detector = self.detector()
        original = np.full((480,640,3),7,np.uint8)
        masks = detector.predict_masks(original, captured_at=0.)
        with patch.object(detector, 'predict_masks', return_value=masks), \
             patch('autodrive.runtime.onboard.render_debug_frame', return_value=original) as render:
            analyze(detector, RoadCenterlineEstimator(), LaneCenteringController(),
                    None, np.full_like(original,99),'center', .05, 1.)
        np.testing.assert_array_equal(render.call_args.args[0], original)

    def test_stabilization_never_overrides_obstacle_missing_road_or_stale_veto(self):
        from autodrive.perception.semantic_stability import SemanticStabilizer
        model = primary_tests.MaskModel()
        detector = self.detector(model)
        detector.stabilizer = SemanticStabilizer()
        frame = np.zeros((480,640,3), np.uint8)
        for i in range(5):
            self.now = i*.05
            masks = detector.predict_masks(frame)
            self.assertTrue(detector.semantic_corridor(*masks).valid)
        model.last_detections = [{'box':[.4,.4,.6,.9],'confidence':.9,'class_id':0}]
        masks = detector.predict_masks(frame)
        track = detector.semantic_corridor(*masks)
        self.assertFalse(track.valid); self.assertIn('obstacle',track.reason)
        self.assertEqual(detector.stabilizer.state['mode'],'reset')
        model.last_detections = []
        model.road[140:] = 0
        masks = detector.predict_masks(frame)
        self.assertFalse(detector.semantic_corridor(*masks).valid)
        model.road[70:,90:230] = 1
        masks = detector.predict_masks(frame)
        self.now += .31
        track = detector.semantic_corridor(*masks)
        self.assertFalse(track.valid); self.assertIn('stale',track.reason)

    def test_archive_keeps_lossless_input_once_per_consumed_sequence(self):
        from autodrive.runtime.onboard import RunArchive
        with tempfile.TemporaryDirectory() as temporary:
            archive = RunArchive(temporary,True,20,'dryrun')
            image = np.arange(48*64*3,dtype=np.uint8).reshape(48,64,3)
            mask = np.ones((24,32),np.uint8)
            observation = {'frame':image,'mask':mask,'lane_mask':np.zeros_like(mask),
                           'sequence':4,'captured_at':.2,'detections':[]}
            for i in range(3):
                archive.write_frames(image,image,None,{'sample':i,'semantic_sequence':4},observation)
            archive.close()
            state = archive.get_state()
            self.assertIsNone(state['error']); self.assertEqual(state['written_frames'],3)
            files = list((archive.run_dir/'semantic_inputs').glob('*.npz'))
            self.assertEqual(len(files),1)
            with np.load(files[0],allow_pickle=False) as data:
                np.testing.assert_array_equal(data['frame'],image)
                np.testing.assert_array_equal(data['road'],mask)
                self.assertEqual(int(data['sequence']),4)
                self.assertEqual(json.loads(str(data['detections_json'])),[])

    def test_filter_cannot_turn_a_missing_current_outer_edge_into_visible_edge(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        model = primary_tests.MaskModel(); model.road[:] = 0; model.road[:,10:250] = 1
        detector = self.detector(model)
        detector.stabilizer = SemanticStabilizer()
        detector.outer_selector = SemanticOuterRoute([[0.,.8],[1.,.8]])
        masks = detector.predict_masks(np.zeros((480,640,3),np.uint8))
        self.assertTrue(detector.semantic_corridor(*masks).valid)
        self.now = .06
        model.road[:,0:250] = 1
        masks = detector.predict_masks(np.zeros((480,640,3),np.uint8))
        track = detector.semantic_corridor(*masks)
        self.assertFalse(track.valid)
        self.assertIn('left edge missing',track.reason)


if __name__ == '__main__': unittest.main()
