import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.perception.semantic_outer_route import SemanticOuterRoute
from autodrive.control.lane_centering import RoadCenterlineEstimator

class StationaryCornerPerceptionTests(unittest.TestCase):
    def test_observed_corner_keeps_forward_preview_until_stationary_controller_turns(self):
        for first_row in (151,178,188):
            selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True)
            road=np.zeros((240,320),np.uint8);road[first_row:]=1
            corridor,valid,_=selector.select(road)
            self.assertTrue(valid)
            self.assertAlmostEqual(selector.front_boundary_ratio,first_row/240.)
            self.assertTrue(selector.right_exit_observed)
            self.assertEqual(selector.preview_point[0],160)
            e=RoadCenterlineEstimator(top_ratio=.54,lookahead_ratio=.62,
                tight_turn_lookahead_ratio=.78).estimate(corridor,preserve_exclusions=True,
                                                    semantic_preview_point=selector.preview_point)
            self.assertTrue(e.valid,e.reason)
            self.assertLess(abs(e.heading_error),.02)

    def test_front_metadata_never_survives_missing_current_exit_or_a_branch(self):
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True)
        road=np.zeros((240,320),np.uint8);road[178:]=1
        selector.select(road)
        road[178:,:]=0;road[178:,:210]=1
        selector.select(road)
        self.assertIsNone(selector.front_boundary_ratio)
        self.assertFalse(selector.right_exit_observed)

    def test_stationary_mode_must_be_boolean(self):
        with self.assertRaises(ValueError):
            SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners='false')

    def test_exit_dropout_preserves_current_front_position_but_cannot_propose_pivot(self):
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True)
        road=np.zeros((240,320),np.uint8);road[178:]=1
        road[180:220,288:]=0
        selector.select(road)
        self.assertTrue(selector.front_observed)
        self.assertAlmostEqual(selector.front_boundary_ratio,178/240.)
        self.assertFalse(selector.right_exit_observed)
        self.assertIsNotNone(selector.preview_point)
        self.assertEqual(selector.preview_point[0],160)
        self.assertLess(selector.right_exit_support_ratio,.85)
        self.assertIn('right-exit',selector.corner_reject_reason)
        estimate=RoadCenterlineEstimator(top_ratio=.54,lookahead_ratio=.62).estimate(
            road,preserve_exclusions=True,semantic_preview_point=selector.preview_point)
        self.assertTrue(estimate.valid,estimate.reason)

    def test_sloped_front_is_diagnostic_only_and_resets_with_new_frame(self):
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True)
        road=np.zeros((240,320),np.uint8)
        for x in range(320):road[128+x//4:,x]=1
        selector.select(road)
        self.assertTrue(selector.front_observed)
        self.assertIsNotNone(selector.observed_front_ratio)
        self.assertIsNone(selector.front_boundary_ratio)
        self.assertFalse(selector.right_exit_observed)
        self.assertIn('spread',selector.corner_reject_reason)
        selector.select(np.zeros_like(road))
        self.assertFalse(selector.front_observed)
        self.assertIsNone(selector.observed_front_ratio)

if __name__=='__main__':unittest.main()
