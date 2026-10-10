"""Image-space bend planning must stay inside current semantic evidence."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.control.lane_centering import RoadCenterlineEstimator


def bend():
    road = np.zeros((240,320),np.uint8)
    for y in range(110,240):
        center = int(min(285,160+max(0,190-y)*3))
        road[y,center-12:center+13] = 1
    return road


class SemanticBendPathTests(unittest.TestCase):
    def test_visual_route_seeds_from_planning_row_not_bottom_border_fragment(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        road=np.zeros((240,320),np.uint8);road[134:]=1
        road[203:,170:200]=0  # Gap attaches to bottom; branches reconnect ahead.
        road[233:,:60]=1;road[233:,60:230]=0
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True,visual_feedback=True)
        corridor,valid,reason=selector.select(road)
        self.assertTrue(valid,reason)
        self.assertTrue(corridor[225,160], 'bottom sliver discarded the planning origin')
        self.assertFalse(corridor[225,240], 'remote branch became the ego route')
        self.assertTrue(np.all(corridor<=road))
        estimate=self.estimator().estimate(corridor,preserve_exclusions=True,semantic_preview_point=selector.preview_point)
        self.assertTrue(estimate.valid,estimate.reason)
        self.assertTrue(all(road[y,x] for x,y in estimate.centerline))

    def test_visual_origin_ratio_matches_nondefault_planning_start(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        road=np.zeros((240,320),np.uint8);road[134:]=1
        road[218:,100:230]=0
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True,
            visual_feedback=True,path_start_ratio=.90)
        corridor,valid,_=selector.select(road)
        self.assertTrue(valid)
        self.assertTrue(corridor[216,160])
        self.assertTrue(np.all(corridor<=road))

    def test_origin_selection_cannot_create_a_missing_forward_start(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        road=np.zeros((240,320),np.uint8);road[134:]=1
        road[203:,100:230]=0
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True,visual_feedback=True)
        corridor,_,_=selector.select(road)
        self.assertTrue(np.all(corridor<=road))
        estimate=self.estimator().estimate(corridor,preserve_exclusions=True,semantic_preview_point=selector.preview_point)
        self.assertFalse(estimate.valid)

    def test_visual_corner_follows_current_curve_when_straight_ray_hits_inner_boundary(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        road=np.zeros((240,320),np.uint8);road[134:]=1
        road[175:210,200:]=0  # Inner island attached to right border.
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True,visual_feedback=True)
        corridor,valid,reason=selector.select(road)
        self.assertTrue(valid,reason)
        self.assertIsNotNone(selector.preview_point)
        estimate=self.estimator().estimate(corridor,preserve_exclusions=True,
            semantic_preview_point=selector.preview_point)
        self.assertTrue(estimate.valid,estimate.reason)
        self.assertTrue(all(road[y,x] for x,y in estimate.centerline))
        self.assertFalse(corridor[180,220])

    def test_visual_corner_keeps_clear_exit_when_inner_curve_splits_remote_rows(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        road=np.zeros((240,320),np.uint8);road[134:]=1
        road[168:202,270:276]=0
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True,visual_feedback=True)
        corridor,valid,reason=selector.select(road)
        self.assertTrue(valid,reason)
        self.assertTrue(selector.right_exit_observed)
        self.assertGreater(selector.preview_point[0],240)
        self.assertTrue(self.estimator().estimate(corridor,preserve_exclusions=True,
            semantic_preview_point=selector.preview_point).valid)
        self.assertFalse(corridor[180,272])

    def test_visual_corner_cannot_cross_current_marking_to_reach_right_exit(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        road=np.zeros((240,320),np.uint8);road[134:]=1
        road[134:210,200:204]=0
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True,visual_feedback=True)
        selector.select(road)
        self.assertFalse(selector.right_exit_observed)
        self.assertIsNone(selector.preview_point)
        corridor,_,_=selector.select(road)
        self.assertFalse(corridor[175,250])
        self.assertFalse(corridor[175,201])

    def test_visual_route_uses_current_exit_as_target_without_entry_distance(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        import inspect
        self.assertIn('visual_feedback', inspect.signature(SemanticOuterRoute).parameters)
        selector=SemanticOuterRoute([[0,.01],[1,.01]], stationary_corners=True, visual_feedback=True)
        road=np.zeros((240,320),np.uint8);road[142:,:]=1
        corridor,valid,reason=selector.select(road)
        self.assertTrue(valid,reason)
        self.assertGreater(selector.preview_point[0],240)
        self.assertTrue(np.all(corridor<=road))
        result=self.estimator().estimate(corridor,preserve_exclusions=True,
                                        semantic_preview_point=selector.preview_point)
        self.assertTrue(result.valid,result.reason)
        self.assertGreater(result.heading_error,.3)
        original_target=selector.preview_point
        road[original_target[1],192:220]=0
        selector.select(road)
        self.assertNotEqual(selector.preview_point,original_target)
        x,y=selector.preview_point
        self.assertTrue(road[y,x])

    def test_stationary_route_keeps_main_road_when_roadside_sliver_connects_far_away(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        road=np.zeros((240,320),np.uint8)
        road[:,85:290]=1
        road[90:212,10:50]=1
        road[90:94,10:290]=1  # Same component, but no current ego path into the sliver.
        selector=SemanticOuterRoute([[0,.8],[1,1.]],stationary_corners=True)
        corridor,valid,reason=selector.select(road)
        self.assertTrue(valid,reason)
        self.assertEqual(corridor[160,200],1)
        self.assertEqual(corridor[160,20],0)
        self.assertTrue(np.all(corridor<=road))
        estimate=self.estimator().estimate(corridor,preserve_exclusions=True)
        self.assertTrue(estimate.valid,estimate.reason)
        self.assertTrue(all(road[y,x] for x,y in estimate.centerline))

    def test_stationary_branch_continuity_keeps_current_internal_exclusions(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        road=np.zeros((240,320),np.uint8);road[:,85:290]=1
        road[160:164,175:180]=0
        selector=SemanticOuterRoute([[0,.8],[1,1.]],stationary_corners=True)
        corridor,valid,_=selector.select(road)
        self.assertTrue(valid)
        self.assertTrue(np.all(corridor[160:164,175:180]==0))
        self.assertTrue(np.all(corridor<=road))

    def test_front_boundary_with_open_right_road_selects_current_right_preview(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector=SemanticOuterRoute([[0,.4],[1,1.]])
        road=np.zeros((240,320),np.uint8);road[151:,:]=1
        corridor,valid,_=selector.select(road)
        self.assertTrue(valid)
        target=selector.preview_point
        self.assertIsNotNone(target)
        self.assertGreater(target[0],240)
        self.assertTrue(corridor[target[1],target[0]])
        result=self.estimator().estimate(corridor,preserve_exclusions=True,
                                         semantic_preview_point=target)
        self.assertTrue(result.valid,result.reason)
        self.assertGreater(result.heading_error,.3)

    def test_front_boundary_cannot_claim_a_right_exit_through_missing_road(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector=SemanticOuterRoute([[0,.4],[1,1.]])
        road=np.zeros((240,320),np.uint8);road[151:,:210]=1
        _,valid,_=selector.select(road)
        self.assertFalse(valid)

    def test_explicit_preview_cannot_cross_an_enclosed_exclusion(self):
        road=np.zeros((240,320),np.uint8);road[151:,:]=1
        road[193:199,212:225]=0
        result=self.estimator().estimate(road,preserve_exclusions=True,
                                         semantic_preview_point=(262,175))
        self.assertFalse(result.valid)

    def test_front_corner_does_not_bypass_an_exterior_connected_separator(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector=SemanticOuterRoute([[0,.4],[1,1.]])
        road=np.zeros((240,320),np.uint8);road[151:,:]=1
        road[151:210,200:204]=0
        corridor,valid,_=selector.select(road)
        self.assertIsNone(selector.preview_point)
        self.assertFalse(valid and corridor[175,262])

    def test_transverse_boundary_near_old_threshold_keeps_corner_target(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector=SemanticOuterRoute([[0,.4],[1,1.]])
        for front in (142,143,144,145):
            road=np.zeros((240,320),np.uint8);road[front:,:]=1
            selector.select(road)
            self.assertIsNotNone(selector.preview_point)

    def test_shallow_exterior_notch_is_not_a_second_route(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector=SemanticOuterRoute([[0,.4],[1,1.]])
        road=np.zeros((240,320),np.uint8);road[151:,:]=1
        road[151:177,19]=0
        road[151:175,:19]=0
        corridor,valid,_=selector.select(road)
        self.assertTrue(valid)
        self.assertIsNotNone(selector.preview_point)
        self.assertEqual(corridor[176,19],0)

    def test_explicit_preview_ray_checks_between_sampled_pixels(self):
        road=np.zeros((240,320),np.uint8);road[151:,:]=1
        road[200,210]=0
        result=self.estimator().estimate(road,preserve_exclusions=True,
                                         semantic_preview_point=(262,175))
        self.assertFalse(result.valid)

    def test_nearest_rows_keep_observed_ego_support_instead_of_remote_left_fragment(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector=SemanticOuterRoute([[0,.8],[1,1.]])
        road=np.zeros((240,320),np.uint8);road[:,40:280]=1
        road[224:,120:126]=0
        corridor,valid,_=selector.select(road)
        self.assertTrue(valid)
        self.assertEqual(corridor[225,160],1)
        self.assertEqual(corridor[225,122],0)
        result=self.estimator().estimate(corridor,preserve_exclusions=True)
        self.assertTrue(result.valid,result.reason)

    def test_near_ego_support_does_not_switch_the_preview_to_an_inner_branch(self):
        from autodrive.perception.semantic_outer_route import SemanticOuterRoute
        selector=SemanticOuterRoute([[0,.8],[1,1.]])
        road=np.zeros((240,320),np.uint8);road[:,40:280]=1
        road[130:,120:126]=0
        corridor,valid,_=selector.select(road)
        self.assertTrue(valid)
        self.assertEqual(corridor[148,160],0)
        result=self.estimator().estimate(corridor,preserve_exclusions=True)
        self.assertFalse(result.valid)

    def test_bend_fallback_stays_near_the_same_road_centre_on_corner_exit(self):
        road=np.zeros((240,320),np.uint8)
        for y in range(110,240):
            left=max(0,int((160-y)*4))
            road[y,left:255]=1
        road[224:,:151]=0
        result=self.estimator().estimate(road,preserve_exclusions=True)
        self.assertTrue(result.valid,result.reason)
        self.assertLess(result.lateral_error,-.08)
        self.assertGreater(result.heading_error,.02)

    def estimator(self):
        return RoadCenterlineEstimator(top_ratio=.54,lookahead_ratio=.62,
                                       tight_turn_lookahead_ratio=.78,bottom_ratio=.94)

    def test_connected_bend_has_a_contained_rightward_local_path(self):
        road = bend()
        result = self.estimator().estimate(road,preserve_exclusions=True)
        self.assertTrue(result.valid,result.reason)
        self.assertGreater(result.heading_error,.5)
        self.assertGreater(result.lookahead_point[0],250)
        self.assertTrue(all(road[y,x] for x,y in result.centerline))

    def test_missing_cross_section_does_not_get_bridged(self):
        road = bend();road[174:178,:] = 0
        result = self.estimator().estimate(road,preserve_exclusions=True)
        self.assertFalse(result.valid)

    def test_internal_hole_on_existing_path_still_stops(self):
        road=np.zeros((240,320),np.uint8);road[:,40:280]=1
        road[162:164,155:164]=0
        result=self.estimator().estimate(road,preserve_exclusions=True)
        self.assertFalse(result.valid)

    def test_cannot_start_from_a_remote_side_corridor(self):
        road=np.zeros((240,320),np.uint8)
        for y in range(110,240):
            center=int(max(145,270-max(0,190-y)*3))
            road[y,center-12:center+13]=1
        result=self.estimator().estimate(road,preserve_exclusions=True)
        # This case must not gain a new valid result from the bend fallback.
        self.assertFalse(result.valid)


if __name__=='__main__': unittest.main()
