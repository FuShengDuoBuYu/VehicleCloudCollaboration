"""Current-image corner approach and conservative visible margins; no devices."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.control.stationary_corner import StationaryCornerConfig
from autodrive.control.drive_runtime import PerceptionMotionGate
from autodrive.control.lane_centering import DifferentialDriveCommand,LaneEstimate
from autodrive.runtime.onboard import StationaryCornerRuntime
from autodrive.perception.semantic_outer_route import SemanticOuterRoute


class VisualCornerGuardTests(unittest.TestCase):
    def runtime(self):
        return StationaryCornerRuntime(StationaryCornerConfig(enabled=True,visual_feedback=True,
            visual_corner_guard=True,continuous_cruise=True,step_settle_seconds=.25,
            semantic_max_age_seconds=.6),PerceptionMotionGate(resume_valid_frames=2),pivot_verified=True)

    def tick(self,runtime,now,sequence,front=.575,heading=.2,exit=True,safe=True,captured=None):
        road=np.zeros((240,320),np.uint8);road[130:,30:300]=1
        boundary=SimpleNamespace(valid=True,source='yolopv2',semantic_sequence=sequence,
            semantic_captured_at=now-.01 if captured is None else captured,
            semantic_hard_safe=safe,corridor_mask=road,
            semantic_yellow_mask=np.zeros_like(road),semantic_front_observed=front is not None,
            semantic_observed_front_ratio=front,semantic_right_exit_observed=exit)
        estimate=LaneEstimate(True,.9,heading_error=.35,near_heading_error=heading,
            lateral_error=.1,centerline=np.array([[160,225],[170,211],[200,187],[240,175]]))
        return runtime.filter(DifferentialDriveCommand('turn-right',.2,.3,.24,.9),estimate,boundary,
            now=now,imu_sample={'value':{'radians':[0.,0.,0.]},'valid':True,'stale':False,
                'age_s':0.,'received_monotonic':now},frame_age_seconds=.01)

    def test_far_corner_advances_in_short_steps_even_when_exit_heading_is_large(self):
        runtime=self.runtime()
        self.assertEqual(self.tick(runtime,1.,1).action,'stop')
        self.assertEqual(self.tick(runtime,1.1,2).action,'forward')
        self.assertFalse(runtime.controller.get_state()['continuous'])
        self.assertEqual(self.tick(runtime,1.26,3).action,'stop')

    def test_near_corner_needs_a_current_right_exit_before_pivot(self):
        runtime=self.runtime()
        self.tick(runtime,1.,1,front=.76,exit=False)
        self.assertEqual(self.tick(runtime,1.1,2,front=.76,exit=False).action,'stop')
        self.assertEqual(self.tick(runtime,1.2,3,front=.76,exit=True).action,'stop')
        self.assertEqual(self.tick(runtime,1.3,4,front=.76,exit=True).action,'pivot-right')

    def test_turning_cannot_resume_cruise_while_transverse_edge_remains(self):
        runtime=self.runtime()
        self.tick(runtime,1.,1,front=.76)
        self.assertEqual(self.tick(runtime,1.1,2,front=.76).action,'pivot-right')
        self.tick(runtime,1.27,3,front=.76)
        self.assertEqual(self.tick(runtime,1.6,4,heading=.02).action,'forward')
        self.assertTrue(runtime.get_state()['visual_corner_active'])
        self.assertFalse(runtime.controller.get_state()['continuous'])
        self.tick(runtime,1.65,5,front=None,heading=.02,safe=False)
        self.assertTrue(runtime.get_state()['visual_corner_active'])
        self.tick(runtime,2.,6,front=None,heading=.02)
        self.tick(runtime,2.01,6,front=None,heading=.02)
        self.assertTrue(runtime.get_state()['visual_corner_active'])
        self.tick(runtime,2.1,7,front=None,heading=.02)
        self.assertFalse(runtime.get_state()['visual_corner_active'])

    def test_route_far_exit_keeps_a_current_forward_target_until_front_is_near(self):
        selector=SemanticOuterRoute([[0,.4],[1,1.]],stationary_corners=True,
            visual_feedback=True,guarded_corner_approach=True)
        road=np.zeros((240,320),np.uint8);road[138:]=1
        corridor,valid,_=selector.select(road)
        self.assertTrue(valid);self.assertTrue(selector.right_exit_observed)
        self.assertEqual(selector.preview_point[0],160)
        self.assertTrue(corridor[selector.preview_point[1],160])
        road[:]=0;road[178:]=1
        selector.select(road)
        self.assertGreater(selector.preview_point[0],240)
        self.assertTrue(np.all(corridor<=np.where(np.arange(240)[:,None]>=138,1,0)))

    def test_exit_images_captured_before_software_turn_stop_cannot_release_corner(self):
        runtime=self.runtime()
        self.tick(runtime,1.,1,front=.76)
        self.assertEqual(self.tick(runtime,1.1,2,front=.76).action,'pivot-right')
        self.tick(runtime,1.27,3,front=.76)
        self.tick(runtime,1.3,4,front=None,heading=.02,captured=1.2)
        self.tick(runtime,1.35,5,front=None,heading=.02,captured=1.24)
        self.assertTrue(runtime.get_state()['visual_corner_active'])
        self.assertEqual(runtime.get_state()['visual_corner_exit_count'],0)
        self.assertEqual(self.tick(runtime,1.54,6,front=None,heading=.02,captured=1.53).action,'stop')
        self.assertEqual(self.tick(runtime,1.65,7,front=None,heading=.02,captured=1.6).action,'turn-right')
        self.assertFalse(runtime.get_state()['visual_corner_active'])

    def test_path_margin_detects_a_side_boundary_outside_old_ego_roi(self):
        from autodrive.perception.visual_clearance import assess_visible_clearance
        road=np.zeros((240,320),np.uint8);road[100:,30:220]=1
        yellow=np.zeros_like(road);yellow[180:,220:223]=1
        # Path itself is road, but it runs too close to the visible right edge.
        path=np.array([[210,225],[210,211],[210,187]])
        result=assess_visible_clearance(road,yellow,path,margin_ratio=.06)
        self.assertFalse(result['path_safe'])
        self.assertTrue(result['forward_clear'])
        road[210:214,160]=0
        self.assertFalse(assess_visible_clearance(road,yellow,path)['forward_clear'])

    def test_regressed_capture_cannot_count_towards_corner_exit(self):
        runtime=self.runtime()
        self.tick(runtime,1.,1,front=.76)
        self.tick(runtime,1.1,2,front=.76)
        self.tick(runtime,1.27,3,front=.76)
        self.tick(runtime,1.6,4,heading=.02)
        self.tick(runtime,1.77,5,heading=.02)
        self.tick(runtime,1.8,6,front=.76,captured=1.79)
        self.tick(runtime,1.9,7,front=None,heading=.02,captured=1.78)
        self.tick(runtime,2.1,8,front=None,heading=.02,captured=1.785)
        self.assertTrue(runtime.get_state()['visual_corner_active'])
        self.assertEqual(runtime.get_state()['visual_corner_exit_count'],0)
        self.assertEqual(self.tick(runtime,2.2,9,front=None,heading=.02).action,'stop')
        self.assertTrue(runtime.get_state()['visual_corner_active'])
        self.tick(runtime,2.3,10,front=None,heading=.02)
        self.assertFalse(runtime.get_state()['visual_corner_active'])

    def test_application_veto_discards_pending_corner_exit_confirmation(self):
        for confirmations in (1,2):
            with self.subTest(confirmations=confirmations):
                runtime=self.runtime()
                self.tick(runtime,1.,1,front=.76)
                self.tick(runtime,1.1,2,front=.76)
                self.tick(runtime,1.27,3,front=.76)
                self.tick(runtime,1.6,4,front=None,heading=.02)
                if confirmations==2:
                    self.tick(runtime,1.7,5,front=None,heading=.02)
                    self.assertFalse(runtime.get_state()['visual_corner_active'])
                runtime.apply_veto('camera expired before application',now=1.85)
                self.assertTrue(runtime.get_state()['visual_corner_active'])
                self.assertEqual(runtime.get_state()['visual_corner_exit_count'],0)
                self.assertEqual(self.tick(runtime,2.2,6,front=None,heading=.02).action,'stop')
                self.assertTrue(runtime.get_state()['visual_corner_active'])
                self.tick(runtime,2.3,7,front=None,heading=.02)
                self.assertFalse(runtime.get_state()['visual_corner_active'])

    def test_missing_yellow_evidence_and_missing_road_never_grant_clearance(self):
        from autodrive.perception.visual_clearance import assess_visible_clearance
        road=np.zeros((240,320),np.uint8);road[100:,30:300]=1
        path=np.array([[160,225],[160,211],[160,187]])
        self.assertFalse(assess_visible_clearance(road,None,path)['path_safe'])
        self.assertTrue(assess_visible_clearance(road,np.zeros_like(road),path)['path_safe'])
        road[211,160]=0
        self.assertFalse(assess_visible_clearance(road,np.zeros_like(road),path)['path_safe'])

    def test_margin_checks_between_sparse_path_samples(self):
        from autodrive.perception.visual_clearance import assess_visible_clearance
        road=np.zeros((240,320),np.uint8);road[100:,30:300]=1
        yellow=np.zeros_like(road);yellow[206,160]=1
        path=np.array([[130,225],[190,187]])
        self.assertFalse(assess_visible_clearance(road,yellow,path)['path_safe'])

if __name__=='__main__':unittest.main()
