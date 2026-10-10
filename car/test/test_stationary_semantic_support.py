"""Failed model output cannot authorize continued stationary rotation."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
from autodrive.perception.yolopv2_fusion import YOLOPv2FusionConfig
from autodrive.runtime.onboard import analyze, StationaryCornerRuntime
from autodrive.control.drive_runtime import PerceptionMotionGate
from autodrive.control.stationary_corner import StationaryCornerConfig
from autodrive.control.lane_centering import LaneEstimate, DifferentialDriveCommand, RoadCenterlineEstimator, LaneCenteringController
from car.test.test_yolopv2_primary import MaskModel


class StationarySemanticSupportTests(unittest.TestCase):
    def observation(self, road, lane=None):
        model=MaskModel(); model.road=road; model.lane=np.zeros_like(road) if lane is None else lane
        detector=YOLOPv2SemanticDetector(YOLOPv2FusionConfig(asynchronous=False,
            required_for_motion=True,detect_objects=True,drivable_only=False),model=model,
            output_width=320,output_height=240,clock=lambda:.7)
        self.addCleanup(detector.close);detector._sequence=10
        return analyze(detector,RoadCenterlineEstimator(),LaneCenteringController(),None,
            np.zeros((480,640,3),np.uint8),'left',.05,1.,captured_at=.7)

    def test_single_pixel_cannot_keep_an_active_pivot_running(self):
        runtime=StationaryCornerRuntime(StationaryCornerConfig(enabled=True),
                                       PerceptionMotionGate(resume_valid_frames=5),pivot_verified=True)
        proposal=DifferentialDriveCommand('forward',0,.2,.2,.9,'test road')
        for seq,now,front in [(i,i*.05,.7) for i in range(1,6)]+[(6,.30,.78),(7,.35,.78),(8,.59,.78),(9,.61,.78)]:
            boundary=SimpleNamespace(valid=True,source='yolopv2',semantic_sequence=seq,
                semantic_result_age_seconds=0.,semantic_front_boundary_ratio=front,
                semantic_right_exit_observed=True,semantic_hard_safe=True)
            result=runtime.filter(proposal,LaneEstimate(True,.9),boundary,now=now,
                imu_sample={'valid':True,'stale':False,'age_s':0.,'received_monotonic':now,
                            'value':{'radians':[0.,0.,0.]}})
        self.assertEqual(result.action,'pivot-right')
        road=np.zeros((240,320),np.uint8);road[1,1]=1
        estimate,proposal,_,_,_,boundary=self.observation(road)
        result=runtime.filter(proposal,estimate,boundary,now=.7,
            imu_sample={'valid':True,'stale':False,'age_s':0.,'received_monotonic':.7,
                        'value':{'radians':[0.,0.,-.1]}})
        self.assertEqual(result.action,'stop')
        self.assertFalse(boundary.semantic_hard_safe)

    def test_disconnected_fragments_are_not_usable_road_support(self):
        road=np.zeros((240,320),np.uint8);road[::3,::3]=1
        self.assertGreater(road.mean(),.05)
        self.assertFalse(self.observation(road)[5].semantic_hard_safe)

    def test_lane_exclusion_can_remove_all_pivot_support(self):
        road=np.zeros((240,320),np.uint8);road[100:200,90:230]=1
        self.assertFalse(self.observation(road,road.copy())[5].semantic_hard_safe)

    def test_contiguous_off_centre_support_keeps_only_rotation_eligible(self):
        road=np.zeros((240,320),np.uint8);road[100:,0:90]=1
        observation=self.observation(road)
        self.assertTrue(observation[5].semantic_hard_safe)
        self.assertFalse(observation[0].valid)
        self.assertEqual(observation[1].action,'stop')

    def test_boundary_retains_consumed_capture_time_for_late_control_gate(self):
        road=np.zeros((240,320),np.uint8);road[100:,0:90]=1
        boundary=self.observation(road)[5]
        self.assertEqual(boundary.semantic_captured_at,.7)
        self.assertEqual(boundary.semantic_result_age_seconds,0.)


if __name__=='__main__': unittest.main()
