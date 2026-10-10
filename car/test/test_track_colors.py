"""Track paint policy uses stored/current images only; no devices or model load."""
import importlib,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.perception.yolopv2_fusion import YOLOPv2FusionConfig
from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
from autodrive.control.lane_centering import RoadCenterlineEstimator,LaneCenteringController
from autodrive.runtime.onboard import analyze


class TrackColorsTests(unittest.TestCase):
    def scene(self,color=(220,220,220)):
        road=np.zeros((240,320),np.uint8);road[90:,40:280]=1
        lane=np.zeros_like(road);lane[176:183,40:280]=1
        frame=np.full((240,320,3),64,np.uint8);frame[176:183,40:280]=color
        return road,lane,frame

    def classify(self,road,lane,frame):
        from autodrive.perception import track_colors
        return track_colors.classify_track_paint(road,lane,frame)

    def test_white_lane_component_is_drivable_only_inside_yolo_road(self):
        road,lane,frame=self.scene();road[179,160]=0
        result=self.classify(road,lane,frame)
        self.assertFalse(result.exclusion_mask[180,160])
        corridor=(road>0)&(result.exclusion_mask==0)
        self.assertFalse(corridor[179,160])
        self.assertTrue(np.all(corridor<=road))
        self.assertGreater(result.allowed_white_pixels,0)

    def test_yellow_is_excluded_even_if_model_lane_head_misses_it(self):
        road,lane,frame=self.scene((0,255,255));lane[:]=0
        result=self.classify(road,lane,frame)
        self.assertTrue(result.exclusion_mask[179,160])
        self.assertGreater(result.yellow_pixels,0)

    def test_yellow_overrides_white_connected_marking_and_unknown_lines_stay(self):
        road,lane,frame=self.scene();frame[176:183,110:120]=(0,255,255)
        result=self.classify(road,lane,frame)
        self.assertTrue(result.exclusion_mask[179,160])
        road,lane,frame=self.scene((64,64,64))
        self.assertTrue(self.classify(road,lane,frame).exclusion_mask[179,160])

    def detector(self,road,lane):
        model=SimpleNamespace(predict_masks=lambda frame:(road.copy(),lane.copy()),last_detections=[])
        d=YOLOPv2SemanticDetector(YOLOPv2FusionConfig(asynchronous=False,required_for_motion=True,
            detect_objects=True,drivable_only=False,max_result_age_seconds=.3),model=model,
            clock=lambda:0.,output_width=320,output_height=240,track_color_config={'enabled':True})
        self.addCleanup(d.close);return d

    def test_white_policy_survives_estimator_second_lane_exclusion(self):
        road,lane,frame=self.scene();d=self.detector(road,lane)
        est,cmd,*_=analyze(d,RoadCenterlineEstimator(),LaneCenteringController(),None,
            frame,'center',.1,1.,captured_at=0.)
        self.assertTrue(est.valid,est.reason)
        self.assertNotEqual(cmd.action,'stop')

    def test_surface_assist_keeps_current_model_gray_floor_through_lane_texture(self):
        road,lane,frame=self.scene((64,64,64));road[179,160]=0
        d=self.detector(road,lane);d.surface_assist=True
        r,l=d.predict_masks(frame,captured_at=0.);b=d.semantic_corridor(r,l)
        self.assertTrue(b.valid,b.reason)
        self.assertTrue(b.corridor_mask[180,160])
        # Without an observed colour boundary, a model hole remains a hole.
        self.assertFalse(b.corridor_mask[179,160])

    def test_color_uses_consumed_frame_and_missing_frame_cannot_authorize(self):
        road,lane,white=self.scene();d=self.detector(road,lane)
        r,l=d.predict_masks(white,captured_at=0.)
        yellow=white.copy();yellow[176:183,40:280]=(0,255,255)
        d._publish_result(d._infer(1,yellow,0.,'fp32'))
        b=d.semantic_corridor(r,l)
        self.assertFalse(b.semantic_exclusion_mask[179,160])
        d._consumer_result=replace(d._consumer_result,source_frame=None)
        b=d.semantic_corridor(r,l)
        self.assertFalse(b.valid)
        self.assertFalse(b.semantic_hard_safe)

    def test_near_yellow_and_detected_obstacle_still_revoke_white_permission(self):
        road,lane,frame=self.scene();frame[211:220,140:180]=(0,255,255)
        d=self.detector(road,lane);r,l=d.predict_masks(frame,captured_at=0.);b=d.semantic_corridor(r,l)
        self.assertFalse(b.semantic_hard_safe)
        self.assertTrue(b.yellow_hazard)
        frame[211:220,140:180]=64
        d.predict_masks(frame,captured_at=0.)
        d._consumer_result=replace(d._consumer_result,detections=({'box':[.4,.6,.6,.9],'confidence':.9},))
        self.assertFalse(d.semantic_corridor(road,lane).valid)

    def test_stale_or_mismatched_masks_cannot_authorize_color_permission(self):
        road,lane,frame=self.scene();d=self.detector(road,lane)
        d.predict_masks(frame,captured_at=0.)
        changed=road.copy();changed[100,100]=0
        b=d.semantic_corridor(changed,lane)
        self.assertFalse(b.valid);self.assertFalse(b.semantic_hard_safe)
        d._clock=lambda:.31
        b=d.semantic_corridor(road,lane)
        self.assertFalse(b.valid);self.assertFalse(b.semantic_hard_safe)
        self.assertIn('stale',b.reason)

    def test_lossless_replay_uses_recorded_color_config_without_model_or_chassis(self):
        import csv,json,tempfile,yaml
        from unittest.mock import patch
        from autodrive.runtime import onboard
        from autodrive.tools.replay_outer_loop import replay
        road,lane,frame=self.scene()
        _,config=onboard.load_config(onboard.DEFAULT_CONFIG,'rosmaster_jetson_yolopv2_visual_feedback')
        config['stationary_corner']['enabled']=False
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';(run/'semantic_inputs').mkdir(parents=True)
            np.savez_compressed(run/'semantic_inputs/00000007.npz',sequence=7,captured_at=2.,
                road=road,lane=lane,frame=frame,detections_json='[]')
            (run/'resolved_runtime_config.yaml').write_text(yaml.safe_dump(config))
            with (run/'onboard_log.csv').open('w') as f:
                out=csv.DictWriter(f,fieldnames=['timestamp_s','sample','action','semantic_sequence',
                    'semantic_result_age_exact_s']);out.writeheader()
                out.writerow(dict(timestamp_s=0,sample=0,action='stop',semantic_sequence=7,
                    semantic_result_age_exact_s=.05))
            (run/'status.json').write_text(json.dumps(dict(termination=dict(stopped=True,completed_at=100.),
                last_result=dict(sample=0),diagnostics=dict(archive=dict(written_frames=1,dropped_frames=0,error=None)))))
            with patch.object(onboard,'create_chassis',side_effect=AssertionError('hardware prohibited')):
                replay(run,root/'derived',profile='recorded',mode='recorded')
            command=json.loads((root/'derived/commands.json').read_text())[0]
            self.assertTrue(command['valid'])
            self.assertTrue(command['track_colors']['enabled'])
            self.assertGreater(command['track_colors']['allowed_white_pixels'],0)


if __name__=='__main__':unittest.main()
