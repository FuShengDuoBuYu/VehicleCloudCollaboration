"""Current-frame track support and turn intent, without hardware."""
import sys,unittest
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.perception import track_colors

class TrackSurfaceReferenceTests(unittest.TestCase):
    def scene(self):
        frame=np.full((240,320,3),(85,100,110),np.uint8)
        frame[:150]=(140,140,140);frame[150:157]=(0,255,255)
        frame[190:225,150:175]=(255,255,255)
        frame[170:230,270:]= (0,150,0)
        return np.zeros((240,320),np.uint8),frame

    def test_dim_gray_ground_is_not_a_yellow_boundary(self):
        road,frame=self.scene();frame[:]=(85,108,110)
        paint=track_colors.classify_track_paint(road,np.zeros_like(road),frame)
        self.assertEqual(paint.yellow_pixels,0)
        self.assertEqual(paint.ego_yellow_ratio,0)

    def test_bright_pale_yellow_boundary_is_observed_without_classifying_dim_gray(self):
        road,frame=self.scene();frame[150:157]=(210,230,240)
        frame[150:157,:20]=(0,255,255)
        # H20/S32/V240: glare reduces saturation but the marking is visible.
        self.assertTrue(track_colors.observe_track_surface(road,frame)['observed'])
        paint=track_colors.classify_track_paint(road,np.zeros_like(road),frame)
        self.assertTrue(paint.yellow_mask[153,160])
        frame[:]=(85,108,110)
        self.assertEqual(track_colors.classify_track_paint(road,np.zeros_like(road),frame).yellow_pixels,0)

    def test_unconnected_bright_warm_floor_is_not_a_pale_yellow_boundary(self):
        road,frame=self.scene();frame[190:225,150:175]=(210,230,240)
        paint=track_colors.classify_track_paint(road,np.zeros_like(road),frame)
        self.assertFalse(paint.yellow_mask[211,160])

    def test_isolated_yellow_sensor_speck_is_not_a_boundary_but_thin_line_is(self):
        road=np.ones((240,320),np.uint8);frame=np.full((480,640,3),100,np.uint8)
        frame[460,305]=(0,255,255)
        paint=track_colors.classify_track_paint(road,np.zeros_like(road),frame)
        self.assertEqual(paint.yellow_pixels,0)
        frame[440,250:390]=(0,255,255)
        paint=track_colors.classify_track_paint(road,np.zeros_like(road),frame)
        self.assertGreater(paint.yellow_pixels,0)
        self.assertTrue(paint.yellow_mask[220,160])

    def test_surface_support_needs_current_yellow_bound_and_does_not_cross_it(self):
        road,frame=self.scene()
        surface=track_colors.observe_track_surface(road,frame)
        self.assertTrue(surface['observed'])
        self.assertTrue(surface['mask'][211,160])
        self.assertTrue(surface['mask'][220,100])
        self.assertFalse(surface['mask'][145,160])
        self.assertFalse(surface['mask'][153,160])
        self.assertFalse(surface['mask'][211,290])
        frame[150:157]=(85,100,110)
        surface=track_colors.observe_track_surface(road,frame)
        self.assertFalse(surface['observed'])
        self.assertFalse(np.any(surface['mask']))

    def test_surface_cannot_invent_boundary_over_missing_columns(self):
        road,frame=self.scene();frame[150:157,140:180]=(85,100,110)
        surface=track_colors.observe_track_surface(road,frame)
        self.assertFalse(surface['mask'][211,160])

    def test_visible_far_front_boundary_supports_gray_floor_without_starting_turn(self):
        road,frame=self.scene();frame[150:157]=(140,140,140)
        frame[88:94]=(0,255,255)
        surface=track_colors.observe_track_surface(road,frame)
        self.assertTrue(surface['observed'])
        self.assertTrue(surface['mask'][235,160])
        self.assertFalse(surface['mask'][90,160])

    def test_blue_tinted_white_print_is_supported_inside_current_yellow_boundary(self):
        road,frame=self.scene();frame[190:225,150:175]=(230,200,175)
        surface=track_colors.observe_track_surface(road,frame)
        self.assertTrue(surface['mask'][211,160])
        # A bright green patch is still excluded, independently of the model.
        frame[190:225,150:175]=(80,220,80)
        self.assertFalse(track_colors.observe_track_surface(road,frame)['mask'][211,160])

    def test_rotation_clearance_is_independent_of_an_unavailable_forward_path(self):
        from autodrive.perception.visual_clearance import assess_rotation_clearance
        road=np.zeros((240,320),np.uint8);road[190:,20:300]=1
        yellow=np.zeros_like(road);yellow[188:190]=1
        self.assertTrue(assess_rotation_clearance(road,yellow)['rotation_clear'])
        yellow[236:240,150:170]=1
        self.assertFalse(assess_rotation_clearance(road,yellow)['rotation_clear'])

    def test_front_yellow_does_not_block_rotation_on_current_nearest_floor(self):
        from autodrive.perception.visual_clearance import assess_rotation_clearance
        road=np.zeros((240,320),np.uint8);road[190:,20:300]=1
        yellow=np.zeros_like(road);yellow[217:226,130:146]=1
        road[yellow>0]=0
        result=assess_rotation_clearance(road,yellow)
        self.assertTrue(result['rotation_clear'])
        # The same line at the actual image origin must revoke rotation.
        yellow[236:240,156:164]=1
        self.assertFalse(assess_rotation_clearance(road,yellow)['rotation_clear'])

    def test_front_side_yellow_is_separate_from_nearest_origin_hazard(self):
        road=np.ones((240,320),np.uint8);frame=np.full((240,320,3),100,np.uint8)
        frame[219:227,136:150]=(0,255,255)
        paint=track_colors.classify_track_paint(road,np.zeros_like(road),frame)
        self.assertEqual(paint.rotation_yellow_ratio,0.)
        self.assertGreater(paint.yellow_pixels,0)
        frame[236:240,156:164]=(0,255,255)
        paint=track_colors.classify_track_paint(road,np.zeros_like(road),frame)
        self.assertGreater(paint.rotation_yellow_ratio,.02)

    def test_rotation_direction_cannot_come_from_detached_road(self):
        from autodrive.perception.visual_clearance import assess_rotation_clearance
        road=np.zeros((240,320),np.uint8)
        road[200:,115:205]=1;road[120:190,225:319]=1
        result=assess_rotation_clearance(road,np.zeros_like(road))
        self.assertTrue(result['rotation_clear'])
        self.assertTrue(result['heading_error'] is None or abs(result['heading_error'])<=.02)

    def test_surface_option_requires_enabled_stationary_guard(self):
        from autodrive.control.field_trial import validate_stationary_corner_config
        for stationary in ({},{'enabled':False},{'enabled':False,'visual_feedback':True,
                'visual_corner_guard':True}):
            with self.subTest(stationary=stationary),self.assertRaises(ValueError):
                validate_stationary_corner_config({'stationary_corner':stationary,
                    'perception':{'track_colors':{'enabled':True,'surface_assist':True}}})

    def test_side_support_does_not_fill_outside_lines_or_follow_two_blobs(self):
        road=np.zeros((240,320),np.uint8);road[192:231,135:185]=1
        frame=np.full((240,320,3),100,np.uint8)
        frame[130:140,45:55]=(0,255,255);frame[130:140,265:275]=(0,255,255)
        self.assertFalse(track_colors.observe_track_surface(road,frame)['observed'])
        frame[:]=100
        frame[130:220,60:65]=(0,255,255);frame[130:220,255:260]=(0,255,255)
        surface=track_colors.observe_track_surface(road,frame)
        self.assertTrue(surface['observed'])
        self.assertTrue(surface['mask'][170,160])
        self.assertFalse(surface['mask'][170,10])
        self.assertFalse(surface['mask'][225,100])

    def test_bounded_side_lines_can_support_current_model_road_texture(self):
        road,frame=self.scene();frame[150:157]=(85,100,110)
        road[130:]=1
        cv2.line(frame,(60,130),(0,220),(0,255,255),5)
        cv2.line(frame,(260,130),(319,220),(0,255,255),5)
        surface=track_colors.observe_track_surface(road,frame)
        self.assertTrue(surface['observed'])
        self.assertTrue(surface['mask'][211,160])
        self.assertFalse(surface['mask'][100,160])

    def test_blue_tinted_gray_floor_does_not_create_false_holes(self):
        road,frame=self.scene();frame[190:230,120:200]=(90,55,55)
        surface=track_colors.observe_track_surface(road,frame)
        self.assertTrue(surface['mask'][223,160])
        self.assertFalse(surface['mask'][211,290])

    def test_sloping_side_lines_are_not_a_transverse_front(self):
        road=np.zeros((240,320),np.uint8);road[130:,80:240]=1
        frame=np.full((240,320,3),100,np.uint8)
        cv2.line(frame,(120,110),(0,220),(0,255,255),5)
        cv2.line(frame,(200,110),(319,220),(0,255,255),5)
        surface=track_colors.observe_track_surface(road,frame)
        self.assertTrue(surface['observed'])
        self.assertTrue(surface['mask'][211,160])
        self.assertFalse(surface['mask'][225,20])

    def test_transverse_missing_columns_cannot_be_reclassified_as_side_lines(self):
        road=np.zeros((240,320),np.uint8);road[185:]=1
        frame=np.full((240,320,3),100,np.uint8)
        frame[145:185]=(0,255,255);frame[145:185,135:185]=100
        surface=track_colors.observe_track_surface(road,frame)
        self.assertFalse(surface['mask'][160,160])

    def test_current_rotation_support_can_turn_when_forward_fit_fails(self):
        from types import SimpleNamespace
        from car.test.test_visual_corner_guard import VisualCornerGuardTests
        from autodrive.control.lane_centering import DifferentialDriveCommand,LaneEstimate
        runtime=VisualCornerGuardTests().runtime()
        road=np.zeros((240,320),np.uint8);road[190:,20:300]=1
        b=SimpleNamespace(valid=True,source='yolopv2',confidence=.9,semantic_hard_safe=True,
            semantic_sequence=1,semantic_captured_at=.99,corridor_mask=road,
            semantic_yellow_mask=np.zeros_like(road),semantic_front_observed=True,
            semantic_observed_front_ratio=.7,semantic_right_exit_observed=True,
            semantic_turn_entry_ready=True,semantic_exit_heading_error=.5,
            semantic_rotation_heading_error=.5,semantic_rotation_mask=road,
            semantic_track_surface_pixels=int(road.sum()),semantic_forward_road_valid=False)
        def tick(now,sequence):
            b.semantic_sequence=sequence;b.semantic_captured_at=now-.01
            return runtime.filter(DifferentialDriveCommand('stop',0.,0.,0.,0.),LaneEstimate(False,0.),b,
                now=now,frame_age_seconds=.01,imu_sample={'value':{'radians':[0.,0.,0.]},
                    'valid':True,'stale':False,'age_s':0.,'received_monotonic':now})
        self.assertEqual(tick(1.,1).action,'stop')
        self.assertEqual(tick(1.1,2).action,'pivot-right')
        b.semantic_hard_safe=False
        self.assertEqual(tick(1.15,3).action,'stop')

    def test_matched_bounded_surface_recovers_model_dropout_but_not_obstacles_or_stale(self):
        from types import SimpleNamespace
        from dataclasses import replace
        from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
        from autodrive.perception.yolopv2_fusion import YOLOPv2FusionConfig
        road,frame=self.scene();model=SimpleNamespace(predict_masks=lambda f:(road.copy(),road.copy()),last_detections=[])
        d=YOLOPv2SemanticDetector(YOLOPv2FusionConfig(asynchronous=False,required_for_motion=True,
            detect_objects=True,drivable_only=False,max_result_age_seconds=.6),model=model,
            clock=lambda:0.,output_width=320,output_height=240,
            track_color_config={'enabled':True,'surface_assist':True})
        self.addCleanup(d.close);d.predict_masks(frame,captured_at=0.)
        b=d.semantic_corridor(road,road)
        self.assertTrue(b.valid,b.reason);self.assertTrue(b.semantic_hard_safe)
        self.assertGreater(b.semantic_track_surface_pixels,0)
        d._consumer_result=replace(d._consumer_result,detections=({'box':[.4,.6,.6,.9],'confidence':.9},))
        self.assertFalse(d.semantic_corridor(road,road).semantic_hard_safe)
        d._clock=lambda:.7
        self.assertFalse(d.semantic_corridor(road,road).valid)
        d._consumer_result=replace(d._consumer_result,source_frame=None);d._clock=lambda:0.
        self.assertFalse(d.semantic_corridor(road,road).semantic_hard_safe)

    def test_near_confirmed_exit_turns_even_when_approach_path_near_heading_is_straight(self):
        from car.test.test_visual_corner_guard import VisualCornerGuardTests
        helper=VisualCornerGuardTests();runtime=helper.runtime()
        helper.tick(runtime,1.,1,front=.76,heading=.02)
        out=helper.tick(runtime,1.1,2,front=.76,heading=.02)
        self.assertEqual(out.action,'pivot-right')

    def test_surface_geometry_uses_current_lookahead_for_turn_entry(self):
        from autodrive.perception.semantic_outer_route import route_options,SemanticOuterRoute
        config={'centerline':{'lookahead_ratio':.62,'bottom_ratio':.94},
            'perception':{'semantic_outer_loop':{'enabled':True,'width_profile':[[0,.4],[1,1.]],
                'visual_feedback':True,'stationary_corners':True},
                'track_colors':{'enabled':True,'surface_assist':True}},
            'stationary_corner':{'visual_corner_guard':True,'front_near_ratio':.74}}
        options=route_options(config)
        self.assertEqual(options['corner_approach_ratio'],.62)
        selector=SemanticOuterRoute(**options);road=np.zeros((240,320),np.uint8);road[158:]=1
        selector.select(road,surface_observed=True)
        self.assertTrue(selector.turn_entry_ready)
        self.assertGreater(selector.exit_heading_error,.18)
        road[:]=0;road[134:]=1
        selector.select(road,surface_observed=True)
        self.assertFalse(selector.turn_entry_ready)
        self.assertEqual(selector.preview_point[0],160)

if __name__=='__main__':unittest.main()
