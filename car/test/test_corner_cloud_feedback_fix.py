"""Regression of the unassisted c8a60d2b failure; CPU only, no devices/network."""
import unittest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from car.test.test_cloud_recovery import evidence, STOP, FORWARD
from car.test.test_cloud_recovery_contract import recovery
from car.test import test_visual_corner_guard as corner_tests
from autodrive.control.cloud_recovery import RecoveryConfig, build_evidence
from autodrive.control.decision_fusion import (
    DecisionFusionAdapter, RuleBasedRecoveryPolicy, FusionInput, CloudAdvice)
from autodrive.control.lane_centering import LaneEstimate


class CornerFeedbackFixTests(unittest.TestCase):
    def tick(self, runtime, now, seq, front, heading, right=300):
        road=np.zeros((240,320),np.uint8);road[130:,30:right]=1
        boundary=SimpleNamespace(valid=True,confidence=.9,semantic_sequence=seq,
            semantic_captured_at=now-.01,semantic_hard_safe=True,
            corridor_mask=road,semantic_rotation_mask=road,
            semantic_yellow_mask=np.zeros_like(road),semantic_track_surface_pixels=1000,
            semantic_front_observed=True,semantic_observed_front_ratio=front,
            semantic_turn_entry_ready=True,semantic_right_exit_observed=True,
            semantic_exit_heading_error=.6,semantic_rotation_heading_error=.8)
        estimate=LaneEstimate(True,.9,heading_error=.35,near_heading_error=heading,
            lateral_error=.1,centerline=np.array([[160,235],[160,211],[160,187]]))
        return runtime.filter(FORWARD,estimate,boundary,now=now,
            imu_sample=dict(value={'radians':[0.,0.,0.]},valid=True,stale=False,received_monotonic=now),
            frame_age_seconds=.01)

    def test_far_preview_does_not_start_pivot_when_short_approach_is_clear(self):
        local=corner_tests.VisualCornerGuardTests().runtime()
        self.tick(local,1.,1,.625,.25625)
        self.assertEqual(self.tick(local,1.1,2,.625,.25625).action,'forward')
        self.assertFalse(local.controller.get_state()['continuous'])

    def test_last_turn_consumed_margin_switches_to_current_clear_forward(self):
        local=corner_tests.VisualCornerGuardTests().runtime()
        self.tick(local,1.,1,.76,.4)
        self.assertEqual(self.tick(local,1.1,2,.76,.4).action,'pivot-right')
        self.tick(local,1.27,3,.76,.4,right=200)
        self.assertEqual(self.tick(local,1.6,4,.76,.4,right=200).action,'forward')

    def test_recovered_current_margin_does_not_latch_one_bad_frame(self):
        local=corner_tests.VisualCornerGuardTests().runtime()
        self.tick(local,1.,1,.76,.4)
        self.assertEqual(self.tick(local,1.1,2,.76,.4).action,'pivot-right')
        self.assertEqual(self.tick(local,1.27,3,.76,.4,right=180).action,'stop')
        self.tick(local,1.6,4,.76,.4,right=250)
        self.assertEqual(self.tick(local,1.7,5,.76,.4,right=250).action,'pivot-right')

    def test_active_corner_can_choose_forward_despite_rightward_far_exit(self):
        local=corner_tests.VisualCornerGuardTests().runtime()
        self.tick(local,1.,1,.76,.4)
        self.assertEqual(self.tick(local,1.1,2,.76,.4).action,'pivot-right')
        self.tick(local,1.27,3,.76,.4)
        self.assertEqual(self.tick(local,1.6,4,.56,.03125).action,'forward')
        self.assertTrue(local.get_state()['visual_corner_active'])
        self.assertFalse(local.controller.get_state()['continuous'])


class StoppedCloudObservationTests(unittest.TestCase):
    def blocked(self, now, seq):
        e=evidence(now,seq)
        e.update(hard_safe=False,observation_ready=True,candidates=[],
                 stop_code='hard_veto',motion_veto_reason='yellow boundary')
        return e

    def test_hard_stop_requests_observation_and_reply_never_releases_motion(self):
        fusion=DecisionFusionAdapter(RuleBasedRecoveryPolicy(RecoveryConfig(enabled=True),'run'))
        for seq,now in enumerate((1.,1.5,2.1),1):
            self.assertEqual(fusion.decide(FusionInput(STOP,self.blocked(now,seq),now,None)).command.action,'stop')
        request=fusion.pending_request
        self.assertIsNotNone(request)
        self.assertEqual(request['context']['candidates'],[])
        self.assertTrue(request['context']['diagnostic_only'])
        event=request['context']['event_id']
        # Even an unsolicited motion suggestion cannot release a hard stop.
        fusion.receive(CloudAdvice(event,recovery(),2.2))
        decision=fusion.decide(FusionInput(FORWARD,self.blocked(2.3,4),2.3,2.45))
        self.assertEqual(decision.command.action,'stop')
        self.assertIsNone(decision.deadline)
        self.assertEqual(fusion.get_state()['event_id'],event)
        self.assertEqual(fusion.get_state()['response_status'],'received')

    def test_failed_external_gate_and_stale_images_do_not_request(self):
        for kind in ('external','stale'):
            fusion=DecisionFusionAdapter(RuleBasedRecoveryPolicy(RecoveryConfig(enabled=True),'run'))
            for seq,now in enumerate((1.,1.5,2.1),1):
                e=self.blocked(now,seq)
                if kind=='external':e['observation_ready']=False
                else:e['observation']['captured_at']=now-1.
                self.assertEqual(fusion.decide(FusionInput(STOP,e,now,None)).command.action,'stop')
            self.assertIsNone(fusion.pending_request)

    def test_yellow_evidence_can_be_observed_but_has_no_motion_candidate(self):
        obs=evidence(2.,1)['observation'];obs.update(lane_mask=np.zeros_like(obs['mask']),detections=[])
        boundary=SimpleNamespace(semantic_sequence=1,semantic_captured_at=1.99,
            semantic_hard_safe=False,yellow_hazard=True)
        kwargs=dict(now=2.,frame_age=.01,external_allowed=True,perception_valid=True,
                    pivot_verified=True,imu_sample={},local_state={})
        e=build_evidence(obs,boundary,STOP,**kwargs)
        self.assertTrue(e.get('observation_ready'))
        self.assertFalse(e['hard_safe']);self.assertEqual(e['candidates'],[])
        kwargs['external_allowed']=False
        self.assertFalse(build_evidence(obs,boundary,STOP,**kwargs).get('observation_ready'))

    def test_cloud_cannot_offer_a_pivot_rejected_by_current_margin_trend(self):
        obs=evidence(2.,1)['observation'];obs.update(lane_mask=np.zeros_like(obs['mask']),detections=[])
        obs['mask'][:]=0;obs['mask'][130:210,180:310]=1;obs['mask'][205:,100:300]=1
        boundary=SimpleNamespace(semantic_sequence=1,semantic_captured_at=1.99,
            semantic_hard_safe=True,yellow_hazard=False)
        kwargs=dict(now=2.,frame_age=.01,external_allowed=True,perception_valid=True,
            pivot_verified=True,imu_sample=dict(value={'radians':[0.,0.,0.]},valid=True,
                stale=False,received_monotonic=2.),local_state={})
        self.assertIn('pivot-right',[c['action'] for c in build_evidence(obs,boundary,STOP,**kwargs)['candidates']])
        kwargs['local_state']={'visual_turn_margin_safe':False}
        self.assertNotIn('pivot-right',[c['action'] for c in build_evidence(obs,boundary,STOP,**kwargs)['candidates']])

    def test_worker_delivers_observation_and_virtual_wheels_stay_zero(self):
        from autodrive.control.recovery_worker import RecoveryCoordinator
        from autodrive.control.drive_runtime import SafeWheelDriver, WheelMappingConfig
        from autodrive.runtime.onboard import apply_runtime_command
        from cloud_client.client import CloudSceneResult
        class ObservationClient:
            def request_scene(self, paths, context):
                scene=recovery();scene.update(recommendation='hold',candidate_id=None,
                    reason='synthetic yellow boundary observation')
                return CloudSceneResult(scene=scene,response_model='mock',usage={},response_id='test',
                    raw_response={'synthetic':True},context=context,schema_version='road-recovery-v1')
        with tempfile.TemporaryDirectory() as tmp:
            coord=RecoveryCoordinator(RecoveryConfig(enabled=True),ObservationClient(),Path(tmp)/'cloud')
            driver=SafeWheelDriver(None,False,WheelMappingConfig(pwm_limit=30))
            now=time.monotonic()
            try:
                for seq,offset in enumerate((0.,.5,1.1),1):
                    stamp=now+offset
                    cmd=coord.filter(STOP,self.blocked(stamp,seq),stamp)
                    with patch('autodrive.control.drive_runtime.time.monotonic',return_value=stamp):
                        cmd,wheels=apply_runtime_command(driver,cmd,recovery=coord)
                    self.assertEqual(wheels['front_left_pwm'],0)
                    self.assertTrue(coord.record_application(cmd,wheels,stamp))
                deadline=time.monotonic()+2.
                while coord._results.empty() and time.monotonic()<deadline:time.sleep(.01)
                cmd=coord.filter(STOP,self.blocked(now+1.2,4),now+1.2)
                self.assertEqual(cmd.action,'stop')
                self.assertEqual(coord.get_state()['response_status'],'received')
                self.assertTrue(coord.get_state()['diagnostic_only'])
            finally:coord.close()
            self.assertEqual(len(list((Path(tmp)/'cloud').glob('*/response.json'))),1)
            self.assertIsNone(coord.get_state()['journal']['error'])

    def test_diagnostic_attempts_are_bounded_and_safe_local_route_can_resume(self):
        fusion=DecisionFusionAdapter(RuleBasedRecoveryPolicy(RecoveryConfig(enabled=True),'run'))
        now=0.;seq=0
        for _ in range(30):
            now+=.5;seq+=1
            decision=fusion.decide(FusionInput(STOP,self.blocked(now,seq),now,None))
            self.assertEqual(decision.command.action,'stop')
            if fusion.get_state()['phase']=='CLOUD_PENDING':
                fusion.receive(CloudAdvice(fusion.pending_request['context']['event_id'],recovery(),now+.01))
        self.assertEqual(fusion.get_state()['attempts'],3)
        for _ in range(3):
            now+=.1;seq+=1
            e=evidence(now,seq);e['local_route_valid']=True
            decision=fusion.decide(FusionInput(FORWARD,e,now,now+.15))
        self.assertEqual(decision.source,'local')
        self.assertEqual(fusion.get_state()['attempts'],0)


if __name__=='__main__':unittest.main()
