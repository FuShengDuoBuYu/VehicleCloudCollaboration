"""Exercise real local controller -> worker -> recovery -> final dry-run wheel mapping."""
import json
import contextlib
from dataclasses import replace
import io
import sys
import cv2
from pathlib import Path
import tempfile
import time
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from car.test.test_cloud_recovery import evidence, STOP
from autodrive.control.cloud_recovery import RecoveryConfig, build_evidence
from autodrive.control.cloud_recovery import RecoveryEngine
from autodrive.control.recovery_worker import RecoveryCoordinator
from autodrive.control.drive_runtime import SafeWheelDriver, WheelMappingConfig, PerceptionMotionGate
from autodrive.control.stationary_corner import StationaryCornerConfig
from autodrive.control.lane_centering import LaneEstimate
from autodrive.runtime.onboard import StationaryCornerRuntime, apply_runtime_command, cloud_configuration, load_config, DEFAULT_CONFIG
from cloud_client.client import CloudSceneResult


class CandidateClient:
    last_request_metadata={}
    def __init__(self):self.calls=0
    def request_scene(self,paths,context):
        self.calls+=1
        candidate=context['candidates'][0]
        scene=dict(schema_version='road-recovery-v1',assessment=candidate['assessment'],
            recommendation='try_candidate',candidate_id=candidate['candidate_id'],uncertain=False,hazards=[],reason='synthetic')
        return CloudSceneResult(scene=scene,response_model='mock',usage={},response_id='r',
            raw_response={'synthetic':True},context=context,schema_version='road-recovery-v1')


class RecoveryRuntimeTests(unittest.TestCase):
    def test_cancel_busy_worker_never_applies_late_selection(self):
        started=threading.Event();release=threading.Event()
        class BlockingClient(CandidateClient):
            def request_scene(self,paths,context):
                started.set();release.wait(3.)
                return super().request_scene(paths,context)
        with tempfile.TemporaryDirectory() as tmp:
            client=BlockingClient();coord=RecoveryCoordinator(RecoveryConfig(enabled=True),client,Path(tmp)/'cloud')
            now=time.monotonic()
            try:
                for seq,offset in enumerate((0.,.5,1.1),1):
                    coord.filter(STOP,evidence(now+offset,seq),now+offset)
                self.assertTrue(started.wait(1.))
                coord.close();release.set();coord._thread.join(2.)
                self.assertEqual(coord.filter(STOP,evidence(now+1.3,4),now+1.3).action,'stop')
                self.assertTrue(coord._results.empty())
                self.assertEqual(client.calls,1)
                self.assertIsNone(coord.motion_deadline)
            finally:
                release.set();coord.close();coord._thread.join(2.)

    def test_onboard_main_archives_recovery_and_never_creates_hardware(self):
        from autodrive.runtime import onboard
        from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
        from car.test.test_yolopv2_primary import MaskModel
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'gray.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),20.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in range(40):writer.write(np.full((480,640,3),100,np.uint8))
            writer.release()
            config_path,config=load_config(DEFAULT_CONFIG,'rosmaster_jetson_yolopv2_visual_feedback')
            config['perception']['yolopv2']['asynchronous']=False
            config['cloud_recovery']['stall_seconds']=.1
            client=CandidateClient();client._require_key=lambda:'synthetic-key'
            actual_analyze=onboard.analyze
            def analyze(*args,**kwargs):
                time.sleep(.04)
                est,cmd,frame,inf,btime,boundary=actual_analyze(*args,**kwargs)
                boundary.valid=False
                return replace(est,valid=False),STOP,frame,inf,btime,boundary
            def detector(config,**kwargs):return YOLOPv2SemanticDetector(config,model=MaskModel(),**kwargs)
            arguments=['onboard','--video',str(video),'--max-samples','25','--output-dir',str(root/'output')]
            with patch.object(sys,'argv',arguments),patch.object(onboard,'load_config',return_value=(config_path,config)), \
                 patch.object(onboard,'analyze',side_effect=analyze), \
                 patch('autodrive.perception.yolopv2_semantic.YOLOPv2SemanticDetector',side_effect=detector), \
                 patch('cloud_client.client.CloudClient',return_value=client) as cloud, \
                 patch.object(onboard,'create_chassis') as hardware,contextlib.redirect_stdout(io.StringIO()):
                onboard.main()
            hardware.assert_not_called()
            self.assertEqual(cloud.call_args.kwargs['contract'],'road-recovery-v1')
            self.assertGreater(client.calls,0)
            logs=list((root/'output').rglob('cloud/applications.jsonl'));self.assertEqual(len(logs),1)
            records=[json.loads(line) for line in logs[0].read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(r['wheels']['front_left_pwm']>0 for r in records))
            self.assertTrue(any(r['action']=='stop' for r in records))
            status=json.loads((root/'output/status.json').read_text())
            self.assertTrue(status['termination']['stopped']);self.assertTrue(status['cloud_arbitration']['closed'])
            self.assertEqual(status['cloud_arbitration']['fusion']['policy_id'],'rule-recovery-v1')
            self.assertTrue(any(r['fusion']['source']=='cloud_recovery' for r in records))

    def test_real_worker_current_geometry_final_pwm_and_stop(self):
        self.exercise_runtime(False)

    def test_real_worker_right_pivot_final_pwm_and_yaw_stop(self):
        self.exercise_runtime(True)

    def exercise_runtime(self,pivot):
        with tempfile.TemporaryDirectory() as tmp:
            client=CandidateClient();coord=RecoveryCoordinator(RecoveryConfig(enabled=True),client,Path(tmp)/'cloud')
            self.addCleanup(coord.close)
            local=StationaryCornerRuntime(StationaryCornerConfig(enabled=True,visual_feedback=True,
                visual_corner_guard=True,semantic_max_age_seconds=.6,step_settle_seconds=.25),
                PerceptionMotionGate(resume_valid_frames=2),pivot_verified=True)
            driver=SafeWheelDriver(None,False,WheelMappingConfig(pwm_limit=30,stationary_pivot_pwm=30))
            origin=time.monotonic();moves=[]
            def tick(offset,seq,external=True):
                now=origin+offset;obs=evidence(now,seq)['observation']
                if pivot:
                    obs['mask'][:]=0
                    obs['mask'][130:210,180:310]=1;obs['mask'][205:,100:300]=1
                obs['lane_mask']=np.zeros_like(obs['mask']);obs['detections']=[]
                boundary=SimpleNamespace(valid=False,confidence=.0,semantic_sequence=seq,
                    semantic_captured_at=now-.01,semantic_hard_safe=True,yellow_hazard=False,
                    corridor_mask=np.zeros_like(obs['mask']),semantic_yellow_mask=np.zeros_like(obs['mask']))
                estimate=LaneEstimate(False,0.,centerline=np.array([[160,235],[160,187]]))
                imu=({'value':{'radians':[0.,0.,-.10 if offset>=1.25 else 0.]},
                      'valid':True,'stale':False,'received_monotonic':now} if pivot else {})
                raw=local.filter(STOP,estimate,boundary,now=now,frame_age_seconds=.01,imu_sample=imu,
                                 external_motion_allowed=external)
                e=build_evidence(obs,boundary,raw,now=now,frame_age=.01,external_allowed=external,
                    perception_valid=True,pivot_verified=True,imu_sample=imu,local_state=local.get_state())
                cmd=coord.filter(raw,e,now,local_deadline=local.controller.get_state()['motion_deadline'])
                with patch('autodrive.control.drive_runtime.time.monotonic',return_value=now):
                    cmd,wheels=apply_runtime_command(driver,cmd,local,recovery=coord)
                coord.record_application(cmd,wheels,now);moves.append((cmd.action,wheels['front_left_pwm']))
                return cmd
            for i,t in enumerate((0.,.5,1.1),1):self.assertEqual(tick(t,i).action,'stop')
            deadline=time.monotonic()+3.
            while coord._results.empty() and time.monotonic()<deadline:time.sleep(.01)
            self.assertFalse(coord._results.empty())
            self.assertEqual(tick(1.2,4).action,'pivot-right' if pivot else 'forward')
            self.assertNotEqual(moves[-1][1],0)
            if pivot:
                self.assertLess(driver.get_state()['front_right_pwm'],0)
                self.assertGreater(driver.get_state()['front_left_pwm'],0)
            self.assertEqual(tick(1.26 if pivot else 1.36,5).action,'stop')
            self.assertEqual(moves[-1][1],0)
            self.assertEqual(tick(1.4,6,external=False).action,'stop')
            self.assertEqual(client.calls,1)
            files=list((Path(tmp)/'cloud').glob('*/response.json'))
            self.assertEqual(len(files),1)
            saved=json.loads(files[0].read_text(encoding='utf-8'))
            self.assertTrue(saved['result']['raw_response']['synthetic'])
            self.assertTrue((Path(tmp)/'cloud/applications.jsonl').exists())

    def test_enabled_profile_selects_recovery_and_rejects_excess_pwm(self):
        _,config=load_config(DEFAULT_CONFIG,'rosmaster_jetson_yolopv2_visual_feedback')
        self.assertIsInstance(cloud_configuration(config),RecoveryConfig)
        config['wheels']['pwm_limit']=31
        with self.assertRaises(ValueError):cloud_configuration(config)

    def test_local_short_step_settle_can_close_cloud_event(self):
        from car.test.test_visual_corner_guard import VisualCornerGuardTests
        helper=VisualCornerGuardTests();local=helper.runtime()
        engine=RecoveryEngine(RecoveryConfig(enabled=True),'run')
        engine.attempts=1;engine.phase='HOLD'
        applied=[]
        for seq in range(1,31):
            now=1.+seq*.1
            command=helper.tick(local,now,seq,front=.575,heading=.02)
            e=evidence(now,seq);e['local_route_valid']=local.get_state()['local_safe']
            e['stop_code']='normal_wait' if command.action=='stop' else 'local_driving'
            applied.append(engine.tick(command,e,now).action)
        self.assertIn('forward',applied)
        self.assertEqual(engine.attempts,0)
        self.assertIsNone(engine.request)


if __name__=='__main__': unittest.main()
