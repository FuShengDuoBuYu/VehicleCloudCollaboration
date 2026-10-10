"""No actuators or network: current evidence, finite attempts and final PWM."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.control.cloud_recovery import RecoveryConfig, RecoveryEngine, build_evidence
from autodrive.control.lane_centering import DifferentialDriveCommand
from autodrive.control.drive_runtime import SafeWheelDriver, WheelMappingConfig
from car.test.test_cloud_recovery_contract import recovery

STOP=DifferentialDriveCommand('stop',0.,0.,0.,.8,'local geometry')
FORWARD=DifferentialDriveCommand('forward',0.,.3,.3,.8,'local route')


def evidence(now,sequence,action='forward'):
    road=np.zeros((240,320),np.uint8);road[130:,30:300]=1
    return dict(hard_safe=True,stop_code='perception_stall',
        observation=dict(sequence=sequence,captured_at=now-.01,mask=road,
                         frame=np.full((480,640,3),100,np.uint8)),
        candidates=[dict(candidate_id='forward-white' if action=='forward' else 'pivot-right',
                        action=action,assessment='white_marking' if action=='forward' else 'right_corner')],
        imu=dict(yaw_rad=0.,imu_timestamp=now,imu_valid=True))


class RecoveryTests(unittest.TestCase):
    def engine(self,**kw):
        return RecoveryEngine(RecoveryConfig(enabled=True,**kw),run_id='run-test')
    def pending(self,engine,action='forward'):
        for i,t in enumerate([1.,1.5,2.1],1): self.assertEqual(engine.tick(STOP,evidence(t,i,action),t).action,'stop')
        self.assertEqual(engine.phase,'CLOUD_PENDING')
        return engine.request
    def test_unique_fresh_persistent_stall_and_history(self):
        engine=self.engine();request=self.pending(engine)
        self.assertEqual(request['context']['run_id'],'run-test')
        self.assertEqual(len(request['frames']),2)
        self.assertEqual(request['context']['frames'][-1]['sequence'],3)
        self.assertEqual(engine.attempts,1)
    def test_duplicate_frames_and_normal_dwell_never_trigger(self):
        engine=self.engine()
        for i in range(10):
            e=evidence(1.,1);engine.tick(STOP,e,1.+i*.1)
        self.assertIsNone(engine.request)
        for i in range(20):
            e=evidence(3.+i*.1,10+i);e['stop_code']='normal_wait'
            engine.tick(STOP,e,3.+i*.1)
        self.assertEqual(engine.attempts,0)
    def test_reply_needs_new_frame_then_one_step_and_post_stop_frame(self):
        engine=self.engine();req=self.pending(engine)
        engine.receive(req['context']['event_id'],recovery(),2.2)
        self.assertEqual(engine.tick(STOP,evidence(2.1,3),2.2).action,'stop')
        command=engine.tick(STOP,evidence(2.3,4),2.3)
        self.assertEqual(command.action,'forward')
        driver=SafeWheelDriver(chassis=None,motors_enabled=False,config=WheelMappingConfig(pwm_limit=30))
        with patch('autodrive.control.drive_runtime.time.monotonic',return_value=2.3):
            state=driver.apply(command,deadline=engine.motion_deadline)
        self.assertTrue(any(state[k]!=0 for k in ('front_left_pwm','front_right_pwm')))
        with patch('autodrive.control.drive_runtime.time.monotonic',return_value=2.46):
            self.assertEqual(driver.check_motion_deadline()['front_left_pwm'],0)
        self.assertEqual(engine.tick(STOP,evidence(2.46,5),2.46).action,'stop')
        self.assertEqual(engine.phase,'OBSERVE')
        self.assertEqual(engine.tick(FORWARD,evidence(2.5,6),2.5).action,'stop')
        for i in range(7,12): engine.tick(STOP,evidence(3.+i*.1,i),3.+i*.1)
        self.assertNotEqual(engine.phase,'PROBE_STEP')
    def test_hard_veto_and_unknown_candidate_never_release(self):
        for changes in ({'candidate_id':'invented'}, {'uncertain':True}, {'assessment':'right_corner'}):
            engine=self.engine();req=self.pending(engine);scene=recovery();scene.update(changes)
            engine.receive(req['context']['event_id'],scene,2.2)
            self.assertEqual(engine.tick(STOP,evidence(2.3,4),2.3).action,'stop')
        engine=self.engine();req=self.pending(engine)
        engine.receive(req['context']['event_id'],recovery(),2.2)
        e=evidence(2.3,4);e['hard_safe']=False
        self.assertEqual(engine.tick(STOP,e,2.3).action,'stop')
        self.assertNotEqual(engine.phase,'PROBE_STEP')
    def test_budget_does_not_reset_on_hard_veto_or_cloud_hold(self):
        engine=self.engine(retry_seconds=.1,stall_seconds=.1)
        now=1.;seq=0
        for _ in range(3):
            for j in range(4):
                now+=.2;seq+=1;engine.tick(STOP,evidence(now,seq),now)
            req=engine.request;self.assertIsNotNone(req)
            engine.receive(req['context']['event_id'],None,now,error='timeout')
        for j in range(10):
            now+=.2;seq+=1;engine.tick(STOP,evidence(now,seq),now)
        self.assertEqual(engine.attempts,3);self.assertEqual(engine.phase,'HOLD')
    def test_scene_change_and_wrong_id_invalidate_reply(self):
        engine=self.engine();req=self.pending(engine)
        engine.receive('wrong',recovery(),2.2)
        self.assertEqual(engine.phase,'CLOUD_PENDING')
        engine.receive(req['context']['event_id'],recovery(),2.2)
        e=evidence(2.3,4);e['observation']['mask'][:]=0
        self.assertEqual(engine.tick(STOP,e,2.3).action,'stop')

    def test_right_pivot_yaw_ceiling_and_imu_loss(self):
        for lost_imu in (False,True):
            engine=self.engine();req=self.pending(engine,'pivot-right');scene=recovery()
            scene.update(candidate_id='pivot-right',assessment='right_corner')
            engine.receive(req['context']['event_id'],scene,2.2)
            e=evidence(2.3,4,'pivot-right')
            self.assertEqual(engine.tick(STOP,e,2.3).action,'pivot-right')
            e=evidence(2.35,5,'pivot-right')
            e['imu'].update(yaw_rad=-.1,imu_valid=not lost_imu)
            self.assertEqual(engine.tick(STOP,e,2.35).action,'stop')
            self.assertEqual(engine.phase,'OBSERVE')

    def test_busy_worker_does_not_spawn_timeout_replacements(self):
        engine=self.engine(request_timeout_seconds=.2,retry_seconds=.1)
        req=self.pending(engine)
        for i in range(4,20):engine.tick(STOP,evidence(2.+i*.3,i),2.+i*.3,worker_busy=True)
        self.assertEqual(engine.attempts,1)
        engine.receive(req['context']['event_id'],recovery(),9.)
        self.assertNotEqual(engine.phase,'PROBE_STEP')

    def test_newer_sequence_with_same_capture_does_not_release(self):
        engine=self.engine();req=self.pending(engine)
        engine.receive(req['context']['event_id'],recovery(),2.2)
        e=evidence(2.1,4)
        self.assertEqual(engine.tick(STOP,e,2.3).action,'stop')

    def test_current_candidate_retains_yellow_obstacle_and_sensor_veto(self):
        e=evidence(2.,1);obs=e['observation'];obs['lane_mask']=np.zeros_like(obs['mask']);obs['detections']=[]
        boundary=SimpleNamespace(semantic_sequence=1,semantic_captured_at=1.99,
            semantic_hard_safe=True,yellow_hazard=False)
        state={'controller':{'stopped_at':None},'visual_corner_phase':'cruise'}
        kwargs=dict(now=2.,frame_age=.01,external_allowed=True,perception_valid=True,
                    pivot_verified=False,imu_sample={},local_state=state)
        initial=build_evidence(obs,boundary,STOP,**kwargs)
        self.assertTrue(initial['hard_safe']);self.assertTrue(initial['candidates'])
        obs['detections']=[dict(box=[.4,.6,.6,.9],confidence=.8)]
        self.assertFalse(build_evidence(obs,boundary,STOP,**kwargs)['hard_safe'])
        obs['detections']=[];obs['frame'][400:470,300:340]=[0,255,255]
        self.assertFalse(build_evidence(obs,boundary,STOP,**kwargs)['candidates'])
        kwargs['external_allowed']=False
        self.assertFalse(build_evidence(obs,boundary,STOP,**kwargs)['hard_safe'])


if __name__=='__main__': unittest.main()
