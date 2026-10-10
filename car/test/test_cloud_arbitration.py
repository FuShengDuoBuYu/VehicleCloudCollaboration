import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.control.cloud_arbitration import CloudArbitrationConfig, SceneArbiter

def scene(recommendation='resume_candidate'):
    return {'scene_summary':'test scene','road_state':'clear','objects':[],'signs':[],
            'risk_level':'low','recommendation':recommendation,'route_hint':'none',
            'uncertainties':[],'reason':'test evidence'}

def observation(sequence, now, changed=False):
    mask=np.zeros((32,32),np.uint8);mask[8:,8:24]=1
    if changed:mask[:,16:]=0
    return {'sequence':sequence,'captured_at':now,'mask':mask}

class ArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.a=SceneArbiter(CloudArbitrationConfig(enabled=True,trigger_observations=2,resume_observations=3,request_timeout_seconds=2,response_max_age_seconds=3,retry_seconds=1))
    def trigger(self):
        self.a.observe(observation(1,0),False,'obstacle',0)
        request=self.a.observe(observation(2,.1),False,'obstacle',.1)
        self.assertIsNotNone(request);self.assertFalse(self.a.allowed)
        return request
    def test_initial_safe_driving_does_not_request_cloud(self):
        self.assertIsNone(self.a.observe(observation(1,0),True,'ok',0));self.assertTrue(self.a.allowed)
    def test_repeated_bad_cache_does_not_count_as_two_observations(self):
        for _ in range(6):self.assertIsNone(self.a.observe(observation(1,0),False,'obstacle',0))
    def test_temporal_change_stops_and_requests_even_if_local_path_is_valid(self):
        self.a.observe(observation(1,0),True,'ok',0)
        request=self.a.observe(observation(2,.1,True),True,'ok',.1)
        self.assertIsNotNone(request);self.assertFalse(self.a.allowed)
    def test_resume_requires_matching_reply_and_three_new_safe_observations(self):
        request=self.trigger();self.a.receive(request['event_id'],scene(),.2)
        self.a.observe(observation(3,.3),True,'ok',.3)
        for _ in range(5):self.a.observe(observation(3,.3),True,'ok',.3)
        self.assertFalse(self.a.allowed)
        self.a.observe(observation(4,.4),True,'ok',.4);self.assertFalse(self.a.allowed)
        self.a.observe(observation(5,.5),True,'ok',.5);self.assertTrue(self.a.allowed)
    def test_cloud_clear_does_not_override_local_stop(self):
        request=self.trigger();self.a.receive(request['event_id'],scene(),.2)
        for i in range(3,8):self.a.observe(observation(i,i/10),False,'obstacle',i/10)
        self.assertFalse(self.a.allowed)
    def test_wrong_event_response_is_ignored(self):
        self.trigger();self.a.receive('other-run',scene(),.2)
        for i in range(3,8):self.a.observe(observation(i,i/10),True,'ok',i/10)
        self.assertFalse(self.a.allowed)
    def test_timeout_late_response_stays_stopped(self):
        request=self.trigger();self.a.observe(None,False,'stale',2.2);self.a.receive(request['event_id'],scene(),2.3)
        self.assertFalse(self.a.allowed);self.assertNotEqual(self.a.phase,'validating')
    def test_unknown_bad_schema_and_route_advice_never_resume(self):
        for reply in ({'pwm':20},scene('wait'),dict(scene('route_candidate'),route_hint='right')):
            self.setUp();request=self.trigger();self.a.receive(request['event_id'],reply,.2)
            for i in range(3,8):self.a.observe(observation(i,i/10),True,'ok',i/10)
            self.assertFalse(self.a.allowed)
    def test_new_scene_invalidates_inflight_or_clear_reply(self):
        old=self.trigger();self.a.receive(old['event_id'],scene(),.2)
        new=self.a.observe(observation(3,.3,True),True,'ok',.3)
        self.assertIsNotNone(new);self.assertNotEqual(new['event_id'],old['event_id'])
        self.a.receive(old['event_id'],scene(),.4);self.assertFalse(self.a.allowed)
    def test_stale_or_future_observation_cannot_release_hold(self):
        request=self.trigger();self.a.receive(request['event_id'],scene(),.2)
        for i in range(3,8):self.a.observe(observation(i,-1),True,'ok',.3)
        self.assertFalse(self.a.allowed)
        self.a.observe(observation(9,10),True,'ok',.4);self.assertFalse(self.a.allowed)
    def test_expired_approval_does_not_release(self):
        request=self.trigger();self.a.receive(request['event_id'],scene(),.2)
        self.a.observe(observation(3,3.3),True,'ok',3.3);self.assertFalse(self.a.allowed)
    def test_invalid_configuration_rejected(self):
        for change in ({'enabled':'false'},{'request_timeout_seconds':float('nan')},{'resume_observations':True},{'mask_change_threshold':0}):
            with self.subTest(change=change),self.assertRaises(ValueError):CloudArbitrationConfig(**change)

if __name__=='__main__':unittest.main()
