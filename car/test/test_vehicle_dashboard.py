import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.web.dashboard import DashboardState, DashboardServer, publish_snapshot


class DashboardTest(unittest.TestCase):
    def test_panel_start_requires_explicit_click_header_and_rejects_cross_origin(self):
        import inspect
        self.assertIn('control_client',inspect.signature(DashboardServer).parameters)
        from autodrive.web.drive_sessions import DriveSessionManager
        manager=DriveSessionManager(self.root/'sessions',command_factory=lambda folder:
            [__import__('sys').executable,'-B','-c','import time;time.sleep(10)'])
        class Client:
            def get_state(self):return manager.get_state()
            def request(self,action,**values):
                return getattr(manager,action)(**values)
        server=DashboardServer(self.state,host='127.0.0.1',port=0,control_client=Client()).make_http_server()
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        address='http://127.0.0.1:'+str(server.server_port)
        try:
            for headers in ({},{'X-Vehicle-Control':'panel','Origin':'http://unrelated.invalid'}):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(address+'/api/drive/start',data=b'{"request_id":"r1"}',headers=headers))
                self.assertEqual(error.exception.code,403)
            headers={'X-Vehicle-Control':'panel','Origin':address,'Content-Type':'application/json'}
            for method in ('PUT','DELETE'):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(address+'/api/drive/start',data=b'{"request_id":"r1"}',headers=headers,method=method))
                self.assertEqual(error.exception.code,403)
                self.assertFalse(manager.get_state()['running'])
            with urlopen(address+'/api/health') as reply:
                self.assertFalse(json.load(reply)['read_only'])
            with urlopen(Request(address+'/api/drive/start',data=b'{"request_id":"r1"}',headers=headers)) as reply:
                started=json.load(reply)
            self.assertTrue(started['running'])
            with urlopen(Request(address+'/api/drive/stop',data=json.dumps({'session_id':started['session_id']}).encode(),headers=headers)) as reply:
                self.assertEqual(json.load(reply)['state'],'stopping')
        finally:
            manager.close();server.shutdown();server.server_close();thread.join(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runtime = self.root / 'runtime'
        self.sensors = self.root / 'sensors'
        self.runtime.mkdir(); self.sensors.mkdir()
        self.state = DashboardState(self.runtime, self.sensors)

    def tearDown(self):
        self.tmp.cleanup()

    def write_runtime(self, value):
        (self.runtime/'status.json').write_text(json.dumps(value))

    def test_missing_feeds_are_explicit_not_healthy(self):
        result = self.state.get_state()
        self.assertTrue(result['read_only'])
        self.assertEqual(result['cloud']['status'], 'not_integrated')
        self.assertEqual(result['navigation']['status'], 'not_integrated')
        self.assertIsNone(result['sensors']['battery']['value'])
        self.assertFalse(result['process']['running'])

    def test_stopped_runtime_never_looks_live_even_with_new_mtime(self):
        self.write_runtime({'termination': {'stopped': True}, 'last_result': {'confidence': .9}})
        result = self.state.get_state()
        self.assertEqual(result['autonomy']['status'], 'stopped')
        self.assertFalse(result['process']['running'])
        self.assertEqual(result['sensors']['camera']['status'], 'stopped')

    def test_external_runtime_becomes_stale_and_old_image_not_live(self):
        self.write_runtime({'last_result': {'confidence': .9, 'frame_age_s': .1}})
        self.assertTrue(self.state.get_state()['process']['running'])
        past=time.time()-20
        os.utime(self.runtime/'status.json', (past,past))
        self.assertEqual(self.state.get_state()['autonomy']['status'], 'stale')

    def test_sensor_age_grows_without_producer_update(self):
        publish_snapshot(self.sensors/'serial.json', {'battery': {'value': {'voltage_v':12.2}, 'age_s': .1, 'status':'online'}})
        result=self.state.get_state()['sensors']['battery']
        self.assertEqual(result['status'],'online')
        self.assertEqual(result['value']['voltage_v'],12.2)
        self.assertEqual(result['value']['percentage'],100)
        self.assertTrue(result['value']['percentage_estimated'])
        p=self.sensors/'serial.json';v=json.loads(p.read_text());v['monotonic_s']-=10;p.write_text(json.dumps(v))
        result=self.state.get_state()['sensors']['battery']
        self.assertEqual(result['status'],'stale')
        self.assertIsNone(result['value']['percentage'])

    def test_previous_boot_and_future_timestamps_rejected(self):
        p=self.sensors/'serial.json'
        publish_snapshot(p, {'imu': {'value': {'gyro_rad_s':[0,0,0]}, 'status':'online','age_s':0}})
        value=json.loads(p.read_text());value['boot_id']='another-boot';p.write_text(json.dumps(value))
        self.assertEqual(self.state.get_state()['sensors']['imu']['status'],'stale')
        publish_snapshot(p, {'imu': {'value': {}, 'status':'online','age_s':0}})
        value=json.loads(p.read_text());value['monotonic_s']+=100;p.write_text(json.dumps(value))
        self.assertEqual(self.state.get_state()['sensors']['imu']['status'],'error')

    def test_bad_sensor_document_and_nonfinite_readings_are_not_online(self):
        (self.sensors/'serial.json').write_text('{bad')
        self.assertEqual(self.state.get_state()['sensors']['battery']['status'],'error')
        publish_snapshot(self.sensors/'serial.json', {'battery': {'value': {'voltage_v': float('nan')}, 'status':'online','age_s':0}})
        result=self.state.get_state()
        self.assertNotEqual(result['sensors']['battery']['status'],'online')
        json.dumps(result, allow_nan=False)

    def test_cloud_fields_allowlisted_and_no_invented_confidence(self):
        self.write_runtime({'last_result': {}, 'cloud': {'status':'completed','event_id':'e1','api_key':'SECRET','result':{'scene_summary':'障碍物','risk_level':'high','recommendation':'stop','reason':'blocked','raw_response':'SECRET'}}})
        result=self.state.get_state()
        self.assertEqual(result['cloud']['event_id'],'e1')
        self.assertIsNone(result['cloud']['confidence'])
        self.assertNotIn('SECRET',json.dumps(result))

    def test_bad_nonbattery_sensor_values_cannot_be_online(self):
        publish_snapshot(self.sensors/'serial.json', {
            'imu': {'status':'online','age_s':0,'value':{'gyro_rad_s':[float('nan')]*3,'accel_m_s2':[0,0,1]}},
            'encoder': {'status':'online','age_s':0,'value':{'native_ticks':'bad'}},
            'attitude': {'status':'online','age_s':0,'value':{'radians':[0,0]}}})
        result=self.state.get_state()['sensors']
        for name in ('imu','encoder','attitude'):
            self.assertEqual(result[name]['status'],'error')

    def test_current_cloud_arbitration_disabled_is_not_missing(self):
        self.write_runtime({'cloud_arbitration':{'enabled':False,'phase':'driving','reason':'cloud disabled','event_id':None}})
        self.assertEqual(self.state.get_state()['cloud']['status'],'disabled')

    def test_cloud_event_result_is_bound_to_this_run(self):
        event='a'*32
        run=self.runtime/'runs'/'r1';folder=run/'cloud'/event;folder.mkdir(parents=True)
        (folder/'request.json').write_text(json.dumps({'event_id':event,'reason':'mask change'}))
        (folder/'response.json').write_text(json.dumps({'event_id':event,'scene':{'scene_summary':'前方障碍','recommendation':'stop'},'metadata':{'request_id':'request1','elapsed_ms':120,'api_key':'SECRET'}}))
        self.write_runtime({'run_archive':str(run),'cloud_arbitration':{'enabled':True,'phase':'hold','reason':'cloud advice: stop','event_id':event}})
        result=self.state.get_state()['cloud']
        self.assertEqual(result['result']['scene_summary'],'前方障碍')
        self.assertEqual(result['request_id'],'request1')
        self.assertEqual(result['trigger_reason'],'mask change')
        self.assertNotIn('SECRET',json.dumps(result))

    def test_invalid_top_level_and_oversized_payload_do_not_crash(self):
        self.write_runtime([1,2,3])
        self.assertEqual(self.state.get_state()['autonomy']['status'],'error')
        (self.sensors/'serial.json').write_text(' '*1100000)
        self.assertEqual(self.state.get_state()['sensors']['battery']['status'],'error')

    def test_http_read_only_health_and_no_path_traversal(self):
        service=DashboardServer(self.state, host='127.0.0.1', port=0)
        server=service.make_http_server()
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        address='http://127.0.0.1:'+str(server.server_port)
        try:
            with urlopen(address+'/api/health') as reply:
                health=json.load(reply)
                self.assertTrue(health['service_alive'])
                self.assertFalse(health['all_sensors_online'])
            with self.assertRaises(HTTPError) as failure:
                urlopen(Request(address+'/api/lcc?action=start',data=b'',method='POST'))
            self.assertEqual(failure.exception.code,403)
            with self.assertRaises(HTTPError) as failure:
                urlopen(address+'/../../etc/passwd')
            self.assertEqual(failure.exception.code,404)
        finally:
            server.shutdown();server.server_close();thread.join(2)

if __name__=='__main__': unittest.main()
