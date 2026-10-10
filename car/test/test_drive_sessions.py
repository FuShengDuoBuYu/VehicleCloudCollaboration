"""Panel lifecycle tests launch only a synthetic Python child, never hardware."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    sessions=importlib.import_module('autodrive.web.drive_sessions')
except ModuleNotFoundError as exc:
    if exc.name!='autodrive.web.drive_sessions':raise
    sessions=None


class DriveSessionTests(unittest.TestCase):
    def test_panel_session_has_no_total_runtime_limit(self):
        command=self.manager._command(Path(self.temp.name))
        self.assertEqual(command[command.index('--max-runtime-seconds')+1],'0')
        from autodrive.runtime.onboard import validate_motor_request, MOTOR_CONFIRMATION
        for seconds in (0,300):
            validate_motor_request(True,None,MOTOR_CONFIRMATION,None,
                image_space_trial=True,max_runtime_seconds=seconds,panel_controlled=True)
        with self.assertRaises(ValueError):
            validate_motor_request(True,None,MOTOR_CONFIRMATION,None,
                image_space_trial=True,max_runtime_seconds=0)

    def test_new_session_does_not_inherit_old_archive_and_waits_for_fresh_model(self):
        self.manager._state.update(run_archive='/old/run',completed_at=1,exit_code=0,
            termination={'stopped':True},sensor_recording={'old':True})
        state=self.manager.start('fresh');time.sleep(.15)
        for name in ('run_archive','completed_at','exit_code','termination','sensor_recording'):
            self.assertNotIn(name,state)
        path=Path(state['directory'])/'runtime/status.json'
        path.write_text(json.dumps({'last_result':{'semantic_result_age_s':5.1}}))
        time.sleep(.1)
        self.assertEqual(self.manager.get_state()['state'],'starting')
        path.write_text(json.dumps({'last_result':{'semantic_result_age_s':.1}}))
        deadline=time.monotonic()+1
        while self.manager.get_state()['state']=='starting' and time.monotonic()<deadline:time.sleep(.02)
        self.assertEqual(self.manager.get_state()['state'],'running')
        self.assertEqual(json.loads(path.parent.parent.joinpath('session.json').read_text())['state'],'running')

    def setUp(self):
        self.assertIsNotNone(sessions,'panel session supervisor missing')
        self.temp=tempfile.TemporaryDirectory()
        def command(folder):
            code="import time,signal,sys,json;from pathlib import Path;p=Path(sys.argv[1]);p.mkdir(exist_ok=True);signal.signal(signal.SIGINT,lambda *_:sys.exit(0));time.sleep(30)"
            return [sys.executable,'-B','-c',code,str(folder/'runtime')]
        self.manager=sessions.DriveSessionManager(Path(self.temp.name),command_factory=command)

    def tearDown(self):
        if hasattr(self,'manager'):self.manager.close()
        if hasattr(self,'temp'):self.temp.cleanup()

    def test_one_click_one_session_and_duplicate_request_does_not_start_twice(self):
        first=self.manager.start('request1')
        time.sleep(.15)
        second=self.manager.start('request1')
        self.assertEqual(first['session_id'],second['session_id'])
        with self.assertRaises(ValueError):self.manager.start('request2')
        self.assertEqual(len(list(Path(self.temp.name).glob('*/session.json'))),1)

    def test_stop_and_heartbeat_are_bound_to_the_active_session(self):
        first=self.manager.start('request1');time.sleep(.15)
        with self.assertRaises(ValueError):self.manager.heartbeat('other')
        with self.assertRaises(ValueError):self.manager.stop('other')
        self.manager.stop(first['session_id'])
        self.assertTrue((Path(first['directory'])/'control/stop.json').is_file())
        deadline=time.monotonic()+2
        while self.manager.get_state()['state']=='stopping' and time.monotonic()<deadline:time.sleep(.02)
        self.assertFalse(self.manager.get_state()['running'])
        self.assertFalse(self.manager.get_state()['stopped_verified'])

    def test_no_operator_heartbeat_expires_and_cancels_child(self):
        first=self.manager.start('request1');time.sleep(.15)
        lease=Path(first['directory'])/'control/lease.json'
        lease.write_text(json.dumps({'expires_monotonic':time.monotonic()-.01}))
        deadline=time.monotonic()+2
        while self.manager.get_state()['running'] and time.monotonic()<deadline:time.sleep(.02)
        self.assertFalse(self.manager.get_state()['running'])
        self.assertTrue((lease.parent/'stop.json').exists())


if __name__=='__main__':unittest.main()
