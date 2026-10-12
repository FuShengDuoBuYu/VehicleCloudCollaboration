"""Public launcher tests run a fake interpreter and fake services only."""
import json
import os
from pathlib import Path
import shutil
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]


from tests.launcher_fixture import copy_launcher as launcher_tree


class VehicleCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.assertTrue((ROOT/'vehicle').is_file(),'single vehicle launcher is missing')
        launcher_tree(self.root)
        self.bin=self.root/'bin';self.bin.mkdir();self.log=self.root/'calls.jsonl'
        self.runner=self.bin/'runtime'
        self.runner.write_text('#!'+sys.executable+'''\nimport json,os,sys,time,signal
from pathlib import Path
log=Path(os.environ['CLI_TEST_LOG'])
def record(data):
    with log.open('a') as stream:stream.write(json.dumps(data)+'\\n')
if os.environ.get('CLI_TEST_PROMPT'):
    record({'pid':os.getpid()})
    answer=input('Ready pulse? ')
    record({'answer':answer})
    print('confirmed')
elif os.environ.get('CLI_TEST_WAIT'):
    signals=[]
    signal.signal(signal.SIGINT,lambda signum,frame:signals.append(signum))
    record({'child':sys.argv[1:],'pid':os.getpid(),'pgid':os.getpgrp()})
    while not signals:time.sleep(.01)
    time.sleep(.15)
    record({'cleanup':signals})
else:record({'child':sys.argv[1:]})
raise SystemExit(int(os.environ.get('CLI_TEST_EXIT','0')))
''');self.runner.chmod(0o755)
        systemctl=self.bin/'systemctl';systemctl.write_text('#!'+sys.executable+'''\nimport json,os,sys
with open(os.environ['CLI_TEST_LOG'],'a') as stream:stream.write(json.dumps({'service':sys.argv[1:]})+'\\n')
raise SystemExit(int(os.environ.get('CLI_TEST_'+sys.argv[1].upper().replace('-','_'),'0')))
''');systemctl.chmod(0o755)
        sudo=self.bin/'sudo';sudo.write_text('#!/bin/sh\nshift\nexec "$@"\n');sudo.chmod(0o755)
        self.env=dict(os.environ,PATH=str(self.bin)+':'+os.environ['PATH'],
                      CAR_PYTHON=str(self.runner),CLI_TEST_LOG=str(self.log))

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def invoke(self,*args,**env):
        return subprocess.run(['bash',str(self.root/'vehicle'),*args],env=dict(self.env,**env),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=6)

    def test_no_arguments_and_help_have_no_side_effects(self):
        for args in ((),('--help',),('drive','--help'),('trial','--help'),('wheel-check','--help')):
            with self.subTest(args=args):self.assertEqual(self.invoke(*args).returncode,0)
        self.assertEqual(self.records(),[])
        self.assertFalse((self.root/'outputs').exists())

    def test_drive_uses_current_profile_and_defaults_to_no_motor(self):
        result=self.invoke('drive','--max-samples','8','--output-dir','path with spaces',CLI_TEST_EXIT='7')
        self.assertEqual(result.returncode,7,result.stderr)
        records=self.records();argv=next(r['child'] for r in records if 'child' in r)
        self.assertEqual(argv[argv.index('--vehicle-profile')+1],'rosmaster_jetson_yolopv2_visual_feedback')
        self.assertNotIn('--enable-motors',argv);self.assertIn('path with spaces',argv)
        self.assertEqual([r['service'] for r in records if 'service'in r],[
            ['is-active','--quiet','vehicle-perception-monitor.service'],
            ['stop','vehicle-perception-monitor.service'],['start','vehicle-perception-monitor.service']])

    def test_unconfirmed_motor_or_trial_request_never_touches_services(self):
        for args in (('drive','--enable-motors'),('trial','10'),('trial','0'),('trial','121')):
            self.assertNotEqual(self.invoke(*args).returncode,0)
        self.assertEqual(self.records(),[])

    def test_confirmed_drive_releases_both_monitors_and_restores_after_child(self):
        result=self.invoke('drive','--enable-motors','--confirm-motor-motion','I_UNDERSTAND_MOTORS_WILL_MOVE')
        self.assertEqual(result.returncode,0,result.stderr)
        records=self.records();child=next(i for i,r in enumerate(records) if 'child'in r)
        for unit in ('vehicle-perception-monitor.service','vehicle-serial-monitor.service'):
            self.assertLess(records.index({'service':['stop',unit]}),child)
            self.assertGreater(records.index({'service':['start',unit]}),child)

    def test_failed_release_prevents_child_and_attempts_restore(self):
        result=self.invoke('drive',CLI_TEST_STOP='1')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any('child'in r for r in self.records()))
        self.assertIn({'service':['start','vehicle-perception-monitor.service']},self.records())

    def test_restore_failure_is_not_reported_as_success(self):
        result=self.invoke('drive',CLI_TEST_START='1')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('restore',result.stderr.lower())

    def test_inactive_services_remain_inactive(self):
        self.assertEqual(self.invoke('drive',CLI_TEST_IS_ACTIVE='3').returncode,0)
        self.assertFalse(any(r.get('service',[''])[0]in ('stop','start') for r in self.records()))

    def test_unknown_service_state_prevents_child(self):
        self.assertNotEqual(self.invoke('drive',CLI_TEST_IS_ACTIVE='1').returncode,0)
        self.assertFalse(any('child'in r for r in self.records()))

    def test_replay_is_explicitly_recorded_and_never_leases_devices(self):
        self.assertEqual(self.invoke('replay','--run','/example/run').returncode,0)
        records=self.records();self.assertEqual(len(records),1)
        argv=records[0]['child'];self.assertEqual(argv[argv.index('--mode')+1],'recorded')
        self.assertEqual(argv[argv.index('--profile')+1],'recorded')

    def test_monitor_forwards_profile_as_one_argument(self):
        result=self.invoke('monitor','perception','--vehicle-profile','selected experiment.yaml')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.records()[0]['child'],['-B','-m','autodrive.web.perception_monitor',
                                                  '--vehicle-profile','selected experiment.yaml'])

    def test_compatibility_scripts_forward_to_the_single_dispatcher(self):
        for script,args,module in (
            ('run_vehicle_dashboard.sh',[], 'autodrive.web.dashboard'),
            ('run_vehicle_dashboard.sh',['serial'],'autodrive.web.sensor_publishers'),
            ('run_autodrive_panel.sh',[], 'autodrive.web.drive_sessions'),
            ('run_vehicle_experiment.sh',[], 'run_onboard.py')):
            with self.subTest(script=script,args=args):
                if self.log.exists():self.log.unlink()
                result=subprocess.run(['bash',str(self.root/script),*args],env=self.env,
                    capture_output=True,text=True,timeout=4)
                self.assertEqual(result.returncode,0,result.stderr)
                children=[r['child'] for r in self.records() if 'child'in r]
                self.assertEqual(len(children),1)
                self.assertTrue(any(module in value for value in children[0]))

    def test_trial_is_bounded_and_replays_only_after_drive_cleanup(self):
        result=self.invoke('trial','10','--confirm-motor-motion','I_UNDERSTAND_MOTORS_WILL_MOVE')
        self.assertEqual(result.returncode,0,result.stderr)
        records=self.records();children=[(i,r['child'])for i,r in enumerate(records)if 'child'in r]
        self.assertEqual(len(children),2)
        drive=children[0][1];replay=children[1][1]
        self.assertEqual(drive[drive.index('--max-runtime-seconds')+1],'30')
        self.assertEqual(drive[-2:],['--max-runtime-seconds','10'])
        self.assertIn('--enable-motors',drive)
        self.assertEqual(replay[replay.index('--mode')+1],'recorded')
        self.assertIn('--after',replay)
        for unit in ('vehicle-perception-monitor.service','vehicle-serial-monitor.service'):
            restored=records.index({'service':['start',unit]})
            self.assertGreater(restored,children[0][0]);self.assertLess(restored,children[1][0])

    def test_sensor_dispatch_preserves_container_isolation_and_argument_boundaries(self):
        docker=self.bin/'docker'
        docker.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
            'open(os.environ["CLI_TEST_LOG"],"a").write(json.dumps({"docker":sys.argv[1:]})+"\\n")\n')
        docker.chmod(0o755)
        result=self.invoke('monitor','sensors','--label','literal argument with spaces')
        self.assertEqual(result.returncode,0,result.stderr)
        argv=self.records()[0]['docker']
        self.assertEqual(argv[argv.index('--network')+1],'none')
        self.assertEqual(argv[argv.index('--cap-drop')+1],'ALL')
        self.assertEqual(argv[-2:],['--label','literal argument with spaces'])
        self.assertIn(str(self.root/'car')+':/workspace/car:ro',argv)
        self.assertFalse(any('/dev/myserial' in part for part in argv))

    def test_terminal_signal_forwarded_once_and_cleanup_before_restore(self):
        process=subprocess.Popen(['bash',str(self.root/'vehicle'),'drive'],
            env=dict(self.env,CLI_TEST_WAIT='1',CLI_TEST_EXIT='7'),start_new_session=True,
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        child=None
        try:
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                child=next((r for r in self.records() if 'pid'in r),None)
                if child:break
                time.sleep(.01)
            self.assertIsNotNone(child)
            self.assertNotEqual(child['pgid'],process.pid)
            os.killpg(process.pid,signal.SIGINT);time.sleep(.04);os.killpg(process.pid,signal.SIGINT)
            self.assertEqual(process.wait(timeout=3),7)
            records=self.records();cleanup=records.index({'cleanup':[signal.SIGINT]})
            self.assertGreater(records.index({'service':['start','vehicle-perception-monitor.service']}),cleanup)
        finally:
            if process.poll()is None:
                if child:
                    try:os.kill(child['pid'],signal.SIGTERM)
                    except ProcessLookupError:pass
                process.kill();process.wait()
            if process.stderr:process.stderr.close()

    def test_wheel_prompt_visible_before_input_and_preserved_in_console_log(self):
        process=subprocess.Popen(['bash',str(self.root/'vehicle'),'wheel-check',
            '--confirm-wheels-lifted','WHEELS_ARE_LIFTED'],
            env=dict(self.env,CLI_TEST_PROMPT='1'),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            seen=b''
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout,selectors.EVENT_READ)
                deadline=time.monotonic()+3
                while time.monotonic()<deadline and b'Ready pulse? ' not in seen:
                    if selector.select(timeout=.1):seen+=os.read(process.stdout.fileno(),4096)
            self.assertIn(b'Ready pulse? ',seen,'interactive prompt hidden before input')
            rest,_=process.communicate(b'y\n',timeout=3)
            self.assertEqual(process.returncode,0)
            self.assertIn(b'confirmed',rest)
            self.assertIn({'answer':'y'},self.records())
            console=next((self.root/'outputs/manual_wheel_check').glob('*/console.log')).read_text()
            self.assertIn('Ready pulse? ',console)
            self.assertIn('confirmed',console)
        finally:
            if process.poll()is None:
                for record in self.records():
                    if 'pid'in record:
                        try:os.kill(record['pid'],signal.SIGTERM)
                        except ProcessLookupError:pass
                process.communicate(timeout=3)
            for stream in (process.stdin,process.stdout):
                if stream:stream.close()

    def test_cancelled_trial_never_replays_even_if_child_exits_with_error(self):
        process=subprocess.Popen(['bash',str(self.root/'vehicle'),'trial','10',
            '--confirm-motor-motion','I_UNDERSTAND_MOTORS_WILL_MOVE'],
            env=dict(self.env,CLI_TEST_WAIT='1',CLI_TEST_EXIT='7'),
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            deadline=time.monotonic()+3
            while time.monotonic()<deadline and not any('pid'in r for r in self.records()):time.sleep(.01)
            self.assertTrue(any('pid'in r for r in self.records()))
            process.send_signal(signal.SIGINT)
            deadline=time.monotonic()+2
            while time.monotonic()<deadline and process.poll()is None:
                if len([r for r in self.records() if 'child'in r])>1:break
                time.sleep(.01)
            self.assertEqual(len([r for r in self.records() if 'child'in r]),1)
            self.assertEqual(process.wait(timeout=2),7)
        finally:
            if process.poll()is None:
                for record in self.records():
                    if 'pid'in record:
                        try:os.kill(record['pid'],signal.SIGTERM)
                        except ProcessLookupError:pass
                process.communicate(timeout=3)
            if process.stderr:process.stderr.close()
