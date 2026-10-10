from pathlib import Path
import os
import json
import signal
import shutil
import subprocess
import tempfile
import time
import unittest

class LauncherTest(unittest.TestCase):
    def run_case(self, args):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bin_dir=root/'bin';bin_dir.mkdir()
            source=Path(__file__).resolve().parents[2]/'run_vehicle_experiment.sh'
            script=root/source.name;shutil.copy2(source,script)
            runner=root/'run_jetson_yolopv2.sh';runner.write_text('#!/usr/bin/env bash\nexit 7\n');runner.chmod(0o755)
            systemctl=bin_dir/'systemctl';systemctl.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$LAUNCH_TEST_LOG"\nexit 0\n');systemctl.chmod(0o755)
            sudo=bin_dir/'sudo';sudo.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n');sudo.chmod(0o755)
            env=dict(os.environ,PATH=str(bin_dir)+':'+os.environ['PATH'],LAUNCH_TEST_LOG=str(root/'log'))
            result=subprocess.run(['bash',str(script)]+args,env=env,timeout=4)
            return result.returncode,(root/'log').read_text()

    def test_dry_run_keeps_serial_and_restores_camera_on_failure(self):
        code,log=self.run_case([])
        self.assertEqual(code,7)
        self.assertNotIn('serial',log)
        self.assertIn('stop vehicle-perception-monitor.service',log)
        self.assertIn('start vehicle-perception-monitor.service',log)

    def test_motor_run_restores_both_previously_active_monitors(self):
        code,log=self.run_case(['--enable-motors'])
        self.assertEqual(code,7)
        self.assertIn('stop vehicle-serial-monitor.service',log)
        self.assertIn('start vehicle-serial-monitor.service',log)

    def test_terminal_signal_is_forwarded_once_to_isolated_child_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bin_dir=root/'bin';bin_dir.mkdir()
            source=Path(__file__).resolve().parents[2]/'run_vehicle_experiment.sh'
            script=root/source.name;shutil.copy2(source,script)
            runner=root/'run_jetson_yolopv2.sh'
            runner.write_text('''#!/usr/bin/env python3
import json, os, pathlib, signal, time
root = pathlib.Path(os.environ['LAUNCH_TEST_ROOT'])
received = []
def interrupt(signum, frame):
    received.append(signum)
signal.signal(signal.SIGINT, interrupt)
(root/'child.json').write_text(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}))
while not received:
    time.sleep(.01)
time.sleep(.15)
(root/'cleanup.json').write_text(json.dumps(received))
raise SystemExit(7)
''');runner.chmod(0o755)
            systemctl=bin_dir/'systemctl'
            systemctl.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
root=pathlib.Path(os.environ['LAUNCH_TEST_ROOT'])
with (root/'log').open('a') as log:
    log.write(' '.join(sys.argv[1:])+' cleanup='+str((root/'cleanup.json').exists())+'\\n')
''');systemctl.chmod(0o755)
            sudo=bin_dir/'sudo';sudo.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n');sudo.chmod(0o755)
            env=dict(os.environ,PATH=str(bin_dir)+':'+os.environ['PATH'],LAUNCH_TEST_ROOT=str(root))
            process=subprocess.Popen(['bash',str(script),'--enable-motors'],env=env,start_new_session=True)
            child=None
            try:
                deadline=time.monotonic()+3
                while not (root/'child.json').exists() and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertTrue((root/'child.json').exists())
                child=json.loads((root/'child.json').read_text())
                self.assertNotEqual(child['pgid'],process.pid)
                os.killpg(process.pid,signal.SIGINT)
                time.sleep(.04)
                os.killpg(process.pid,signal.SIGINT)
                self.assertEqual(process.wait(timeout=3),7)
                self.assertEqual(json.loads((root/'cleanup.json').read_text()),[signal.SIGINT])
                log=(root/'log').read_text()
                for unit in ('vehicle-perception-monitor.service','vehicle-serial-monitor.service'):
                    self.assertIn('start '+unit+' cleanup=True',log)
            finally:
                if process.poll() is None:
                    if child is not None:
                        try:os.kill(child['pid'],signal.SIGTERM)
                        except ProcessLookupError:pass
                    try:process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=2)

if __name__=='__main__':unittest.main()
