"""Launcher ownership and interruption tests; no device or real service access."""
from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


class WheelCheckLauncherTest(unittest.TestCase):
    def run_case(self, mode="ok", *, active=True, stop_fail=False,
                 start_fail=False, interrupt=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "run_wheel_check.sh"
            shutil.copy2(Path(__file__).resolve().parents[2] / script.name, script)
            python = root / "outputs/field_yolopv2/venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            tool = root / "car/autodrive/tools/check_wheel_directions.py"
            tool.parent.mkdir(parents=True)
            tool.write_text('''import os, sys, time
from pathlib import Path
p = Path(os.environ["LAUNCH_LOG"])
def log(s):
    with p.open("a") as f: f.write(s + "\\n")
if "--help" in sys.argv:
    log("help")
    sys.exit(0)
log("child " + " ".join(sys.argv[1:]))
try:
    if os.environ["LAUNCH_MODE"] == "wait": time.sleep(10)
finally:
    log("STOP")
sys.exit(7 if os.environ["LAUNCH_MODE"] == "fail" else 0)
''')
            bin_dir = root / "bin"
            bin_dir.mkdir()
            systemctl = bin_dir / "systemctl"
            systemctl.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$LAUNCH_LOG"
case "$1" in
is-active) exit "$LAUNCH_INACTIVE";;
stop) exit "$LAUNCH_STOP_FAIL";;
start) exit "$LAUNCH_START_FAIL";;
esac
''')
            sudo = bin_dir / "sudo"
            sudo.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
            for executable in (systemctl, sudo):
                executable.chmod(0o755)
            log = root / "log"
            env = dict(os.environ, PATH=str(bin_dir) + ":" + os.environ["PATH"],
                       LAUNCH_LOG=str(log), LAUNCH_MODE=mode,
                       LAUNCH_INACTIVE=str(int(not active)),
                       LAUNCH_STOP_FAIL=str(int(stop_fail)),
                       LAUNCH_START_FAIL=str(int(start_fail)))
            args = ["--help"] if mode == "help" else [
                "--confirm-wheels-lifted", "WHEELS_ARE_LIFTED", "--all-wheels",
                "--pwm", "20", "--duration", "0.3"]
            process = subprocess.Popen(["bash", str(script)] + args, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if interrupt:
                deadline = time.monotonic() + 3
                while not (log.exists() and "child " in log.read_text()):
                    if time.monotonic() > deadline:
                        process.kill()
                        self.fail("mock child failed to start")
                    time.sleep(0.01)
                process.send_signal(interrupt)
            output, _ = process.communicate(timeout=5)
            records = log.read_text() if log.exists() else ""
            return process.returncode, records, output.decode()

    def test_restores_active_monitor_and_preserves_parameters(self):
        code, log, _ = self.run_case()
        self.assertEqual(code, 0)
        self.assertIn("--pwm 20 --duration 0.3", log)
        self.assertLess(log.index("stop vehicle-serial"), log.index("child "))
        self.assertLess(log.index("STOP"), log.index("start vehicle-serial"))

    def test_child_failure_preserved_after_restore(self):
        code, log, _ = self.run_case("fail")
        self.assertEqual(code, 7)
        self.assertIn("start vehicle-serial", log)

    def test_inactive_monitor_stays_inactive(self):
        code, log, _ = self.run_case(active=False)
        self.assertEqual(code, 0)
        self.assertNotIn("stop vehicle-serial", log)
        self.assertNotIn("start vehicle-serial", log)

    def test_failed_release_prevents_motor_child(self):
        code, log, _ = self.run_case(stop_fail=True)
        self.assertNotEqual(code, 0)
        self.assertNotIn("child ", log)

    def test_restore_failure_reported(self):
        code, _, output = self.run_case(start_fail=True)
        self.assertNotEqual(code, 0)
        self.assertIn("FAILED to restore", output)

    def test_interrupts_stop_child_before_restoring_monitor(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig):
                code, log, _ = self.run_case("wait", interrupt=sig)
                self.assertNotEqual(code, 0)
                self.assertLess(log.index("STOP"), log.index("start vehicle-serial"))

    def test_help_does_not_touch_services(self):
        code, log, _ = self.run_case("help")
        self.assertEqual(code, 0)
        self.assertEqual(log, "help\n")


if __name__ == "__main__":
    unittest.main()
