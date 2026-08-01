import json
from pathlib import Path
import sys
import tempfile
import time
import unittest


CAR_DIR = Path(__file__).resolve().parents[1]
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.lcc_web import (
    LCCProcessManager,
    MOTOR_CONFIRMATION,
    build_lcc_command,
)


class LCCWebTest(unittest.TestCase):
    def test_command_only_enables_motors_when_server_is_armed(self):
        dry = build_lcc_command(
            "/env/bin/python", "/repo/run.py", "/repo/config.yaml", 12, False
        )
        armed = build_lcc_command(
            "/env/bin/python", "/repo/run.py", "/repo/config.yaml", 12, True
        )
        self.assertNotIn("--enable-motors", dry)
        self.assertEqual(dry[-2:], ["--max-runtime-seconds", "12"])
        self.assertIn("--enable-motors", armed)
        self.assertEqual(armed[-1], MOTOR_CONFIRMATION)

    def test_manager_starts_and_stops_one_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "runner.py"
            runner.write_text(
                "import time\nprint('runner ready', flush=True)\ntime.sleep(30)\n",
                encoding="utf-8",
            )
            config = root / "config.yaml"
            config.write_text("version: 1\n", encoding="utf-8")
            output = root / "outputs"
            manager = LCCProcessManager(
                repo_root=root,
                python_executable=sys.executable,
                runner_path=runner,
                config_path=config,
                output_dir=output,
                motors_enabled=False,
                default_max_runtime_seconds=10,
            )
            started = manager.start(max_runtime_seconds=5)
            self.assertTrue(started["process"]["running"])
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                logs = manager.get_state()["process"]["logs"]
                if any("runner ready" in item["message"] for item in logs):
                    break
                time.sleep(0.02)
            else:
                self.fail("runner output was not captured")
            stopped = manager.stop("test stop")
            self.assertFalse(stopped["process"]["running"])
            self.assertEqual(stopped["process"]["state"], "stopped")

    def test_status_from_before_current_start_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            output.mkdir()
            status = output / "status.json"
            status.write_text(json.dumps({"old": True}), encoding="utf-8")
            runner = root / "runner.py"
            runner.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("version: 1\n", encoding="utf-8")
            time.sleep(0.02)
            manager = LCCProcessManager(
                repo_root=root,
                python_executable=sys.executable,
                runner_path=runner,
                config_path=config,
                output_dir=output,
                motors_enabled=False,
                default_max_runtime_seconds=10,
            )
            manager.start(max_runtime_seconds=5)
            try:
                self.assertIsNone(manager.get_state()["runtime"])
            finally:
                manager.stop("test cleanup")

    def test_runtime_bounds_and_motor_confirmation_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "runner.py"
            runner.write_text("pass\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("version: 1\n", encoding="utf-8")
            manager = LCCProcessManager(
                repo_root=root,
                python_executable=sys.executable,
                runner_path=runner,
                config_path=config,
                output_dir=root / "outputs",
                motors_enabled=True,
            )
            with self.assertRaisesRegex(ValueError, "1 and 300"):
                manager.start(max_runtime_seconds=0, confirmation=MOTOR_CONFIRMATION)
            with self.assertRaisesRegex(ValueError, "安全确认"):
                manager.start(max_runtime_seconds=5, confirmation="")


if __name__ == "__main__":
    unittest.main()

