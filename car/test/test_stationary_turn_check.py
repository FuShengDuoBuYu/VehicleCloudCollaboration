"""Offline safety contracts for the stationary-turn calibration tool."""
import copy
import hashlib
import math
import json
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from car.autodrive.tools import check_stationary_turn as tool


PROFILE = {
    "id": "rosmaster_jetson", "backend": "rosmaster",
    "chassis": {"serial_port": "/dev/myserial", "car_type": 1,
                "command_limit": 100, "motor_order": [0, 1, 2, 3],
                "wheel_signs": [1, 1, 1, 1], "delay": .002, "debug": False},
}


class RecordingChassis:
    def __init__(self):
        self.commands = []
        self.stop_error = False

    def set_four_wheels(self, *values):
        self.commands.append(tuple(values))

    def stop(self):
        self.commands.append((0, 0, 0, 0))
        if self.stop_error:
            raise OSError("stop transport failed")


class FakeSession:
    def __init__(self, fail_during_motion=False):
        self.commands = []
        self.closed = False
        self.sequence = 0
        self.fail_during_motion = fail_during_motion

    def set_motor(self, *values):
        self.commands.append(values)

    def poll_available(self):
        self.sequence += 1
        return b""

    def telemetry(self, max_age):
        data = snapshot(yaw=3.1 if not self.commands else -3.1)
        for value in data.values():
            value["sequence"] = self.sequence
        if self.fail_during_motion and any(any(c) for c in self.commands):
            data["encoder"]["valid"] = False
        return data

    def close(self):
        self.closed = True

    def get_status(self):
        return {"healthy": not self.closed, "last_tx": {"acknowledged": False},
                "tx_bytes": 9 * len(self.commands), "closed": self.closed}


class FakeCamera:
    def __init__(self, fail_after=None):
        self.count = 0
        self.fail_after = fail_after
        self.closed = False

    def read(self):
        time.sleep(.02)
        self.count += 1
        if self.fail_after and self.count >= self.fail_after:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.closed = True


class FakeWriter:
    def __init__(self):
        self.count = 0
        self.closed = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.count += 1

    def release(self):
        self.closed = True


def snapshot(age=0, yaw=3.1):
    now = time.monotonic()
    return {
        "attitude": {"valid": True, "stale": False, "age_s": age,
                     "received_monotonic": now-age, "sequence": 1,
                     "value": {"radians": [0., 0., yaw]}},
        "encoder": {"valid": True, "stale": False, "age_s": age,
                    "received_monotonic": now-age, "sequence": 1,
                    "value": {"native_ticks": [0, 0, 0, 0]}},
    }


class StationaryTurnTests(unittest.TestCase):
    def args(self, *extra):
        return tool.build_parser().parse_args([
            "--direction", "right", "--enable-motors",
            "--confirm-motor-motion", "I_UNDERSTAND_MOTORS_WILL_MOVE", *extra])

    def test_authorization_and_invalid_parameters_fail_before_device_open(self):
        cases = [self.args("--pwm", "29"), self.args("--pwm", "31"),
                 self.args("--duration", "nan"), self.args("--duration", "0"),
                 self.args("--duration", "1.01"),
                 tool.build_parser().parse_args(["--direction", "right"])]
        for args in cases:
            with self.subTest(args=args), patch.object(tool, "open_camera") as camera, \
                    patch.object(tool, "open_session") as serial:
                with self.assertRaises(ValueError):
                    tool.run(args)
                camera.assert_not_called()
                serial.assert_not_called()

    def test_only_verified_mapping_and_rosmaster_type_are_accepted(self):
        tool.validate_profile(PROFILE)
        for key, value in [("motor_order", [0, 2, 1, 3]),
                           ("motor_order", [False, 1, 2, 3]),
                           ("wheel_signs", [1, 1, -1, 1]),
                           ("wheel_signs", [True, 1, 1, 1]),
                           ("car_type", True), ("command_limit", 29)]:
            profile = copy.deepcopy(PROFILE)
            profile["chassis"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                tool.validate_profile(profile)

    def test_direction_targets_select_explicit_forward_or_stationary_rotation(self):
        self.assertEqual(tool.turn_targets("right"), (30, 30, -30, -30))
        self.assertEqual(tool.turn_targets("left"), (-30, -30, 30, 30))
        self.assertEqual(tool.turn_targets("forward"), (30, 30, 30, 30))
        with self.assertRaises(ValueError):
            tool.turn_targets("reverse")

    def test_stale_invalid_or_nonfinite_feedback_cannot_arm(self):
        tool.require_feedback(snapshot())
        for data in [snapshot(age=.5), snapshot(yaw=math.nan)]:
            with self.assertRaises(ValueError):
                tool.require_feedback(data)
        data = snapshot()
        data["encoder"]["valid"] = False
        with self.assertRaises(ValueError):
            tool.require_feedback(data)

    def test_receipt_timestamp_is_checked_even_when_cached_age_claims_fresh(self):
        data = snapshot()
        data["attitude"]["received_monotonic"] -= 1.
        with self.assertRaises(ValueError):
            tool.require_feedback(data)

    def test_wrapped_delta_crosses_pi_boundary(self):
        self.assertAlmostEqual(tool.wrapped_yaw_delta(3.1, -3.1), .083185307179586)

    def test_deadline_stops_even_when_main_thread_does_nothing(self):
        chassis = RecordingChassis()
        guard = tool.PulseGuard(chassis, .06)
        try:
            guard.start(tool.turn_targets("right"))
            self.assertTrue(guard.done.wait(.4))
            self.assertEqual(chassis.commands[0], (30, 30, -30, -30))
            self.assertEqual(chassis.commands[-1], (0, 0, 0, 0))
            self.assertEqual(guard.reason, "duration reached")
            self.assertLess(guard.stop_started - guard.command_started, .25)
        finally:
            guard.stop("cleanup")

    def test_camera_or_telemetry_heartbeat_loss_stops_early(self):
        for missing in ("camera", "telemetry"):
            chassis = RecordingChassis()
            guard = tool.PulseGuard(chassis, 1.)
            guard.start(tool.turn_targets("left"))
            try:
                end = time.monotonic() + .5
                while not guard.done.is_set() and time.monotonic() < end:
                    guard.touch("telemetry" if missing == "camera" else "camera")
                    time.sleep(.015)
                self.assertTrue(guard.done.is_set())
                self.assertIn(missing, guard.reason)
                self.assertEqual(chassis.commands[-1], (0, 0, 0, 0))
            finally:
                guard.stop("cleanup")

    def test_stop_failure_is_recorded_and_never_claims_zero(self):
        chassis = RecordingChassis()
        guard = tool.PulseGuard(chassis, .04)
        chassis.stop_error = True
        guard.start(tool.turn_targets("right"))
        self.assertTrue(guard.done.wait(.4))
        self.assertIn("stop transport failed", guard.error)
        self.assertIsNone(guard.final_pwm)

    def test_cancel_before_start_cannot_later_move(self):
        chassis = RecordingChassis()
        guard = tool.PulseGuard(chassis, .4)
        guard.stop("signal 2")
        with self.assertRaises(RuntimeError):
            guard.start(tool.turn_targets("right"))
        self.assertTrue(all(cmd == (0, 0, 0, 0) for cmd in chassis.commands))

    def test_output_runs_never_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            first = tool.create_run_directory(Path(directory))
            (first / "raw.avi").write_bytes(b"keep")
            second = tool.create_run_directory(Path(directory))
            self.assertNotEqual(first, second)
            self.assertEqual((first / "raw.avi").read_bytes(), b"keep")

    def run_mock_pulse(self, directory, session, camera, *extra):
        writer = FakeWriter()
        profile_path = Path(directory) / "profile.yaml"
        profile_path.write_text("verified fixture")
        with patch.object(tool, "load_vehicle_profile", return_value=(profile_path, PROFILE)), \
                patch.object(tool, "open_session", return_value=session), \
                patch.object(tool, "open_camera", return_value=camera), \
                patch("cv2.VideoWriter", return_value=writer):
            result = tool.run(self.args("--duration", ".08", "--output", str(Path(directory) / "runs"), *extra))
        output = next((Path(directory) / "runs").iterdir())
        report = json.loads((output / "summary.json").read_text())
        return result, report, output, writer

    def test_full_pulse_records_before_after_yaw_and_only_one_motion_command(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession()
            camera = FakeCamera()
            result, report, output, writer = self.run_mock_pulse(directory, session, camera)
            self.assertEqual(result, 0)
            self.assertEqual([c for c in session.commands if any(c)], [(30, 30, -30, -30)])
            self.assertEqual(session.commands[-1], (0, 0, 0, 0))
            self.assertEqual(report["final_pwm"], [0, 0, 0, 0])
            self.assertAlmostEqual(report["yaw_delta_rad"], .083185307179586)
            self.assertFalse(report["ninety_degree_turn_verified"])
            self.assertFalse(report["physical_stop_verified"])
            self.assertTrue(session.closed and camera.closed and writer.closed)
            self.assertEqual(report["video_frames"], writer.count)
            self.assertFalse(report["serial_status"]["last_tx"]["acknowledged"])
            self.assertGreater(report["serial_status"]["tx_bytes"], 9)
            rows = [json.loads(line) for line in (output / "telemetry.jsonl").read_text().splitlines()]
            self.assertIn("before", [row["phase"] for row in rows])
            self.assertIn("after", [row["phase"] for row in rows])
            self.assertGreaterEqual(report["after"]["attitude"]["received_monotonic"] - report["stop"]["stop_completed"], .3)
            self.assertGreaterEqual(report["post_stop_video_s"], .3)

    def test_telemetry_loss_during_pulse_records_failure_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession(fail_during_motion=True)
            result, report, _, _ = self.run_mock_pulse(directory, session, FakeCamera())
            self.assertEqual(result, 1)
            self.assertEqual(session.commands[-1], (0, 0, 0, 0))
            self.assertIsNone(report["yaw_delta_rad"])
            self.assertIn("fresh encoder", " ".join(report["errors"]))

    def test_camera_loss_during_pulse_records_failure_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession()
            result, report, _, _ = self.run_mock_pulse(directory, session, FakeCamera(fail_after=2))
            self.assertEqual(result, 1)
            self.assertEqual(session.commands[-1], (0, 0, 0, 0))
            self.assertIn("camera", " ".join(report["errors"]))

    def test_slow_first_camera_frame_is_allowed_before_any_motor_command(self):
        session = FakeSession()
        class SlowCamera(FakeCamera):
            def read(self):
                if self.count == 0:
                    time.sleep(.35)
                    if any(any(c) for c in session.commands):
                        raise AssertionError("motion before camera preflight")
                return super().read()
        with tempfile.TemporaryDirectory() as directory:
            result, report, _, _ = self.run_mock_pulse(directory, session, SlowCamera())
        self.assertEqual(result, 0)
        self.assertGreater(report["video_frames"], 0)

    def test_first_frame_wait_has_one_total_deadline(self):
        release = threading.Event()
        class BlockedCamera(FakeCamera):
            def read(self):
                release.wait(2.)
                return super().read()
        frames = tool.CameraFrames(BlockedCamera(), startup_timeout=.1)
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(OSError, "startup"):
                frames.next()
            self.assertLess(time.monotonic() - started, .3)
            started = time.monotonic()
            with self.assertRaisesRegex(OSError, "startup"):
                frames.next()
            self.assertLess(time.monotonic() - started, .05)
        finally:
            release.set()
            frames.close()

    def test_first_frame_warmup_does_not_extend_later_frame_timeout(self):
        release = threading.Event()
        class CameraStallsAfterFirst(FakeCamera):
            def read(self):
                if self.count:
                    release.wait(2.)
                return super().read()
        frames = tool.CameraFrames(CameraStallsAfterFirst())
        try:
            frames.next()
            started = time.monotonic()
            with self.assertRaisesRegex(OSError, "frame timeout"):
                frames.next()
            self.assertLess(time.monotonic() - started, .4)
        finally:
            release.set()
            frames.close()

    def test_repeated_signals_do_not_interrupt_stop_cleanup(self):
        handlers = {}
        def install(sig, handler):
            handlers[sig] = handler
            return signal.SIG_DFL
        session = FakeSession()
        write = session.set_motor
        signals = []
        def interrupted_write(*commands):
            write(*commands)
            if len(signals) < 2:
                signals.append(True)
                handlers[signal.SIGINT](signal.SIGINT, None)
        session.set_motor = interrupted_write
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.signal, "signal", side_effect=install):
            result, report, _, _ = self.run_mock_pulse(directory, session, FakeCamera())
        self.assertEqual(result, 1)
        self.assertEqual(report["final_pwm"], [0, 0, 0, 0])
        self.assertEqual(session.commands[-1], (0, 0, 0, 0))

    def test_angle_bounds_are_checked_before_devices_are_opened(self):
        for value in ("0", "-1", "90.1", "nan", "inf"):
            args = self.args("--target-yaw-degrees", value)
            with self.subTest(value=value), patch.object(tool, "open_camera") as camera:
                with self.assertRaises(ValueError):
                    tool.run(args)
                camera.assert_not_called()

    def test_signed_yaw_progress_and_wrap_are_accumulated_from_new_samples(self):
        for direction, initial, readings in (
                ("left", 179, [-179, -174, -169]),
                ("right", -179, [179, 174, 169])):
            progress = tool.YawProgress(direction, 10., math.radians(initial), 0., 0)
            with self.subTest(direction=direction):
                self.assertFalse(progress.update(math.radians(readings[0]), .1, 1))
                self.assertFalse(progress.update(math.radians(readings[1]), .2, 2))
                self.assertTrue(progress.update(math.radians(readings[2]), .3, 3))
                self.assertAlmostEqual(progress.directional_degrees, 12.)
                # Re-reading an already consumed sequence must not add rotation.
                progress.update(math.radians(readings[2]), .3, 3)
                self.assertAlmostEqual(progress.directional_degrees, 12.)

    def test_wrong_direction_jump_and_initial_no_progress_are_rejected(self):
        for yaw, received, expected in [(-9., .1, "wrong direction"),
                                         (46., .1, "jump"), (0., .81, "no yaw progress")]:
            tracker = tool.YawProgress("left", 30., 0., 0., 0)
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                tracker.update(math.radians(yaw), received, 1)

    def test_angle_target_stops_before_deadline_and_records_poststop_frames(self):
        class TurningSession(FakeSession):
            def __init__(self, direction):
                super().__init__()
                self.yaw = 0.
                self.direction = direction
            def telemetry(self, max_age):
                if self.commands and any(self.commands[-1]):
                    self.yaw += math.radians(6. if self.direction == "left" else -6.)
                data = snapshot(yaw=self.yaw)
                for sample in data.values():
                    sample["sequence"] = self.sequence
                return data
        for direction in ("left", "right"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(direction=direction):
                session = TurningSession(direction)
                result, report, _, _ = self.run_mock_pulse(directory, session, FakeCamera(),
                    "--direction", direction, "--target-yaw-degrees", "15", "--duration", "3")
                self.assertEqual(result, 0)
                self.assertEqual(report["stop"]["reason"], "yaw target reached")
                self.assertTrue(report["target_reached"])
                self.assertAlmostEqual(report["directional_yaw_degrees"], 18.)
                self.assertLess(report["stop"]["stop_started"] - report["stop"]["command_started"], .5)
                self.assertGreaterEqual(report["after"]["attitude"]["received_monotonic"] - report["stop"]["stop_completed"], .3)

    def test_angle_deadline_without_target_hit_is_failure_and_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession()
            result, report, _, _ = self.run_mock_pulse(directory, session, FakeCamera(),
                "--direction", "left", "--target-yaw-degrees", "30")
        self.assertEqual(result, 1)
        self.assertFalse(report["target_reached"])
        self.assertEqual(report["target_yaw_degrees"], 30.)
        self.assertEqual(report["stop"]["reason"], "duration reached")
        self.assertEqual(session.commands[-1], (0, 0, 0, 0))

    def test_wrong_direction_in_full_run_stops_and_reports_measured_negative_progress(self):
        class WrongDirection(FakeSession):
            def telemetry(self, max_age):
                data = snapshot(yaw=math.radians(-9.) if self.commands else 0.)
                for sample in data.values():
                    sample["sequence"] = self.sequence
                return data
        with tempfile.TemporaryDirectory() as directory:
            session = WrongDirection()
            result, report, _, _ = self.run_mock_pulse(directory, session, FakeCamera(),
                "--direction", "left", "--target-yaw-degrees", "15", "--duration", "3")
        self.assertEqual(result, 1)
        self.assertIn("wrong direction", " ".join(report["errors"]))
        self.assertFalse(report["target_reached"])
        self.assertAlmostEqual(report["directional_yaw_degrees"], -9.)
        self.assertEqual(session.commands[-1], (0, 0, 0, 0))

    def test_yaw_no_progress_guard_stops_even_with_fresh_heartbeats(self):
        chassis = RecordingChassis()
        guard = tool.PulseGuard(chassis, 3., require_yaw_progress=True)
        guard.start(tool.turn_targets("left"))
        try:
            end = time.monotonic() + 1.2
            while not guard.done.is_set() and time.monotonic() < end:
                guard.touch("camera")
                guard.touch("telemetry")
                time.sleep(.02)
            self.assertTrue(guard.done.is_set())
            self.assertIn("no yaw progress", guard.reason)
            self.assertLess(guard.stop_started - guard.command_started, 1.1)
            self.assertEqual(chassis.commands[-1], (0, 0, 0, 0))
        finally:
            guard.stop("cleanup")


    def supervisor_fixture(self, directory, **overrides):
        import cv2
        frame = Path(directory) / "basis.png"
        if not frame.exists():
            self.assertTrue(cv2.imwrite(str(frame), np.zeros((24, 32, 3), dtype=np.uint8)))
        now = time.time()
        data = {"version": 1, "proposal_id": "corner-left-01",
                "source": "assistant_visual_supervision", "action": "pivot-left",
                "target_yaw_degrees": 2., "max_duration_s": .08,
                "issued_unix_s": now - 1., "expires_unix_s": now + 20.,
                "basis_frame": str(frame.resolve()),
                "basis_frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                "observation": "Visible right corridor; car remains stopped.",
                "expected_result": "Rotate a small relative angle, stop and inspect again."}
        data.update(overrides)
        proposal = Path(directory) / "proposal.json"
        proposal.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return proposal, data

    def test_supervisor_invalid_contract_is_refused_before_any_device_open(self):
        now = time.time()
        cases = [{"version": True}, {"version": 2}, {"proposal_id": ""},
                 {"proposal_id": "x" * 101}, {"source": "cloud"},
                 {"action": "forward"}, {"action": "pivot-right"},
                 {"target_yaw_degrees": 3.}, {"target_yaw_degrees": 31.},
                 {"target_yaw_degrees": math.nan}, {"target_yaw_degrees": True},
                 {"max_duration_s": .09}, {"max_duration_s": 3.01},
                 {"max_duration_s": math.inf}, {"issued_unix_s": now + 5.},
                 {"expires_unix_s": now - 1.}, {"expires_unix_s": now + 31.},
                 {"issued_unix_s": math.nan}, {"expires_unix_s": True},
                 {"basis_frame": "relative.png"}, {"basis_frame": "/nonexistent/basis.png"},
                 {"basis_frame_sha256": "0" * 64}, {"observation": ""},
                 {"expected_result": " "}, {"uncertainty": []},
                 {"observation": "x" * 2001}, {"pwm": 30},
                 {"motors": {"PWM": 30}}]
        for changes in cases:
            with tempfile.TemporaryDirectory() as directory, self.subTest(changes=changes):
                proposal, _ = self.supervisor_fixture(directory, **changes)
                args = self.args("--direction", "left", "--target-yaw-degrees", "2",
                                 "--duration", ".08", "--supervisor-feedback", str(proposal))
                with patch.object(tool, "open_camera") as camera, patch.object(tool, "open_session") as serial:
                    with self.assertRaises(ValueError):
                        tool.run(args)
                    camera.assert_not_called()
                    serial.assert_not_called()

    def test_supervisor_requires_cli_yaw_target_and_small_pivot_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            proposal, _ = self.supervisor_fixture(directory)
            cases = [self.args("--direction", "left", "--supervisor-feedback", str(proposal)),
                     self.args("--direction", "left", "--target-yaw-degrees", "31",
                               "--duration", "3", "--supervisor-feedback", str(proposal))]
            for args in cases:
                with self.subTest(args=args), patch.object(tool, "open_camera") as camera, \
                        patch.object(tool, "open_session"):
                    with self.assertRaises(ValueError):
                        tool.run(args)
                    camera.assert_not_called()

    def test_supervisor_rejects_empty_or_nonimage_basis_and_duplicate_json_keys(self):
        for contents in (b"", b"not an image"):
            with tempfile.TemporaryDirectory() as directory:
                proposal, data = self.supervisor_fixture(directory)
                Path(data["basis_frame"]).write_bytes(contents)
                data["basis_frame_sha256"] = hashlib.sha256(contents).hexdigest()
                proposal.write_text(json.dumps(data))
                args = self.args("--direction", "left", "--target-yaw-degrees", "2", "--duration", ".08",
                                 "--supervisor-feedback", str(proposal))
                with patch.object(tool, "open_camera") as camera, \
                    patch.object(tool, "open_session"), self.assertRaises(ValueError):
                    tool.run(args)
                camera.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            proposal, _ = self.supervisor_fixture(directory)
            proposal.write_text('{"version": 1, "version": 2}')
            args = self.args("--direction", "left", "--target-yaw-degrees", "2", "--duration", ".08",
                             "--supervisor-feedback", str(proposal))
            with patch.object(tool, "open_camera") as camera, \
                    patch.object(tool, "open_session"), self.assertRaises(ValueError):
                tool.run(args)
            camera.assert_not_called()

    def test_supervisor_execution_preserves_exact_feedback_image_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            proposal, data = self.supervisor_fixture(directory, uncertainty="Small visual correction only.")
            proposal_bytes = proposal.read_bytes()
            frame_bytes = Path(data["basis_frame"]).read_bytes()
            session = FakeSession()
            result, report, output, _ = self.run_mock_pulse(directory, session, FakeCamera(),
                "--direction", "left", "--target-yaw-degrees", "2", "--supervisor-feedback", str(proposal))
            self.assertEqual(result, 0)
            self.assertEqual(report["control_source"], "assistant_visual_supervision")
            self.assertFalse(report["yolopv2_only"])
            feedback = report["supervisor_feedback"]
            self.assertEqual(feedback["proposal_id"], "corner-left-01")
            self.assertEqual(feedback["source_sha256"], hashlib.sha256(proposal_bytes).hexdigest())
            self.assertEqual((output / feedback["feedback_artifact"]).read_bytes(), proposal_bytes)
            self.assertEqual((output / feedback["basis_frame_artifact"]).read_bytes(), frame_bytes)
            self.assertEqual(feedback["basis_frame_sha256"], data["basis_frame_sha256"])
            self.assertTrue(feedback["execution"]["motion_started"])
            self.assertTrue(feedback["execution"]["target_reached"])
            self.assertEqual(feedback["execution"]["final_pwm"], [0, 0, 0, 0])
            self.assertLessEqual(feedback["initial_validation_unix_s"], feedback["motion_validation_unix_s"])
            self.assertLessEqual(feedback["motion_validation_monotonic"], report["stop"]["command_started"])
            self.assertEqual([c for c in session.commands if any(c)], [(-30, -30, 30, 30)])

    def test_supervisor_expiry_during_preflight_never_sends_nonzero_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            proposal, data = self.supervisor_fixture(directory)
            clock = [data["issued_unix_s"] + .5]
            session, camera, writer = FakeSession(), FakeCamera(), FakeWriter()
            profile_path = Path(directory) / "profile.yaml"
            profile_path.write_text("verified fixture")
            def open_and_expire(_index):
                clock[0] = data["expires_unix_s"] + .01
                return camera
            with patch.object(tool.time, "time", side_effect=lambda: clock[0]), \
                    patch.object(tool, "load_vehicle_profile", return_value=(profile_path, PROFILE)), \
                    patch.object(tool, "open_session", return_value=session), \
                    patch.object(tool, "open_camera", side_effect=open_and_expire), \
                    patch("cv2.VideoWriter", return_value=writer):
                result = tool.run(self.args("--direction", "left", "--target-yaw-degrees", "2", "--duration", ".08",
                    "--supervisor-feedback", str(proposal), "--output", str(Path(directory) / "runs")))
            output = next((Path(directory) / "runs").iterdir())
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(any(any(command) for command in session.commands))
            self.assertIn("expired", " ".join(report["errors"]))
            self.assertFalse(report["supervisor_feedback"]["execution"]["motion_started"])
            self.assertIsNone(report["stop"]["command_started"])
            self.assertEqual(report["final_pwm"], [0, 0, 0, 0])
            self.assertEqual((output / "supervisor-feedback.json").read_bytes(), proposal.read_bytes())


    def test_supervisor_basis_must_be_jpeg_or_png_encoding(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            proposal, data = self.supervisor_fixture(directory)
            ok, encoded = cv2.imencode(".bmp", np.zeros((24, 32, 3), dtype=np.uint8))
            self.assertTrue(ok)
            content = encoded.tobytes()
            Path(data["basis_frame"]).write_bytes(content)
            data["basis_frame_sha256"] = hashlib.sha256(content).hexdigest()
            proposal.write_text(json.dumps(data))
            args = self.args("--direction", "left", "--target-yaw-degrees", "2", "--duration", ".08",
                             "--supervisor-feedback", str(proposal))
            with patch.object(tool, "open_camera") as camera, \
                    patch.object(tool, "open_session"), self.assertRaises(ValueError):
                tool.run(args)
            camera.assert_not_called()


    def test_forward_requires_supervisor_feedback_before_device_open(self):
        args = self.args("--direction", "forward", "--duration", "1")
        with patch.object(tool, "open_camera") as camera, patch.object(tool, "open_session") as serial:
            with self.assertRaises(ValueError):
                tool.run(args)
            camera.assert_not_called()
            serial.assert_not_called()

    def test_supervisor_forward_one_second_records_fixed_pwm_deadline_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            proposal, _ = self.supervisor_fixture(directory, action="forward", target_yaw_degrees=0,
                                                  max_duration_s=1.)
            session = FakeSession()
            result, report, output, _ = self.run_mock_pulse(directory, session, FakeCamera(),
                "--direction", "forward", "--duration", "1", "--supervisor-feedback", str(proposal))
            self.assertEqual(result, 0)
            self.assertEqual([c for c in session.commands if any(c)], [(30, 30, 30, 30)])
            self.assertEqual(session.commands[-1], (0, 0, 0, 0))
            self.assertEqual(report["stop"]["reason"], "duration reached")
            self.assertGreaterEqual(report["stop"]["stop_started"] - report["stop"]["command_started"], 1.)
            self.assertLess(report["stop"]["stop_started"] - report["stop"]["command_started"], 1.3)
            self.assertEqual(report["control_source"], "assistant_visual_supervision")
            self.assertFalse(report["yolopv2_only"])
            self.assertIsNone(report["target_yaw_degrees"])
            self.assertFalse(report["target_reached"])
            self.assertIsNone(report["directional_yaw_degrees"])
            self.assertEqual(report["supervisor_feedback"]["action"], "forward")
            self.assertEqual(report["supervisor_feedback"]["execution"]["status"], "pulse_recorded")
            self.assertEqual(report["supervisor_feedback"]["execution"]["final_pwm"], [0, 0, 0, 0])
            self.assertEqual((output / "supervisor-feedback.json").read_bytes(), proposal.read_bytes())
            self.assertGreaterEqual(report["post_stop_video_s"], .3)

    def test_supervisor_forward_rejects_invalid_angle_duration_expiry_and_other_actions_before_devices(self):
        now = time.time()
        cases = [({"target_yaw_degrees": 2}, ()),
                 ({"action": "pivot-right"}, ()),
                 ({"action": "reverse"}, ()),
                 ({"action": "forward", "expires_unix_s": now - 1., "issued_unix_s": now - 2.}, ()),
                 ({"max_duration_s": 1.01}, ("--duration", "1.01")),
                 ({}, ("--target-yaw-degrees", "1"))]
        for change, extra in cases:
            with tempfile.TemporaryDirectory() as directory, self.subTest(change=change, extra=extra):
                values = {"action": "forward", "target_yaw_degrees": 0, "max_duration_s": 1.}
                values.update(change)
                proposal, _ = self.supervisor_fixture(directory, **values)
                args = self.args("--direction", "forward", "--duration", "1",
                                 "--supervisor-feedback", str(proposal), *extra)
                with patch.object(tool, "open_camera") as camera, patch.object(tool, "open_session") as serial:
                    with self.assertRaises(ValueError):
                        tool.run(args)
                    camera.assert_not_called()
                    serial.assert_not_called()

    def test_unsupported_direction_is_rejected_before_devices_even_for_programmatic_args(self):
        args = self.args()
        args.direction = "reverse"
        with tempfile.TemporaryDirectory() as directory:
            args.output = Path(directory)
            with patch.object(tool, "open_camera", return_value=FakeCamera()) as camera, \
                    patch.object(tool, "open_session", return_value=FakeSession()) as serial, \
                    patch("cv2.VideoWriter", return_value=FakeWriter()):
                with self.assertRaises(ValueError):
                    tool.run(args)
                camera.assert_not_called()
                serial.assert_not_called()


if __name__ == "__main__":
    unittest.main()
