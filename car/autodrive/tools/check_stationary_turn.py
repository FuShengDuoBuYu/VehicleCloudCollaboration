#!/usr/bin/env python3
"""Record one guarded ground stationary-turn pulse; never claims a 90-degree turn.

Exclusive UART and camera access must be arranged by the caller. This tool does
not stop services, initialize a gimbal, or change calibration flags.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROL_DIR = REPO_ROOT / "car" / "control"
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

from vehicle_control.platforms.rosmaster import RosmasterChassis
from vehicle_control.profile import load_vehicle_profile

CONFIRMATION = "I_UNDERSTAND_MOTORS_WILL_MOVE"
MAX_AGE = .25


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("left", "right", "forward"), required=True)
    parser.add_argument("--pwm", type=int, default=30, help="Fixed at 30 for this calibration")
    parser.add_argument("--duration", type=float, default=.4, help="Maximum seconds: <=1 for pulses, <=5 with a yaw target")
    parser.add_argument("--target-yaw-degrees", type=float, default=None,
                        help="Optional relative turn magnitude in (0,90]; stop on fresh yaw feedback")
    parser.add_argument("--supervisor-feedback", type=Path, default=None,
                        help="Optional short-lived, image-bound assistant visual motion proposal; not a YOLO-only run")
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--confirm-motor-motion", default="")
    parser.add_argument("--vehicle-profile", default="rosmaster_jetson")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "stationary_turn")
    return parser



def require_supervisor_lease(feedback):
    """Check the retained proposal immediately before any nonzero command."""
    now = time.time()
    if now < feedback["issued_unix_s"]:
        raise ValueError("supervisor feedback is not yet issued")
    if now >= feedback["expires_unix_s"]:
        raise ValueError("supervisor feedback expired")
    return now


def validate_supervisor_feedback(args):
    """Validate and retain exact input bytes before opening a hardware device."""
    path = getattr(args, "supervisor_feedback", None)
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read supervisor feedback: {exc}") from exc
    if not 0 < len(raw) <= 65536:
        raise ValueError("supervisor feedback must be a nonempty JSON file at most 64 KiB")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate supervisor feedback key: {key}")
            result[key] = value
        return result

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("supervisor feedback must be valid UTF-8 JSON") from exc
    required = {"version", "proposal_id", "source", "action", "target_yaw_degrees",
                "max_duration_s", "issued_unix_s", "expires_unix_s", "basis_frame",
                "basis_frame_sha256", "observation", "expected_result"}
    if not isinstance(data, dict) or not required <= set(data) or set(data) - required - {"uncertainty"}:
        raise ValueError("supervisor feedback has missing or unsupported fields; PWM is not accepted")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("supervisor feedback version must be 1")
    for name, limit in (("proposal_id", 100), ("observation", 2000), ("expected_result", 2000)):
        value = data[name]
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"supervisor {name} must be a nonempty string at most {limit} characters")
    if "uncertainty" in data and (not isinstance(data["uncertainty"], str) or len(data["uncertainty"]) > 2000):
        raise ValueError("supervisor uncertainty must be a string at most 2000 characters")
    if data["source"] != "assistant_visual_supervision":
        raise ValueError("supervisor source must be assistant_visual_supervision")
    expected_action = "forward" if args.direction == "forward" else "pivot-" + args.direction
    if data["action"] not in ("pivot-left", "pivot-right", "forward") or data["action"] != expected_action:
        raise ValueError("supervisor action must match the requested forward or left/right pivot")
    for name in ("target_yaw_degrees", "max_duration_s", "issued_unix_s", "expires_unix_s"):
        if type(data[name]) not in (int, float) or not math.isfinite(data[name]):
            raise ValueError(f"supervisor {name} must be finite numeric data")
    if args.direction == "forward":
        if data["target_yaw_degrees"] != 0 or args.target_yaw_degrees is not None:
            raise ValueError("supervisor forward requires zero proposal target and no CLI yaw target")
        max_duration = 1
    else:
        if (not 0 < data["target_yaw_degrees"] <= 30
                or args.target_yaw_degrees is None or data["target_yaw_degrees"] != args.target_yaw_degrees):
            raise ValueError("supervisor target yaw must match the CLI target inside (0,30] degrees")
        max_duration = 3
    if not 0 < data["max_duration_s"] <= max_duration or data["max_duration_s"] != args.duration:
        raise ValueError(f"supervisor duration must match the CLI duration inside (0,{max_duration}] seconds")
    lease = data["expires_unix_s"] - data["issued_unix_s"]
    if not 0 < lease <= 30:
        raise ValueError("supervisor lease must be inside (0,30] seconds")
    if not isinstance(data["basis_frame"], str) or not Path(data["basis_frame"]).is_absolute():
        raise ValueError("supervisor basis_frame must be an absolute local image path")
    frame = Path(data["basis_frame"]).resolve()
    if frame.suffix.lower() not in (".jpg", ".jpeg", ".png") or not frame.is_file():
        raise ValueError("supervisor basis_frame must be an existing JPEG or PNG file")
    try:
        frame_bytes = frame.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read supervisor basis image: {exc}") from exc
    if not 0 < len(frame_bytes) <= 20 * 1024 * 1024:
        raise ValueError("supervisor basis image must be nonempty and at most 20 MiB")
    frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    if data["basis_frame_sha256"] != frame_hash:
        raise ValueError("supervisor basis image SHA256 mismatch")
    jpeg = frame_bytes.startswith(b"\xff\xd8\xff")
    png = frame_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if (frame.suffix.lower() == ".png" and not png
            or frame.suffix.lower() in (".jpg", ".jpeg") and not jpeg):
        raise ValueError("supervisor basis image encoding must match its JPEG or PNG extension")
    import cv2
    import numpy as np
    if cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
        raise ValueError("supervisor basis image cannot be decoded")
    checked_at = require_supervisor_lease(data)
    metadata = {"proposal_id": data["proposal_id"], "source": data["source"], "action": data["action"],
                "target_yaw_degrees": data["target_yaw_degrees"], "max_duration_s": data["max_duration_s"],
                "issued_unix_s": data["issued_unix_s"], "expires_unix_s": data["expires_unix_s"],
                "source_path": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
                "basis_frame_path": str(frame), "basis_frame_sha256": frame_hash,
                "feedback_artifact": "supervisor-feedback.json", "basis_frame_artifact": "supervisor-basis" + frame.suffix.lower(),
                "observation": data["observation"], "expected_result": data["expected_result"],
                "uncertainty": data.get("uncertainty"), "initial_validation_unix_s": checked_at,
                "initial_validation_monotonic": time.monotonic(),
                "motion_validation_unix_s": None, "motion_validation_monotonic": None}
    return {"data": data, "raw": raw, "frame_bytes": frame_bytes, "metadata": metadata}


def validate_profile(profile):
    if profile.get("backend") != "rosmaster":
        raise ValueError("only the verified Rosmaster platform is supported")
    chassis = profile.get("chassis", {})
    for name, expected in (("motor_order", [0, 1, 2, 3]), ("wheel_signs", [1, 1, 1, 1])):
        value = chassis.get(name)
        if not isinstance(value, (list, tuple)) or any(type(x) is not int for x in value) or list(value) != expected:
            raise ValueError(f"{name} must match the verified physical wheel mapping {expected}")
    if type(chassis.get("car_type")) is not int or chassis["car_type"] != 1:
        raise ValueError("only verified car_type 1 is supported")
    limit = chassis.get("command_limit")
    if type(limit) is not int or not 30 <= limit <= 100:
        raise ValueError("command_limit must be an integer in [30,100]")
    if not isinstance(chassis.get("serial_port"), str) or not chassis["serial_port"].strip():
        raise ValueError("serial_port must be a nonempty path")


def turn_targets(direction):
    if direction == "forward":
        return (30, 30, 30, 30)
    if direction not in ("left", "right"):
        raise ValueError("direction must be left, right or forward")
    return (30, 30, -30, -30) if direction == "right" else (-30, -30, 30, 30)


def require_feedback(data):
    for group in ("attitude", "encoder"):
        sample = data.get(group, {})
        age = sample.get("age_s")
        received = sample.get("received_monotonic")
        if (sample.get("valid") is not True or sample.get("stale") is not False
                or type(age) not in (int, float) or not math.isfinite(age)
                or not 0 <= age <= MAX_AGE
                or type(received) not in (int, float) or not math.isfinite(received)
                or not 0 <= time.monotonic() - received <= MAX_AGE):
            raise ValueError(f"fresh {group} telemetry required")
        values = (sample.get("value") or {}).get("radians" if group == "attitude" else "native_ticks")
        expected = 3 if group == "attitude" else 4
        if (not isinstance(values, (list, tuple)) or len(values) != expected
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
            raise ValueError(f"invalid {group} telemetry values")
    return data


def wrapped_yaw_delta(before, after):
    return (after - before + math.pi) % (2 * math.pi) - math.pi


class YawProgress:
    """Integrate new yaw samples across +/-pi without inventing progress."""
    def __init__(self, direction, target, yaw, started, sequence):
        self.sign = 1 if direction == "left" else -1
        self.target = target
        self.previous_yaw = yaw
        self.started = started
        self.sequence = sequence
        self.total_radians = 0.
        self.maximum_progress = 0.

    @property
    def directional_degrees(self):
        return self.sign * math.degrees(self.total_radians)

    def update(self, yaw, received, sequence, enforce=True):
        if sequence <= self.sequence:
            return self.directional_degrees >= self.target
        delta = wrapped_yaw_delta(self.previous_yaw, yaw)
        if abs(math.degrees(delta)) > 45.:
            raise ValueError("yaw jump exceeds 45 degrees")
        self.total_radians += delta
        self.previous_yaw = yaw
        self.sequence = sequence
        self.maximum_progress = max(self.maximum_progress, self.directional_degrees)
        if enforce:
            if self.directional_degrees < -8.:
                raise ValueError("yaw moved in wrong direction by more than 8 degrees")
            if received - self.started >= .8 and self.maximum_progress < 1.:
                raise ValueError("no yaw progress of 1 degree within 0.8 seconds")
        return self.directional_degrees >= self.target


def create_run_directory(root):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    path = Path(root).expanduser().resolve() / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


class PulseGuard:
    """Independent deadline/heartbeat stop; serialized against initial movement."""
    def __init__(self, chassis, duration, require_yaw_progress=False):
        self.chassis = chassis
        self.duration = duration
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.command_started = None
        self.command_completed = None
        self.stop_started = None
        self.stop_completed = None
        self.reason = None
        self.error = None
        self.final_pwm = None
        self.heartbeats = {}
        self.thread = None
        self.require_yaw_progress = require_yaw_progress
        self.yaw_progress_seen = False

    def start(self, targets):
        with self.lock:
            if self.reason is not None or self.command_started is not None:
                raise RuntimeError("pulse already started or cancelled")
            self.command_started = time.monotonic()
            self.heartbeats = {"camera": self.command_started, "telemetry": self.command_started}
            self.thread = threading.Thread(target=self._watch, name="stationary-turn-stop", daemon=True)
            self.thread.start()
            try:
                self.chassis.set_four_wheels(*targets)
                self.command_completed = time.monotonic()
            except BaseException:
                self.stop("command failed")
                raise

    def touch(self, source, received=None):
        with self.lock:
            self.heartbeats[source] = time.monotonic() if received is None else received

    def note_yaw_progress(self, degrees):
        with self.lock:
            self.yaw_progress_seen = self.yaw_progress_seen or degrees >= 1.

    def _watch(self):
        while not self.done.wait(.01):
            with self.lock:
                now = time.monotonic()
                if now >= self.command_started + self.duration:
                    self.stop("duration reached")
                    return
                if self.require_yaw_progress and not self.yaw_progress_seen and now - self.command_started >= .8:
                    self.stop("no yaw progress of 1 degree within 0.8 seconds")
                    return
                for source, received in self.heartbeats.items():
                    if now - received > MAX_AGE:
                        self.stop(f"{source} heartbeat expired")
                        return

    def stop(self, reason):
        with self.lock:
            if self.reason is not None:
                return
            self.reason = reason
            self.stop_started = time.monotonic()
            try:
                self.chassis.stop()
                self.final_pwm = [0, 0, 0, 0]
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.stop_completed = time.monotonic()
                self.done.set()

    def status(self):
        return {name: getattr(self, name) for name in (
            "command_started", "command_completed", "stop_started", "stop_completed",
            "reason", "error", "final_pwm")}


def open_session(profile):
    from vehicle_control.platforms.rosmaster_transport import RosmasterSerialSession
    return RosmasterSerialSession(profile["chassis"]["serial_port"], timeout=.02, write_timeout=.1, delay=.002)


def open_camera(index):
    import cv2
    camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 20)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise OSError("cannot open RGB camera")
    return camera


class CameraFrames:
    """Bounded waiting for a camera whose backend read might block indefinitely."""
    def __init__(self, camera, startup_timeout=3.):
        self.camera = camera
        self.startup_deadline = time.monotonic() + startup_timeout
        self.first_frame_pending = True
        self.frames = queue.Queue(maxsize=2)
        self.finished = threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self._capture, name="stationary-turn-camera", daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while not self.finished.is_set():
                started = time.monotonic()
                ok, frame = self.camera.read()
                received = time.monotonic()
                if not ok or frame is None or frame.shape[:2] != (480, 640):
                    raise OSError("camera failed or did not return 640x480")
                try:
                    self.frames.put((received, frame), timeout=MAX_AGE)
                except queue.Full:
                    raise OSError("video recording cannot keep up")
                # Same target cadence as CameraStream: the camera read may have
                # consumed most/all of the 20 Hz period already.
                self.finished.wait(max(0., .05 - (time.monotonic() - started)))
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
        finally:
            self.camera.release()

    def next(self):
        if self.failure:
            raise OSError(self.failure)
        timeout = max(0., self.startup_deadline - time.monotonic()) if self.first_frame_pending else MAX_AGE
        try:
            received, frame = self.frames.get(timeout=timeout)
        except queue.Empty:
            if self.first_frame_pending:
                raise OSError("camera startup timeout")
            raise OSError("camera frame timeout")
        if time.monotonic() - received > MAX_AGE:
            raise OSError("camera frame stale")
        self.first_frame_pending = False
        return received, frame

    def close(self):
        self.finished.set()
        self.thread.join(.5)
        if self.thread.is_alive():
            raise OSError("camera read blocked; capture daemon did not exit")


def run(args):
    # All authorization/parameter/profile checks precede any hardware access.
    if args.enable_motors is not True or args.confirm_motor_motion != CONFIRMATION:
        raise ValueError(f"requires --enable-motors --confirm-motor-motion {CONFIRMATION}")
    if args.direction not in ("left", "right", "forward"):
        raise ValueError("direction must be left, right or forward")
    if args.direction == "forward" and getattr(args, "supervisor_feedback", None) is None:
        raise ValueError("forward motion requires --supervisor-feedback")
    if type(args.pwm) is not int or args.pwm != 30:
        raise ValueError("this calibration uses fixed PWM 30")
    target = args.target_yaw_degrees
    if target is not None and (not math.isfinite(target) or not 0 < target <= 90):
        raise ValueError("target yaw must be finite and inside (0,90] degrees")
    max_duration = 5 if target is not None else 1
    if not math.isfinite(args.duration) or not 0 < args.duration <= max_duration:
        raise ValueError(f"duration must be finite and inside (0,{max_duration}] seconds")
    feedback = validate_supervisor_feedback(args)
    profile_path, profile = load_vehicle_profile(args.vehicle_profile)
    validate_profile(profile)
    output = create_run_directory(args.output)
    summary = {"run_id": output.name, "created_utc": datetime.now(timezone.utc).isoformat(),
               "profile": profile, "direction": args.direction, "pwm": 30,
               "requested_duration_s": args.duration, "status": "failed", "errors": [],
               "target_yaw_degrees": target, "target_reached": False,
               "cumulative_yaw_degrees": None, "directional_yaw_degrees": None,
               "before": None, "after": None, "yaw_delta_rad": None,
               "ninety_degree_turn_verified": False, "physical_stop_verified": False,
               "video_frames": 0, "final_pwm": None, "last_frame_received_monotonic": None,
               "post_stop_video_s": None}
    if feedback is not None:
        summary["control_source"] = "assistant_visual_supervision"
        summary["yolopv2_only"] = False
        summary["supervisor_feedback"] = feedback["metadata"]
    session = chassis = camera = writer = guard = tracker = None
    previous_signals = {}
    interruption_requested = threading.Event()

    def interrupted(signum, _frame):
        if interruption_requested.is_set():
            return  # Preserve the first interruption while its finally block stops.
        interruption_requested.set()
        raise InterruptedError(f"signal {signum}")

    try:
        if feedback is not None:
            (output / feedback["metadata"]["feedback_artifact"]).write_bytes(feedback["raw"])
            (output / feedback["metadata"]["basis_frame_artifact"]).write_bytes(feedback["frame_bytes"])
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signals[sig] = signal.signal(sig, interrupted)
        camera = CameraFrames(open_camera(args.camera_index))
        import cv2
        writer = cv2.VideoWriter(str(output / "raw.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 20., (640, 480))
        if not writer.isOpened():
            raise OSError("video writer failed to open")
        session = open_session(profile)
        options = profile["chassis"]
        chassis = RosmasterChassis(controller=session, owns_controller=False,
                                   command_limit=options["command_limit"],
                                   motor_order=options["motor_order"], wheel_signs=options["wheel_signs"])
        guard = PulseGuard(chassis, args.duration, require_yaw_progress=target is not None)
        with (output / "telemetry.jsonl").open("x") as telemetry_log, \
                (output / "frames.jsonl").open("x") as frames_log, \
                (output / "uart_rx.bin").open("xb") as raw_uart:

            def record(phase, required=True):
                received, frame = camera.next()
                writer.write(frame)  # Original camera frame: no overlays, rotation, crop or resizing.
                summary["video_frames"] += 1
                summary["last_frame_received_monotonic"] = received
                frames_log.write(json.dumps({"frame": summary["video_frames"], "received_monotonic": received,
                                             "phase": phase}) + "\n")
                frames_log.flush()
                raw = session.poll_available()
                raw_uart.write(raw)
                data = session.telemetry(max_age=MAX_AGE)
                telemetry_log.write(json.dumps({"monotonic": time.monotonic(), "phase": phase,
                                                "rx_bytes": len(raw), "telemetry": data}) + "\n")
                telemetry_log.flush()
                raw_uart.flush()
                if required:
                    require_feedback(data)
                if guard.command_started is not None:
                    guard.touch("camera", received)
                    if required:
                        guard.touch("telemetry", min(data[g]["received_monotonic"] for g in ("attitude", "encoder")))
                return data

            timeout = time.monotonic() + 3.
            while True:
                before = record("before", required=False)
                try:
                    require_feedback(before)
                    break
                except ValueError:
                    if time.monotonic() >= timeout:
                        raise
            summary["before"] = before
            motion_error = None
            try:
                if feedback is not None:
                    feedback["metadata"]["motion_validation_unix_s"] = require_supervisor_lease(feedback["data"])
                    feedback["metadata"]["motion_validation_monotonic"] = time.monotonic()
                guard.start(turn_targets(args.direction))
                if target is not None:
                    tracker = YawProgress(args.direction, target, before["attitude"]["value"]["radians"][2],
                                          guard.command_started, before["attitude"]["sequence"])
                while not guard.done.is_set():
                    data = record("pulse")
                    if tracker is not None:
                        attitude = data["attitude"]
                        reached = tracker.update(attitude["value"]["radians"][2],
                                                 attitude["received_monotonic"], attitude["sequence"])
                        guard.note_yaw_progress(tracker.directional_degrees)
                        if reached:
                            guard.stop("yaw target reached")
            except (Exception, KeyboardInterrupt) as exc:
                motion_error = exc
                guard.stop(f"motion failed: {exc}")
            # Record at least .3 s after stop, including for a target deadline
            # or direction failure. A broken camera/telemetry still fails closed.
            deadline = time.monotonic() + 1.
            try:
                while True:
                    after = record("after")
                    if tracker is not None:
                        attitude = after["attitude"]
                        tracker.update(attitude["value"]["radians"][2], attitude["received_monotonic"],
                                       attitude["sequence"], enforce=False)
                    post_video_s = summary["last_frame_received_monotonic"] - guard.stop_completed
                    if post_video_s >= .3 and all(after[g]["received_monotonic"] >= guard.stop_completed + .3 for g in ("attitude", "encoder")):
                        summary["after"] = after
                        summary["post_stop_video_s"] = post_video_s
                        break
                    if time.monotonic() >= deadline:
                        raise OSError("no fresh post-stop attitude/encoder pair after 0.3 seconds")
            except (Exception, KeyboardInterrupt) as exc:
                if motion_error is None:
                    motion_error = exc
            if summary["after"] is not None:
                summary["yaw_delta_rad"] = wrapped_yaw_delta(
                    before["attitude"]["value"]["radians"][2], after["attitude"]["value"]["radians"][2])
            if motion_error is not None:
                raise motion_error
            expected_reason = "yaw target reached" if target is not None else "duration reached"
            if guard.error or guard.reason != expected_reason:
                raise OSError(guard.error or f"target not reached; stop reason: {guard.reason}")
            summary["status"] = "yaw_target_reached" if target is not None else "pulse_recorded"
    except (Exception, KeyboardInterrupt) as exc:
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if guard is not None:
            guard.stop("finally")
            summary["stop"] = guard.status()
            summary["final_pwm"] = guard.final_pwm
            summary["target_reached"] = guard.reason == "yaw target reached"
            if guard.error:
                summary["errors"].append(guard.error)
        if tracker is not None:
            summary["cumulative_yaw_degrees"] = math.degrees(tracker.total_radians)
            summary["directional_yaw_degrees"] = tracker.directional_degrees
        for resource, close in ((chassis, lambda: chassis.close(stop=True)),
                                (session, lambda: session.close()),
                                (camera, lambda: camera.close()),
                                (writer, lambda: writer.release())):
            if resource is not None:
                try:
                    close()
                except Exception as exc:
                    summary["errors"].append(f"cleanup: {type(exc).__name__}: {exc}")
        if session is not None:
            summary["serial_status"] = session.get_status()
        for sig, previous in previous_signals.items():
            signal.signal(sig, previous)
        if summary["errors"]:
            summary["status"] = "failed"
        if feedback is not None:
            feedback["metadata"]["execution"] = {
                "motion_started": guard is not None and guard.command_started is not None,
                "command_started_monotonic": None if guard is None else guard.command_started,
                "command_completed_monotonic": None if guard is None else guard.command_completed,
                "stop_started_monotonic": None if guard is None else guard.stop_started,
                "stop_completed_monotonic": None if guard is None else guard.stop_completed,
                "stop_reason": None if guard is None else guard.reason,
                "status": summary["status"], "target_reached": summary["target_reached"],
                "directional_yaw_degrees": summary["directional_yaw_degrees"],
                "final_pwm": summary["final_pwm"], "errors": list(summary["errors"])}
        summary["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        summary["profile_sha256"] = hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()
        summary["artifacts_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in output.iterdir() if p.is_file()}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {output}", flush=True)
    return 0 if summary["status"] in ("pulse_recorded", "yaw_target_reached") else 1


def main():
    args = build_parser().parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
