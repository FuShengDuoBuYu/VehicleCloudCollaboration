#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import replace


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LONGTAIL_DIR = os.path.join(CURRENT_DIR, "longtail")
CONTROL_DIR = os.path.join(CURRENT_DIR, "control")

for path in (CURRENT_DIR, LONGTAIL_DIR, CONTROL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from cloud_client import DEFAULT_CLOUD_API_BASE_URL, DEFAULT_CLOUD_MODEL


DEFAULT_FRAME_DIR = os.path.join(CURRENT_DIR, "captured_frames")
LOGGER = logging.getLogger("vehicle_cloud.closed_loop")


def _now_label():
    return time.strftime("%H:%M:%S")


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RuntimeLogHandler(logging.Handler):
    def __init__(self, runtime_control):
        super().__init__(logging.INFO)
        self.runtime_control = runtime_control

    def emit(self, record):
        try:
            self.runtime_control.add_log(record.levelname, self.format(record))
        except Exception:
            return


def setup_logging(runtime_control, level=logging.INFO):
    LOGGER.handlers.clear()
    LOGGER.setLevel(level)
    LOGGER.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    runtime_handler = RuntimeLogHandler(runtime_control)
    runtime_handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_handler.setLevel(level)

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(runtime_handler)
    return LOGGER


class ClosedLoopRuntimeControl:
    def __init__(self):
        self._started = threading.Event()
        self._lock = threading.Lock()
        self._last_command = "waiting"
        self._message = "waiting for web start"
        self._logs = deque(maxlen=200)
        self._latest_frame_path = None
        self._last_detection = None
        self._last_cloud_decision = None
        self._last_cycle = None
        self._latencies_ms = {}

    def start(self):
        with self._lock:
            self._last_command = "start"
            self._message = "closed loop start requested"
        self._started.set()
        self.add_log("INFO", "closed loop start requested")

    def pause(self, message="closed loop paused"):
        with self._lock:
            self._last_command = "pause"
            self._message = message
        self._started.clear()
        self.add_log("INFO", message)

    def set_message(self, message):
        with self._lock:
            self._message = message

    def is_started(self):
        return self._started.is_set()

    def wait_until_started(self, timeout=0.2):
        return self._started.wait(timeout)

    def add_log(self, level, message):
        with self._lock:
            self._logs.append(
                {
                    "time": _now_label(),
                    "level": str(level),
                    "message": str(message),
                }
            )

    def set_latest_frame(self, frame_path):
        with self._lock:
            self._latest_frame_path = frame_path

    def latest_frame_path(self):
        with self._lock:
            return self._latest_frame_path

    def record_detection(self, result, latency_ms, frame_path):
        individual_scores = []
        for item in result.get("individual_scores", []):
            inference_time = _float_or_none(item.get("inference_time"))
            score = _float_or_none(item.get("score"))
            individual_scores.append(
                {
                    "detector": item.get("detector"),
                    "score": None if score is None else round(score, 4),
                    "latency_ms": None if inference_time is None else round(inference_time * 1000, 1),
                }
            )

        score = _float_or_none(result.get("score"))
        threshold = _float_or_none(result.get("threshold"))
        fps = _float_or_none(result.get("fps"))
        with self._lock:
            self._latest_frame_path = frame_path
            self._last_detection = {
                "is_long_tail": bool(result.get("is_long_tail")),
                "score": None if score is None else round(score, 4),
                "threshold": None if threshold is None else round(threshold, 4),
                "fps": None if fps is None else round(fps, 2),
                "latency_ms": round(float(latency_ms), 1),
                "frame_path": frame_path,
                "individual_scores": individual_scores,
            }
            self._latencies_ms["detect"] = round(float(latency_ms), 1)

    def record_cloud_decision(self, decision):
        timings = getattr(decision, "timings_ms", {}) or {}
        with self._lock:
            self._last_cloud_decision = {
                "command": decision.command,
                "action": decision.action,
                "reason": decision.reason,
                "latency_ms": round(float(decision.latency_ms), 1),
                "timings_ms": timings,
            }
            self._latencies_ms["cloud"] = round(float(decision.latency_ms), 1)

    def record_cycle(self, **latencies_ms):
        compact = {
            key: round(float(value), 1)
            for key, value in latencies_ms.items()
            if value is not None
        }
        with self._lock:
            self._last_cycle = compact
            self._latencies_ms.update(compact)

    def get_state(self):
        with self._lock:
            return {
                "started": self._started.is_set(),
                "last_command": self._last_command,
                "message": self._message,
                "latest_frame_path": self._latest_frame_path,
                "last_detection": self._last_detection,
                "last_cloud_decision": self._last_cloud_decision,
                "last_cycle": self._last_cycle,
                "latencies_ms": dict(self._latencies_ms),
                "logs": list(self._logs),
            }


def build_parser():
    parser = argparse.ArgumentParser(description="Run the car-side vehicle-cloud closed loop.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--camera-fps", type=int, default=20, help="Camera capture FPS")
    parser.add_argument("--cruise-speed", type=int, help="Override forward cruise wheel speed")
    parser.add_argument("--lane-speed", type=int, help="Override lane-change base wheel speed")
    parser.add_argument("--lane-approach-time", type=float, help="Seconds to keep moving forward before lane-change steering")
    parser.add_argument("--lane-steer-delta", type=int, help="Lane-change wheel-speed steering delta; lower is gentler")
    parser.add_argument("--lane-speed-scale", type=float, help="Lane-change forward speed scale; higher keeps the car moving while steering")
    parser.add_argument("--lane-min-turn-speed-scale", type=float, help="Minimum inner-wheel speed scale during lane-change steering")
    parser.add_argument("--lane-transition-time", type=float, help="Seconds for each lane-change speed ramp; higher is smoother")
    parser.add_argument("--lane-turn-time", type=float, help="Seconds to hold the outward steering arc")
    parser.add_argument("--lane-return-time", type=float, help="Seconds to hold the return steering arc")
    detection_group = parser.add_mutually_exclusive_group()
    detection_group.add_argument("--stop-for-detection", dest="stop_for_detection", action="store_true", default=True, help="Stop the vehicle while running each long-tail detection pass")
    detection_group.add_argument("--no-stop-for-detection", dest="stop_for_detection", action="store_false", help="Keep moving while running long-tail detection")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between long-tail checks")
    parser.add_argument("--frame-dir", default=DEFAULT_FRAME_DIR, help="Directory for latest analysis frame")
    parser.add_argument("--action-cooldown", type=float, default=5.0, help="Seconds to keep a cloud action before changing")
    parser.add_argument("--startup-frame-timeout", type=float, default=10.0, help="Seconds to wait for the first camera frame before refusing to drive")
    parser.add_argument("--frame-stale-timeout", type=float, default=2.0, help="Stop the vehicle if the latest camera frame is older than this many seconds")
    parser.add_argument("--cloud-mode", choices=["cloud", "none"], default="cloud", help="Cloud decision mode")
    parser.add_argument(
        "--cloud-url",
        default=None,
        help=f"Cloud API base URL; defaults to CAR_CLOUD_API_BASE_URL ({DEFAULT_CLOUD_API_BASE_URL or 'unset'})",
    )
    parser.add_argument(
        "--cloud-model",
        default=None,
        help=f"Cloud model name; defaults to CAR_CLOUD_MODEL ({DEFAULT_CLOUD_MODEL})",
    )
    parser.add_argument("--cloud-timeout", type=float, default=None, help="Cloud API timeout in seconds")
    parser.add_argument("--web-host", default=None, help="Web UI bind host; defaults to vehicle_control settings")
    parser.add_argument("--web-port", type=int, default=None, help="Web UI port; defaults to vehicle_control settings")
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument("--warmup-detector", dest="warmup_detector", action="store_true", default=True, help="Run one synthetic long-tail detection pass before the vehicle loop starts")
    warmup_group.add_argument("--no-warmup-detector", dest="warmup_detector", action="store_false", help="Skip detector warmup before the vehicle loop starts")

    web_group = parser.add_mutually_exclusive_group()
    web_group.add_argument("--web", dest="web", action="store_true", default=True, help="Start vehicle web UI")
    web_group.add_argument("--no-web", dest="web", action="store_false", help="Disable vehicle web UI")
    parser.add_argument("--start-immediately", action="store_true", help="Start the closed loop without waiting for the web start button")
    return parser


def start_web_server(controller, camera, runtime_control=None, host=None, port=None):
    from vehicle_control.settings import SERVER_CONFIG
    from vehicle_control.web import VehicleControlServer

    server_config = SERVER_CONFIG
    if host is not None or port is not None:
        server_config = replace(
            SERVER_CONFIG,
            host=host if host is not None else SERVER_CONFIG.host,
            port=port if port is not None else SERVER_CONFIG.port,
        )

    server = VehicleControlServer(
        controller=controller,
        camera=camera,
        server_config=server_config,
        runtime_control=runtime_control,
    )

    def serve():
        try:
            server.start()
        except RuntimeError as exc:
            LOGGER.warning("Web UI not started: %s", exc)
        except Exception as exc:
            LOGGER.exception("Web UI stopped unexpectedly")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server


def should_hold_action(last_action_time, cooldown):
    return last_action_time > 0 and (time.monotonic() - last_action_time) < cooldown


def save_frame(frame, frame_dir):
    import cv2

    os.makedirs(frame_dir, exist_ok=True)
    path = os.path.join(frame_dir, "latest_frame.jpg")
    cv2.imwrite(path, frame)
    return path


def warmup_longtail_classifier(classifier):
    try:
        import cv2
        import numpy as np

        path = os.path.join(tempfile.gettempdir(), "car_longtail_warmup.jpg")
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.imwrite(path, frame)
        start = time.monotonic()
        result = classifier.predict(path)
        LOGGER.info(
            "Detector warmup complete: "
            f"detect={(time.monotonic() - start) * 1000:.0f}ms, "
            f"score={result['score']:.3f}"
        )
    except Exception as exc:
        LOGGER.warning("Detector warmup skipped: %s", exc)


def wait_for_fresh_frame(camera, timeout, stale_timeout):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        frame = camera.get_frame()
        frame_age = camera.get_frame_age()
        if frame is not None and frame_age is not None and frame_age <= stale_timeout:
            return frame
        time.sleep(0.1)
    return None


def wait_for_vehicle_action(controller, runtime_control, action):
    action_start = time.monotonic()
    runtime_control.set_message(f"executing {action}")
    while controller.is_action_running():
        if not runtime_control.is_started():
            controller.execute("stop")
            return False, (time.monotonic() - action_start) * 1000
        controller.wait_for_current_action(timeout=0.1)
    return True, (time.monotonic() - action_start) * 1000


def run_active_loop(args, controller, camera, classifier, cloud_client, runtime_control):
    LOGGER.info("Closed loop started")
    runtime_control.set_message("closed loop running")
    last_check = 0.0
    last_cloud_action = 0.0
    waiting_for_camera = False

    controller.execute("forward")
    while runtime_control.is_started():
        frame = camera.get_frame()
        frame_age = camera.get_frame_age()
        frame_stale = frame_age is None or frame_age > args.frame_stale_timeout
        if frame is None or frame_stale:
            if not waiting_for_camera:
                LOGGER.warning("Camera frame missing or stale; stopping until fresh frames return")
                runtime_control.set_message("waiting for fresh camera frame")
                controller.execute("stop")
                waiting_for_camera = True
            time.sleep(0.1)
            continue

        if waiting_for_camera:
            LOGGER.info("Camera frame recovered; resuming forward motion")
            runtime_control.set_message("closed loop running")
            controller.execute("forward")
            waiting_for_camera = False

        now = time.monotonic()
        if now - last_check < args.interval:
            time.sleep(0.05)
            continue

        if args.stop_for_detection:
            controller.execute("stop")

        cycle_start = time.monotonic()
        save_start = time.monotonic()
        frame_path = save_frame(frame, args.frame_dir)
        save_latency_ms = (time.monotonic() - save_start) * 1000
        runtime_control.set_latest_frame(frame_path)
        detect_start = time.monotonic()
        result = classifier.predict(frame_path)
        detect_latency_ms = (time.monotonic() - detect_start) * 1000
        status = "LONG-TAIL" if result["is_long_tail"] else "NORMAL"
        runtime_control.record_detection(result, detect_latency_ms, frame_path)
        LOGGER.info(
            f"[{status}] score={result['score']:.3f}, "
            f"save={save_latency_ms:.0f}ms, detect={detect_latency_ms:.0f}ms, "
            f"fps={result['fps']:.2f}, frame={frame_path}"
        )

        if not result["is_long_tail"]:
            controller.execute("forward")
            runtime_control.record_cycle(
                save=save_latency_ms,
                detect=detect_latency_ms,
                cloud=0,
                vehicle_action=0,
                total=(time.monotonic() - cycle_start) * 1000,
            )
            last_check = now
            continue

        if should_hold_action(last_cloud_action, args.action_cooldown):
            LOGGER.info("Skipping repeated long-tail trigger during action cooldown")
            runtime_control.set_message("cooldown after cloud action")
            controller.execute("forward")
            runtime_control.record_cycle(
                save=save_latency_ms,
                detect=detect_latency_ms,
                cloud=0,
                vehicle_action=0,
                total=(time.monotonic() - cycle_start) * 1000,
            )
            last_check = now
            continue

        LOGGER.info("Long-tail detected; stopping before cloud decision")
        runtime_control.set_message("long-tail detected; waiting for cloud")
        controller.execute("stop")

        if cloud_client is None:
            LOGGER.warning("Long-tail triggered without cloud client; stopping")
            runtime_control.set_message("long-tail triggered without cloud client")
            controller.execute("stop")
            runtime_control.record_cycle(
                save=save_latency_ms,
                detect=detect_latency_ms,
                cloud=0,
                vehicle_action=0,
                total=(time.monotonic() - cycle_start) * 1000,
            )
            last_cloud_action = time.monotonic()
            last_check = now
            continue

        try:
            decision = cloud_client.request_decision(frame_path, result)
        except Exception as exc:
            LOGGER.exception("Cloud API request failed; stopping")
            runtime_control.set_message("cloud API request failed")
            controller.execute("stop")
            runtime_control.record_cycle(
                save=save_latency_ms,
                detect=detect_latency_ms,
                cloud=0,
                vehicle_action=0,
                total=(time.monotonic() - cycle_start) * 1000,
            )
            last_cloud_action = time.monotonic()
            last_check = now
            continue

        runtime_control.record_cloud_decision(decision)
        LOGGER.info(
            "Cloud decision: "
            f"command={decision.command}, action={decision.action}, "
            f"cloud={decision.latency_ms:.0f}ms, "
            f"cycle={(time.monotonic() - cycle_start) * 1000:.0f}ms, reason={decision.reason}"
        )
        controller.execute(decision.action)
        action_latency_ms = 0
        if decision.action.startswith("lane-"):
            completed, action_latency_ms = wait_for_vehicle_action(controller, runtime_control, decision.action)
            LOGGER.info(
                f"Vehicle action: action={decision.action}, "
                f"completed={completed}, duration={action_latency_ms:.0f}ms"
            )
            if not completed:
                break

        runtime_control.set_message(f"cloud action completed: {decision.action}")
        runtime_control.record_cycle(
            save=save_latency_ms,
            detect=detect_latency_ms,
            cloud=decision.latency_ms,
            vehicle_action=action_latency_ms,
            total=(time.monotonic() - cycle_start) * 1000,
        )
        last_cloud_action = time.monotonic()
        last_check = time.monotonic()

    controller.execute("stop")
    runtime_control.set_message("waiting for web start")
    LOGGER.info("Closed loop paused; vehicle stopped")


def run_closed_loop(args):
    from cloud_client import CloudClient
    from classifier import LongTailClassifier
    from env_config import load_longtail_config_from_env
    from vehicle_control.camera import CameraStream
    from vehicle_control.controller import VehicleController
    from vehicle_control.settings import CAMERA_CONFIG, CRUISE_CONFIG, LANE_CHANGE_CONFIG, SERVER_CONFIG

    runtime_control = ClosedLoopRuntimeControl()
    setup_logging(runtime_control)

    config = load_longtail_config_from_env()

    LOGGER.info("Initializing long-tail classifier")
    classifier = LongTailClassifier(config)
    LOGGER.info("Classifier ready with %d detector(s)", len(classifier.detectors))
    if args.warmup_detector:
        warmup_longtail_classifier(classifier)

    camera_config = replace(
        CAMERA_CONFIG,
        camera_index=args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    camera = CameraStream(config=camera_config)
    cruise_config = CRUISE_CONFIG
    lane_change_config = LANE_CHANGE_CONFIG
    if args.cruise_speed is not None:
        cruise_config = replace(cruise_config, speed=args.cruise_speed)
    if args.lane_speed is not None:
        lane_change_config = replace(lane_change_config, speed=args.lane_speed)
    if args.lane_approach_time is not None:
        lane_change_config = replace(lane_change_config, approach_time=args.lane_approach_time)
    if args.lane_steer_delta is not None:
        lane_change_config = replace(lane_change_config, steer_delta=args.lane_steer_delta)
    if args.lane_speed_scale is not None:
        lane_change_config = replace(lane_change_config, lane_speed_scale=args.lane_speed_scale)
    if args.lane_min_turn_speed_scale is not None:
        lane_change_config = replace(lane_change_config, min_turn_speed_scale=args.lane_min_turn_speed_scale)
    if args.lane_transition_time is not None:
        lane_change_config = replace(lane_change_config, lane_transition_time=args.lane_transition_time)
    if args.lane_turn_time is not None:
        lane_change_config = replace(lane_change_config, turn_time=args.lane_turn_time)
    if args.lane_return_time is not None:
        lane_change_config = replace(lane_change_config, return_time=args.lane_return_time)
    controller = VehicleController(cruise_config=cruise_config, lane_change_config=lane_change_config)
    if args.start_immediately or not args.web:
        runtime_control.start()

    cloud_client = None
    if args.cloud_mode == "cloud":
        cloud_client = CloudClient(url=args.cloud_url, timeout=args.cloud_timeout, model=args.cloud_model)

    server = None
    camera.start()

    try:
        if args.web:
            server = start_web_server(
                controller,
                camera,
                runtime_control=runtime_control,
                host=args.web_host,
                port=args.web_port,
            )

        LOGGER.info("Closed loop starting")
        LOGGER.info("Camera index: %s", args.camera_index)
        LOGGER.info("Frame interval: %.3fs", args.interval)
        if cloud_client:
            LOGGER.info("Cloud API: %s", cloud_client.base_url)
            LOGGER.info("Cloud model: %s", cloud_client.model)
        if args.web:
            host = args.web_host if args.web_host is not None else SERVER_CONFIG.host
            port = args.web_port if args.web_port is not None else SERVER_CONFIG.port
            LOGGER.info("Vehicle web UI: http://%s:%s", host, port)

        while True:
            if not runtime_control.wait_until_started(timeout=0.2):
                continue

            LOGGER.info("Start signal received; checking camera frame")
            first_frame = wait_for_fresh_frame(camera, args.startup_frame_timeout, args.frame_stale_timeout)
            if first_frame is None:
                LOGGER.warning("No fresh camera frame available; staying idle")
                controller.execute("stop")
                runtime_control.pause("no fresh camera frame")
                continue

            run_active_loop(args, controller, camera, classifier, cloud_client, runtime_control)

    except KeyboardInterrupt:
        LOGGER.info("Closed loop interrupted")
    finally:
        LOGGER.info("Stopping vehicle and camera")
        controller.stop()
        if server is not None:
            server.shutdown()
        else:
            camera.stop()


def main():
    args = build_parser().parse_args()
    run_closed_loop(args)


if __name__ == "__main__":
    main()
