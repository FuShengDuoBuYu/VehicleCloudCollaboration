#!/usr/bin/env python3

import argparse
import os
import sys
import threading
import time
from dataclasses import replace

import yaml


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LONGTAIL_DIR = os.path.join(CURRENT_DIR, "longtail")
CONTROL_DIR = os.path.join(CURRENT_DIR, "control")

for path in (CURRENT_DIR, LONGTAIL_DIR, CONTROL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from cloud_client.mock_client import DEFAULT_MOCK_CHAT_URL


DEFAULT_CONFIG = os.path.join(LONGTAIL_DIR, "config.yaml")
DEFAULT_FRAME_DIR = os.path.join(CURRENT_DIR, "captured_frames")


class ClosedLoopRuntimeControl:
    def __init__(self):
        self._started = threading.Event()
        self._lock = threading.Lock()
        self._last_command = "waiting"
        self._message = "waiting for web start"

    def start(self):
        with self._lock:
            self._last_command = "start"
            self._message = "closed loop start requested"
        self._started.set()

    def pause(self, message="closed loop paused"):
        with self._lock:
            self._last_command = "pause"
            self._message = message
        self._started.clear()

    def set_message(self, message):
        with self._lock:
            self._message = message

    def is_started(self):
        return self._started.is_set()

    def wait_until_started(self, timeout=0.2):
        return self._started.wait(timeout)

    def get_state(self):
        with self._lock:
            return {
                "started": self._started.is_set(),
                "last_command": self._last_command,
                "message": self._message,
            }


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_parser():
    parser = argparse.ArgumentParser(description="Run the car-side vehicle-cloud closed loop.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Long-tail detector config path")
    parser.add_argument("--threshold", type=float, help="Override long-tail threshold")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--camera-fps", type=int, default=20, help="Camera capture FPS")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between long-tail checks")
    parser.add_argument("--frame-dir", default=DEFAULT_FRAME_DIR, help="Directory for latest analysis frame")
    parser.add_argument("--action-cooldown", type=float, default=5.0, help="Seconds to keep a cloud action before changing")
    parser.add_argument("--startup-frame-timeout", type=float, default=10.0, help="Seconds to wait for the first camera frame before refusing to drive")
    parser.add_argument("--frame-stale-timeout", type=float, default=2.0, help="Stop the vehicle if the latest camera frame is older than this many seconds")
    parser.add_argument("--cloud-mode", choices=["mock", "none"], default="mock", help="Cloud decision mode")
    parser.add_argument("--cloud-mock-url", default=DEFAULT_MOCK_CHAT_URL, help="Mock LLM chat endpoint")
    parser.add_argument("--cloud-timeout", type=float, default=30.0, help="Cloud mock timeout in seconds")
    parser.add_argument("--web-host", default=None, help="Web UI bind host; defaults to vehicle_control settings")
    parser.add_argument("--web-port", type=int, default=None, help="Web UI port; defaults to vehicle_control settings")

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
            print(f"Web UI not started: {exc}")
        except Exception as exc:
            print(f"Web UI stopped unexpectedly: {exc}")

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


def wait_for_fresh_frame(camera, timeout, stale_timeout):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        frame = camera.get_frame()
        frame_age = camera.get_frame_age()
        if frame is not None and frame_age is not None and frame_age <= stale_timeout:
            return frame
        time.sleep(0.1)
    return None


def run_active_loop(args, controller, camera, classifier, cloud_client, runtime_control):
    print("Closed loop started")
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
                print("Camera frame missing or stale; stopping until fresh frames return")
                runtime_control.set_message("waiting for fresh camera frame")
                controller.execute("stop")
                waiting_for_camera = True
            time.sleep(0.1)
            continue

        if waiting_for_camera:
            print("Camera frame recovered; resuming forward motion")
            runtime_control.set_message("closed loop running")
            controller.execute("forward")
            waiting_for_camera = False

        now = time.monotonic()
        if now - last_check < args.interval:
            time.sleep(0.05)
            continue

        frame_path = save_frame(frame, args.frame_dir)
        result = classifier.predict(frame_path)
        status = "LONG-TAIL" if result["is_long_tail"] else "NORMAL"
        print(f"[{status}] score={result['score']:.3f}, fps={result['fps']:.2f}, frame={frame_path}")

        if should_hold_action(last_cloud_action, args.action_cooldown):
            print("Holding current cloud action during cooldown")
            last_check = now
            continue

        if not result["is_long_tail"]:
            controller.execute("forward")
            last_check = now
            continue

        if cloud_client is None:
            print("Long-tail triggered without cloud client; stopping")
            runtime_control.set_message("long-tail triggered without cloud client")
            controller.execute("stop")
            last_cloud_action = time.monotonic()
            last_check = now
            continue

        try:
            decision = cloud_client.request_decision(frame_path, result)
        except Exception as exc:
            print(f"Cloud mock request failed: {exc}; stopping")
            runtime_control.set_message("cloud mock request failed")
            controller.execute("stop")
            last_cloud_action = time.monotonic()
            last_check = now
            continue

        print(
            "Cloud decision: "
            f"command={decision.command}, action={decision.action}, "
            f"latency={decision.latency_ms:.0f}ms, reason={decision.reason}"
        )
        controller.execute(decision.action)
        runtime_control.set_message(f"cloud action: {decision.action}")
        last_cloud_action = time.monotonic()
        last_check = now

    controller.execute("stop")
    runtime_control.set_message("waiting for web start")
    print("Closed loop paused; vehicle stopped")


def run_closed_loop(args):
    from cloud_client.mock_client import CloudMockClient
    from classifier import LongTailClassifier
    from vehicle_control.camera import CameraStream
    from vehicle_control.controller import VehicleController
    from vehicle_control.settings import CAMERA_CONFIG, SERVER_CONFIG

    config = load_config(args.config)
    if args.threshold is not None:
        config["threshold"] = args.threshold

    print("Initializing long-tail classifier...")
    classifier = LongTailClassifier(config)
    print(f"Classifier ready with {len(classifier.detectors)} detector(s)")

    camera_config = replace(
        CAMERA_CONFIG,
        camera_index=args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    camera = CameraStream(config=camera_config)
    controller = VehicleController()
    runtime_control = ClosedLoopRuntimeControl()
    if args.start_immediately or not args.web:
        runtime_control.start()

    cloud_client = None
    if args.cloud_mode == "mock":
        cloud_client = CloudMockClient(url=args.cloud_mock_url, timeout=args.cloud_timeout, force_left=True)

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

        print("\nClosed loop starting")
        print(f"Camera index: {args.camera_index}")
        print(f"Frame interval: {args.interval}s")
        if cloud_client:
            print(f"Cloud mock: {args.cloud_mock_url}")
        if args.web:
            host = args.web_host if args.web_host is not None else SERVER_CONFIG.host
            port = args.web_port if args.web_port is not None else SERVER_CONFIG.port
            print(f"Vehicle web UI: http://{host}:{port}")

        while True:
            if not runtime_control.wait_until_started(timeout=0.2):
                continue

            print("Start signal received; checking camera frame")
            first_frame = wait_for_fresh_frame(camera, args.startup_frame_timeout, args.frame_stale_timeout)
            if first_frame is None:
                print("No fresh camera frame available; staying idle")
                controller.execute("stop")
                runtime_control.pause("no fresh camera frame")
                continue

            run_active_loop(args, controller, camera, classifier, cloud_client, runtime_control)

    except KeyboardInterrupt:
        print("\nClosed loop interrupted")
    finally:
        print("Stopping vehicle and camera")
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
