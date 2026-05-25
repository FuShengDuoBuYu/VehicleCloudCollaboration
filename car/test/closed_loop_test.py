#!/usr/bin/env python3

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional


TEST_DIR = Path(__file__).resolve().parent
CAR_DIR = TEST_DIR.parent
LONGTAIL_DIR = CAR_DIR / "longtail"
CONTROL_DIR = CAR_DIR / "control"

for path in (CAR_DIR, LONGTAIL_DIR, CONTROL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cloud_client import CloudMockClient
from vehicle_control.controller import VehicleController


class ClosedLoopTestError(RuntimeError):
    pass


@dataclass
class StageReport:
    name: str
    ok: bool
    latency_ms: float
    expected_input: Dict[str, Any]
    actual_output: Dict[str, Any]
    error: Optional[str] = None


class FakeChassis:
    def __init__(self):
        self.commands: List[Dict[str, Any]] = []

    def ramp_to(self, left_speed, right_speed, duration, stop_event=None):
        self.commands.append(
            {
                "op": "ramp_to",
                "left_speed": int(left_speed),
                "right_speed": int(right_speed),
                "duration_s": float(duration),
            }
        )

    def hold(self, duration, stop_event=None):
        self.commands.append({"op": "hold", "duration_s": float(duration)})
        return not (stop_event and stop_event.is_set())

    def stop(self):
        self.commands.append({"op": "stop"})


def default_image_path() -> Path:
    for name in ("test_image.jpg", "test_image.jpeg", "test_image.png", "test_image.bmp"):
        path = TEST_DIR / name
        if path.exists():
            return path
    return TEST_DIR / "test_image.jpg"


def load_image(image_path: Path):
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ClosedLoopTestError(f"failed to read image: {image_path}")

    height, width = frame.shape[:2]
    return frame, {
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "channels": int(frame.shape[2]) if len(frame.shape) == 3 else 1,
        "file_size_bytes": image_path.stat().st_size,
    }


def run_longtail_detector(image_path: Path):
    from classifier import LongTailClassifier
    from env_config import load_longtail_config_from_env

    start = time.monotonic()
    config = load_longtail_config_from_env()
    classifier = LongTailClassifier(config)
    if not classifier.detectors:
        raise ClosedLoopTestError("LongTailClassifier initialized zero detectors")

    result = classifier.predict(str(image_path))
    result["detector"] = "LongTailClassifier"
    result["detector_count"] = len(classifier.detectors)
    result["detector_names"] = [detector.__class__.__name__ for detector, _ in classifier.detectors]
    result["config_source"] = "environment"
    result["image_path"] = str(image_path)
    result["end_to_end_detector_time"] = time.monotonic() - start
    return result


def make_chat_handler(captured_request: Dict[str, Any]):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {"raw_body": body}

            captured_request["path"] = self.path
            captured_request["payload"] = payload

            response = {
                "command": "left",
                "action": "lane-left",
                "reason": "offline closed-loop test: traffic cone triggers left lane change",
            }
            data = json.dumps(response, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            return

    return Handler


def start_local_cloud_mock(captured_request: Dict[str, Any]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_chat_handler(captured_request))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/chat"


def request_cloud_decision(image_path: Path, detection_result: Dict[str, Any], timeout: float, backend: str):
    if backend == "inprocess":
        return request_inprocess_cloud_decision(image_path, detection_result, timeout)
    return request_local_http_cloud_decision(image_path, detection_result, timeout)


def request_inprocess_cloud_decision(image_path: Path, detection_result: Dict[str, Any], timeout: float):
    client = CloudMockClient(url="inprocess://offline-mock", timeout=timeout, force_left=True)
    payload = client.build_payload(str(image_path), detection_result)
    body = json.dumps(
        {
            "command": "left",
            "action": "lane-left",
            "reason": "offline in-process mock: traffic cone triggers left lane change",
        },
        ensure_ascii=False,
    )

    start = time.monotonic()
    decision = client.parse_response(body, latency_ms=(time.monotonic() - start) * 1000)

    return {
        "command": decision.command,
        "action": decision.action,
        "reason": decision.reason,
        "latency_ms": round(float(decision.latency_ms), 3),
        "backend": "inprocess",
        "mock_url": client.url,
        "request_path": None,
        "request_has_prompt": bool(payload.get("prompt")),
        "request_json_mode": payload.get("json"),
        "request_prompt_preview": str(payload.get("prompt", ""))[:240],
        "raw_response": decision.raw_response,
    }


def request_local_http_cloud_decision(image_path: Path, detection_result: Dict[str, Any], timeout: float):
    captured_request: Dict[str, Any] = {}
    server, url = start_local_cloud_mock(captured_request)
    try:
        client = CloudMockClient(url=url, timeout=timeout, force_left=True)
        decision = client.request_decision(str(image_path), detection_result)
    finally:
        server.shutdown()
        server.server_close()

    payload = captured_request.get("payload", {})
    return {
        "command": decision.command,
        "action": decision.action,
        "reason": decision.reason,
        "latency_ms": round(float(decision.latency_ms), 3),
        "backend": "local-http",
        "mock_url": url,
        "request_path": captured_request.get("path"),
        "request_has_prompt": bool(payload.get("prompt")),
        "request_json_mode": payload.get("json"),
        "request_prompt_preview": str(payload.get("prompt", ""))[:240],
        "raw_response": decision.raw_response,
    }


def execute_vehicle_action(action: str, timeout: float):
    chassis = FakeChassis()
    controller = VehicleController(chassis=chassis)

    try:
        controller.execute(action)
        thread = getattr(controller, "_action_thread", None)
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise ClosedLoopTestError(f"vehicle action thread did not finish within {timeout}s")

        state = controller.get_state()
        ramp_commands = [cmd for cmd in chassis.commands if cmd["op"] == "ramp_to"]
        left_turn_arcs = [
            cmd for cmd in ramp_commands if cmd["left_speed"] < cmd["right_speed"]
        ]
        approach_holds = [
            cmd for cmd in chassis.commands
            if cmd["op"] == "hold" and cmd["duration_s"] == controller.lane_change_config.approach_time
        ]
        return {
            "action_requested": action,
            "state_after_action": state,
            "command_count": len(chassis.commands),
            "ramp_command_count": len(ramp_commands),
            "contains_left_turn_arc": bool(left_turn_arcs),
            "contains_approach_hold": bool(approach_holds),
            "first_commands": chassis.commands[:12],
        }
    finally:
        controller.stop()


def run_stage(name: str, expected_input: Dict[str, Any], func, reports: List[StageReport]):
    start = time.monotonic()
    try:
        output = func()
        report = StageReport(
            name=name,
            ok=True,
            latency_ms=round((time.monotonic() - start) * 1000, 3),
            expected_input=expected_input,
            actual_output=output,
        )
        reports.append(report)
        return output
    except Exception as exc:
        report = StageReport(
            name=name,
            ok=False,
            latency_ms=round((time.monotonic() - start) * 1000, 3),
            expected_input=expected_input,
            actual_output={},
            error=str(exc),
        )
        reports.append(report)
        raise


def assert_stage(condition: bool, message: str):
    if not condition:
        raise ClosedLoopTestError(message)


def build_parser():
    parser = argparse.ArgumentParser(description="Run an offline vehicle-cloud closed-loop data test.")
    parser.add_argument("--image", type=Path, default=default_image_path(), help="Test image path")
    parser.add_argument(
        "--cloud-backend",
        choices=["inprocess", "local-http"],
        default="inprocess",
        help="Cloud mock backend. inprocess is fully offline; local-http also tests HTTP serialization.",
    )
    parser.add_argument("--cloud-timeout", type=float, default=5.0, help="Local cloud mock request timeout")
    parser.add_argument("--action-timeout", type=float, default=2.0, help="Fake vehicle action timeout")
    parser.add_argument(
        "--report",
        type=Path,
        default=TEST_DIR / "closed_loop_report.json",
        help="JSON report output path",
    )
    return parser


def write_report(path: Path, reports: List[StageReport], ok: bool, total_latency_ms: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": ok,
        "total_latency_ms": round(total_latency_ms, 3),
        "stages": [asdict(report) for report in reports],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_summary(reports: List[StageReport], total_latency_ms: float, report_path: Path):
    print("\nClosed-loop data test")
    print("=" * 88)
    print(f"{'stage':<22} {'ok':<6} {'latency(ms)':>12}  output")
    print("-" * 88)
    for report in reports:
        status = "PASS" if report.ok else "FAIL"
        output = report.actual_output
        if report.name == "longtail_detection":
            summary = (
                f"is_long_tail={output.get('is_long_tail')}, "
                f"score={output.get('score')}, "
                f"detectors={output.get('detector_names')}"
            )
        elif report.name == "cloud_decision":
            summary = f"command={output.get('command')}, action={output.get('action')}"
        elif report.name == "vehicle_control":
            summary = (
                f"action={output.get('action_requested')}, "
                f"left_arc={output.get('contains_left_turn_arc')}, "
                f"commands={output.get('command_count')}"
            )
        else:
            summary = json.dumps(output, ensure_ascii=False)[:120]
        print(f"{report.name:<22} {status:<6} {report.latency_ms:>12.3f}  {summary}")
    print("-" * 88)
    print(f"{'total':<22} {'':<6} {total_latency_ms:>12.3f}")
    print(f"Report: {report_path}")


def main():
    args = build_parser().parse_args()
    reports: List[StageReport] = []
    context: Dict[str, Any] = {}
    total_start = time.monotonic()
    ok = False

    try:
        image_info = run_stage(
            "image_input",
            {"image_path": str(args.image), "expected": "readable image file"},
            lambda: _stage_image_input(args, context),
            reports,
        )

        detection = run_stage(
            "longtail_detection",
            {"input": image_info, "expected": "LongTailClassifier returns is_long_tail=true"},
            lambda: _stage_detection(args, context),
            reports,
        )
        assert_stage(detection["is_long_tail"], "expected test image to trigger LongTailClassifier")

        decision = run_stage(
            "cloud_decision",
            {"detection": _compact_detection(detection), "expected": "command=left, action=lane-left"},
            lambda: request_cloud_decision(args.image, detection, args.cloud_timeout, args.cloud_backend),
            reports,
        )
        assert_stage(decision["command"] == "left", f"expected cloud command left, got {decision['command']}")
        assert_stage(decision["action"] == "lane-left", f"expected action lane-left, got {decision['action']}")

        vehicle = run_stage(
            "vehicle_control",
            {"action": decision["action"], "expected": "VehicleController accepts lane-left without hardware"},
            lambda: execute_vehicle_action(decision["action"], args.action_timeout),
            reports,
        )
        assert_stage(vehicle["action_requested"] == "lane-left", "vehicle action was not lane-left")
        assert_stage(vehicle["contains_approach_hold"], "lane-left sequence did not include forward approach hold")
        assert_stage(vehicle["contains_left_turn_arc"], "vehicle command sequence did not contain a left-turn arc")
        assert_stage(vehicle["ramp_command_count"] >= 5, "lane-left sequence produced too few ramp commands")

        ok = True
    finally:
        total_latency_ms = (time.monotonic() - total_start) * 1000
        write_report(args.report, reports, ok, total_latency_ms)
        print_summary(reports, total_latency_ms, args.report)

    print("\nPASS: data loop image -> detection -> cloud decision -> vehicle action is normal.")


def _stage_image_input(args, context):
    assert_stage(args.image.exists(), f"image does not exist: {args.image}")
    _frame, metadata = load_image(args.image)
    return metadata


def _stage_detection(args, context):
    return run_longtail_detector(args.image)


def _compact_detection(detection: Dict[str, Any]):
    return {
        "is_long_tail": detection.get("is_long_tail"),
        "score": detection.get("score"),
        "threshold": detection.get("threshold"),
        "detector": detection.get("detector"),
        "object": detection.get("object"),
    }


if __name__ == "__main__":
    main()
