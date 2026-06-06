#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


TEST_DIR = Path(__file__).resolve().parent
CAR_DIR = TEST_DIR.parent
LONGTAIL_DIR = CAR_DIR / "longtail"
CONTROL_DIR = CAR_DIR / "control"

for path in (CAR_DIR, LONGTAIL_DIR, CONTROL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cloud_client import CloudClient
from vehicle_control.controller import VehicleController


LOGGER = logging.getLogger("vehicle_cloud.closed_loop_test")
ALLOWED_COMMANDS = {"left", "right"}
ALLOWED_ACTIONS = {"lane-left", "lane-right"}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


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
    config_start = time.monotonic()
    config = load_longtail_config_from_env()
    config_latency_ms = (time.monotonic() - config_start) * 1000
    init_start = time.monotonic()
    classifier = LongTailClassifier(config)
    init_latency_ms = (time.monotonic() - init_start) * 1000
    if not classifier.detectors:
        raise ClosedLoopTestError("LongTailClassifier initialized zero detectors")

    predict_start = time.monotonic()
    result = classifier.predict(str(image_path))
    predict_latency_ms = (time.monotonic() - predict_start) * 1000
    result["detector"] = "LongTailClassifier"
    result["detector_count"] = len(classifier.detectors)
    result["detector_names"] = [detector.__class__.__name__ for detector, _ in classifier.detectors]
    result["config_source"] = "environment"
    result["image_path"] = str(image_path)
    result["end_to_end_detector_time"] = time.monotonic() - start
    result["latencies_ms"] = {
        "config_load": round(config_latency_ms, 3),
        "classifier_init": round(init_latency_ms, 3),
        "predict": round(predict_latency_ms, 3),
        "end_to_end": round((time.monotonic() - start) * 1000, 3),
    }
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

            decision_content = json.dumps(
                {
                    "command": "left",
                    "action": "lane-left",
                    "reason": "offline closed-loop test: traffic cone triggers left lane change",
                },
                ensure_ascii=False,
            )
            response = {
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "offline-test"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": decision_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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


def start_local_cloud_api(captured_request: Dict[str, Any]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_chat_handler(captured_request))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def request_cloud_decision(
    image_path: Path,
    detection_result: Dict[str, Any],
    timeout: float,
    backend: str,
    cloud_url: Optional[str],
    cloud_model: Optional[str],
):
    if backend == "cloud":
        return request_public_cloud_decision(image_path, detection_result, timeout, cloud_url, cloud_model)
    if backend == "inprocess":
        return request_inprocess_cloud_decision(image_path, detection_result, timeout)
    return request_local_http_cloud_decision(image_path, detection_result, timeout)


def _offline_chat_completion(content: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-offline",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "offline-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(content, ensure_ascii=False)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
        ensure_ascii=False,
    )


def _cloud_request_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = payload.get("messages", [])
    content_parts: List[Dict[str, Any]] = []
    if len(messages) >= 2 and isinstance(messages[1], dict):
        content = messages[1].get("content", [])
        if isinstance(content, list):
            content_parts = [part for part in content if isinstance(part, dict)]

    text_part = next((part for part in content_parts if part.get("type") == "text"), {})
    return {
        "request_has_messages": bool(messages),
        "request_has_image": any(part.get("type") in {"image_url", "input_image"} for part in content_parts),
        "request_response_format": payload.get("response_format"),
        "request_prompt_preview": str(text_part.get("text", ""))[:240],
    }


def _redact_public_api_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.path:
        return "<configured-cloud>"
    return f"{parsed.scheme}://<configured-cloud>{parsed.path}"


def request_inprocess_cloud_decision(image_path: Path, detection_result: Dict[str, Any], timeout: float):
    client = CloudClient(url="inprocess://offline-cloud", timeout=timeout, force_left=True)
    payload_start = time.monotonic()
    payload = client.build_payload(str(image_path), detection_result)
    payload_latency_ms = (time.monotonic() - payload_start) * 1000
    body = _offline_chat_completion(
        {
            "command": "left",
            "action": "lane-left",
            "reason": "offline in-process cloud: traffic cone triggers left lane change",
        }
    )

    start = time.monotonic()
    decision = client.parse_response(body, latency_ms=(time.monotonic() - start) * 1000)
    parse_latency_ms = (time.monotonic() - start) * 1000
    decision.latency_ms = payload_latency_ms + parse_latency_ms
    decision.timings_ms = {
        "payload_build": round(payload_latency_ms, 3),
        "parse": round(parse_latency_ms, 3),
        "total": round(payload_latency_ms + parse_latency_ms, 3),
    }

    return {
        "command": decision.command,
        "action": decision.action,
        "reason": decision.reason,
        "latency_ms": round(float(decision.latency_ms), 3),
        "timings_ms": decision.timings_ms,
        "backend": "inprocess",
        "api_url": client.url,
        "request_path": None,
        **_cloud_request_summary(payload),
        "raw_response": decision.raw_response,
    }


def request_public_cloud_decision(
    image_path: Path,
    detection_result: Dict[str, Any],
    timeout: float,
    cloud_url: Optional[str],
    cloud_model: Optional[str],
):
    client = CloudClient(url=cloud_url, timeout=timeout, model=cloud_model)
    decision = client.request_decision(str(image_path), detection_result)
    payload = client.last_request_payload or {}
    return {
        "command": decision.command,
        "action": decision.action,
        "reason": decision.reason,
        "latency_ms": round(float(decision.latency_ms), 3),
        "timings_ms": decision.timings_ms or {},
        "backend": "cloud",
        "api_url": _redact_public_api_url(client.url),
        "request_path": "/v1/chat/completions",
        **_cloud_request_summary(payload),
        "raw_response": decision.raw_response,
    }


def request_local_http_cloud_decision(image_path: Path, detection_result: Dict[str, Any], timeout: float):
    captured_request: Dict[str, Any] = {}
    server, url = start_local_cloud_api(captured_request)
    try:
        client = CloudClient(url=url, timeout=timeout, force_left=True)
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
        "timings_ms": decision.timings_ms or {},
        "backend": "local-http",
        "api_url": url,
        "request_path": captured_request.get("path"),
        **_cloud_request_summary(payload),
        "raw_response": decision.raw_response,
    }


def execute_vehicle_action(action: str, timeout: float):
    chassis = FakeChassis()
    controller = VehicleController(chassis=chassis)

    try:
        action_start = time.monotonic()
        controller.execute(action)
        dispatch_latency_ms = (time.monotonic() - action_start) * 1000
        thread = getattr(controller, "_action_thread", None)
        wait_start = time.monotonic()
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                raise ClosedLoopTestError(f"vehicle action thread did not finish within {timeout}s")
        wait_latency_ms = (time.monotonic() - wait_start) * 1000

        state = controller.get_state()
        ramp_commands = [cmd for cmd in chassis.commands if cmd["op"] == "ramp_to"]
        left_turn_arcs = [
            cmd for cmd in ramp_commands if cmd["left_speed"] < cmd["right_speed"]
        ]
        right_turn_arcs = [
            cmd for cmd in ramp_commands if cmd["left_speed"] > cmd["right_speed"]
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
            "contains_right_turn_arc": bool(right_turn_arcs),
            "contains_approach_hold": bool(approach_holds),
            "latencies_ms": {
                "dispatch": round(dispatch_latency_ms, 3),
                "wait_for_action": round(wait_latency_ms, 3),
                "total": round((time.monotonic() - action_start) * 1000, 3),
            },
            "first_commands": chassis.commands[:12],
        }
    finally:
        controller.stop()


def run_stage(name: str, expected_input: Dict[str, Any], func, reports: List[StageReport]):
    LOGGER.info("stage=%s start", name)
    start = time.monotonic()
    try:
        output = func()
        latency_ms = round((time.monotonic() - start) * 1000, 3)
        report = StageReport(
            name=name,
            ok=True,
            latency_ms=latency_ms,
            expected_input=expected_input,
            actual_output=output,
        )
        reports.append(report)
        LOGGER.info("stage=%s ok latency=%.3fms", name, latency_ms)
        return output
    except Exception as exc:
        latency_ms = round((time.monotonic() - start) * 1000, 3)
        report = StageReport(
            name=name,
            ok=False,
            latency_ms=latency_ms,
            expected_input=expected_input,
            actual_output={},
            error=str(exc),
        )
        reports.append(report)
        LOGGER.exception("stage=%s failed latency=%.3fms", name, latency_ms)
        raise


def assert_stage(condition: bool, message: str):
    if not condition:
        raise ClosedLoopTestError(message)


def build_parser():
    parser = argparse.ArgumentParser(description="Run a vehicle-cloud closed-loop data test.")
    parser.add_argument("--image", type=Path, default=default_image_path(), help="Test image path")
    parser.add_argument(
        "--cloud-backend",
        choices=["cloud", "inprocess", "local-http"],
        default="cloud",
        help="Cloud backend. Default cloud calls CAR_CLOUD_API_BASE_URL; inprocess/local-http are offline fallbacks.",
    )
    parser.add_argument("--cloud-url", default=None, help="Cloud API base URL override")
    parser.add_argument("--cloud-model", default=None, help="Cloud model override")
    parser.add_argument("--cloud-timeout", type=float, default=30.0, help="Cloud API request timeout")
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

    print("\nLatency details")
    print("-" * 88)
    for report in reports:
        print(f"{report.name}: stage={report.latency_ms:.3f}ms")
        for line in _latency_detail_lines(report.actual_output):
            print(f"  {line}")


def _latency_detail_lines(output: Dict[str, Any]) -> List[str]:
    if not output:
        return []

    lines = []
    latencies = output.get("latencies_ms")
    if isinstance(latencies, dict):
        for key, value in latencies.items():
            lines.append(f"{key}={float(value):.3f}ms")

    if "latency_ms" in output:
        lines.append(f"reported_latency={float(output['latency_ms']):.3f}ms")

    timings = output.get("timings_ms")
    if isinstance(timings, dict):
        for key, value in timings.items():
            lines.append(f"cloud_{key}={float(value):.3f}ms")

    for item in output.get("individual_scores", []) or []:
        if isinstance(item, dict) and "inference_time" in item:
            lines.append(f"{item.get('detector')}_inference={float(item['inference_time']) * 1000:.3f}ms")

    return lines


def main():
    setup_logging()
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
            {"detection": _compact_detection(detection), "expected": "valid left/right lane-change command and vehicle action"},
            lambda: request_cloud_decision(
                args.image,
                detection,
                args.cloud_timeout,
                args.cloud_backend,
                args.cloud_url,
                args.cloud_model,
            ),
            reports,
        )
        assert_stage(decision["command"] in ALLOWED_COMMANDS, f"unsupported cloud command: {decision['command']}")
        assert_stage(decision["action"] in ALLOWED_ACTIONS, f"unsupported vehicle action: {decision['action']}")

        vehicle = run_stage(
            "vehicle_control",
            {"action": decision["action"], "expected": "VehicleController accepts the cloud action without hardware"},
            lambda: execute_vehicle_action(decision["action"], args.action_timeout),
            reports,
        )
        assert_stage(vehicle["action_requested"] == decision["action"], "vehicle action did not match cloud decision")
        _assert_vehicle_action_shape(vehicle)

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


def _assert_vehicle_action_shape(vehicle: Dict[str, Any]):
    action = vehicle["action_requested"]
    if action == "lane-left":
        assert_stage(vehicle["contains_approach_hold"], "lane-left sequence did not include forward approach hold")
        assert_stage(vehicle["contains_left_turn_arc"], "vehicle command sequence did not contain a left-turn arc")
        assert_stage(vehicle["ramp_command_count"] >= 5, "lane-left sequence produced too few ramp commands")
    elif action == "lane-right":
        assert_stage(vehicle["contains_approach_hold"], "lane-right sequence did not include forward approach hold")
        assert_stage(vehicle["contains_right_turn_arc"], "vehicle command sequence did not contain a right-turn arc")
        assert_stage(vehicle["ramp_command_count"] >= 5, "lane-right sequence produced too few ramp commands")
    elif action == "forward":
        assert_stage(vehicle["state_after_action"]["current_action"] == "forward", "forward action did not enter forward state")
        assert_stage(vehicle["ramp_command_count"] >= 1, "forward action did not send a ramp command")
    elif action == "stop":
        assert_stage(vehicle["state_after_action"]["current_action"] == "stopped", "stop action did not enter stopped state")
    else:
        raise ClosedLoopTestError(f"unsupported vehicle action: {action}")


if __name__ == "__main__":
    main()
