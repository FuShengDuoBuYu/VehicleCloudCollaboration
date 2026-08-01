"""Small, dependency-free web supervisor for the onboard LCC process."""

from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse


MOTOR_CONFIRMATION = "I_UNDERSTAND_MOTORS_WILL_MOVE"


def build_lcc_command(
    python_executable,
    runner_path,
    config_path,
    max_runtime_seconds,
    motors_enabled,
):
    command = [
        str(python_executable),
        str(runner_path),
        "--config",
        str(config_path),
        "--max-runtime-seconds",
        f"{float(max_runtime_seconds):g}",
    ]
    if motors_enabled:
        command.extend(
            [
                "--enable-motors",
                "--confirm-motor-motion",
                MOTOR_CONFIRMATION,
            ]
        )
    return command


class LCCProcessManager:
    """Own at most one run_onboard subprocess and expose its persisted state."""

    def __init__(
        self,
        repo_root,
        python_executable,
        runner_path,
        config_path,
        output_dir,
        motors_enabled=False,
        default_max_runtime_seconds=60.0,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.python_executable = Path(python_executable).resolve()
        self.runner_path = Path(runner_path).resolve()
        self.config_path = Path(config_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.motors_enabled = bool(motors_enabled)
        self.default_max_runtime_seconds = self._validate_runtime(
            default_max_runtime_seconds
        )
        self.status_path = self.output_dir / "status.json"
        self.latest_frame = self.output_dir / "latest.jpg"
        self.latest_birdeye = self.output_dir / "latest_birdeye.jpg"

        self._lock = threading.Lock()
        self._process = None
        self._reader_thread = None
        self._state = "idle"
        self._message = "等待网页启动 LCC"
        self._started_monotonic = None
        self._started_wall = None
        self._exit_code = None
        self._active_runtime = None
        self._logs = deque(maxlen=200)

    @staticmethod
    def _validate_runtime(value):
        value = float(value)
        if not 1.0 <= value <= 300.0:
            raise ValueError("max runtime must be between 1 and 300 seconds")
        return value

    def _append_log_locked(self, level, message):
        self._logs.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "level": str(level),
                "message": str(message),
            }
        )

    def start(self, max_runtime_seconds=None, confirmation=""):
        runtime = self._validate_runtime(
            self.default_max_runtime_seconds
            if max_runtime_seconds is None
            else max_runtime_seconds
        )
        if self.motors_enabled and confirmation != MOTOR_CONFIRMATION:
            raise ValueError("启动电机前必须勾选现场安全确认")

        with self._lock:
            self._refresh_process_locked()
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("LCC 已经在运行")

            command = build_lcc_command(
                self.python_executable,
                self.runner_path,
                self.config_path,
                runtime,
                self.motors_enabled,
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._logs.clear()
            self._state = "starting"
            self._message = "正在启动 LCC，等待相机和连续有效感知"
            self._started_monotonic = time.monotonic()
            self._started_wall = time.time()
            self._exit_code = None
            self._active_runtime = runtime
            self._append_log_locked("INFO", "LCC start requested from web")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            except Exception:
                self._state = "failed"
                self._message = "LCC 进程启动失败"
                raise
            self._process = process
            self._state = "running"
            self._message = "LCC 正在运行"
            self._reader_thread = threading.Thread(
                target=self._read_output,
                args=(process,),
                daemon=True,
            )
            self._reader_thread.start()
        return self.get_state()

    def _read_output(self, process):
        if process.stdout is not None:
            try:
                for line in process.stdout:
                    line = line.rstrip()
                    if not line:
                        continue
                    with self._lock:
                        self._append_log_locked("LCC", line)
            finally:
                process.stdout.close()
        exit_code = process.wait()
        with self._lock:
            if self._process is process:
                self._exit_code = exit_code
                if self._state != "stopping":
                    self._state = "exited" if exit_code == 0 else "failed"
                    self._message = (
                        "LCC 已按计划结束并停车"
                        if exit_code == 0
                        else f"LCC 异常退出，代码 {exit_code}"
                    )
                self._append_log_locked(
                    "INFO" if exit_code == 0 else "ERROR",
                    f"LCC process exited with code {exit_code}",
                )

    def _refresh_process_locked(self):
        if self._process is None:
            return
        exit_code = self._process.poll()
        if exit_code is not None and self._exit_code is None:
            self._exit_code = exit_code
            if self._state not in {"stopped", "stopping"}:
                self._state = "exited" if exit_code == 0 else "failed"

    def _force_hardware_stop(self):
        if not self.motors_enabled:
            return
        try:
            import sys

            control_dir = self.repo_root / "car" / "control"
            if str(control_dir) not in sys.path:
                sys.path.insert(0, str(control_dir))
            from vehicle_control.hardware import RospbotChassis

            RospbotChassis().stop()
        except Exception as exc:
            with self._lock:
                self._append_log_locked(
                    "ERROR", f"fallback hardware stop failed: {exc}"
                )

    def stop(self, reason="网页急停"):
        with self._lock:
            self._refresh_process_locked()
            process = self._process
            if process is None or process.poll() is not None:
                self._state = "stopped"
                self._message = "车辆已停止"
                already_stopped = True
            else:
                already_stopped = False
                self._state = "stopping"
                self._message = f"正在停车：{reason}"
                self._append_log_locked("WARNING", reason)

        if already_stopped:
            return self.get_state()

        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

        try:
            process.wait(timeout=0.8)
        except subprocess.TimeoutExpired:
            # SIGINT normally reaches run_onboard's finally block. If cleanup
            # is slow or stuck, write zero PWM independently before escalating.
            self._force_hardware_stop()
            try:
                process.wait(timeout=1.2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=0.8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.8)
                self._force_hardware_stop()

        with self._lock:
            self._exit_code = process.poll()
            self._state = "stopped"
            self._message = "车辆已停止"
            self._append_log_locked("INFO", "LCC stopped; wheel command is zero")
        return self.get_state()

    def get_state_unlocked(self):
        running = self._process is not None and self._process.poll() is None
        elapsed = (
            None
            if self._started_monotonic is None
            else max(0.0, time.monotonic() - self._started_monotonic)
        )
        return {
            "state": self._state,
            "running": running,
            "message": self._message,
            "pid": None if self._process is None else self._process.pid,
            "exit_code": self._exit_code,
            "elapsed_seconds": None if elapsed is None else round(elapsed, 2),
            "max_runtime_seconds": self._active_runtime,
            "motors_enabled": self.motors_enabled,
            "logs": list(self._logs),
        }

    def _runtime_status(self, started_wall):
        try:
            if started_wall is not None and self.status_path.stat().st_mtime < started_wall:
                return None
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _file_state(self, path):
        try:
            modified = path.stat().st_mtime
        except OSError:
            return {"available": False, "age_seconds": None}
        return {
            "available": True,
            "age_seconds": round(max(0.0, time.time() - modified), 2),
        }

    def get_state(self):
        with self._lock:
            self._refresh_process_locked()
            process_state = self.get_state_unlocked()
            started_wall = self._started_wall
        return {
            "process": process_state,
            "runtime": self._runtime_status(started_wall),
            "frames": {
                "annotated": self._file_state(self.latest_frame),
                "birdeye": self._file_state(self.latest_birdeye),
            },
        }


class LCCWebServer:
    def __init__(self, manager, host="0.0.0.0", port=8080, html_path=None):
        self.manager = manager
        self.host = str(host)
        self.port = int(port)
        self.html_path = Path(html_path or Path(__file__).with_name("lcc_web.html"))
        self._httpd = None

    def build_handler(self):
        manager = self.manager
        html_path = self.html_path

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._serve_file(html_path, "text/html; charset=utf-8")
                elif parsed.path == "/api/state":
                    self._send_json(manager.get_state())
                elif parsed.path == "/api/latest-frame.jpg":
                    self._serve_file(manager.latest_frame, "image/jpeg")
                elif parsed.path == "/api/latest-birdeye.jpg":
                    self._serve_file(manager.latest_birdeye, "image/jpeg")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def do_POST(self):
                parsed = urlparse(self.path)
                if parsed.path != "/api/lcc":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                    return
                params = parse_qs(parsed.query)
                action = params.get("action", [None])[0]
                try:
                    if action == "start":
                        runtime = params.get("max_runtime_seconds", [None])[0]
                        confirmation = params.get("confirm", [""])[0]
                        state = manager.start(runtime, confirmation)
                    elif action in {"stop", "emergency-stop"}:
                        state = manager.stop("网页急停按钮已按下")
                    else:
                        self._send_json(
                            {"error": "unsupported LCC action"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                except (ValueError, RuntimeError) as exc:
                    self._send_json(
                        {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST
                    )
                    return
                self._send_json(state)

            def _serve_file(self, path, content_type):
                try:
                    payload = Path(path).read_bytes()
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "No frame available")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(self, payload, status=HTTPStatus.OK):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        return Handler

    def start(self):
        try:
            self._httpd = ThreadingHTTPServer(
                (self.host, self.port), self.build_handler()
            )
        except OSError as exc:
            if exc.errno == 98:
                raise RuntimeError(f"port {self.port} is already in use") from exc
            raise
        print(f"LCC web console: http://{self.host}:{self.port}", flush=True)
        print(
            "The camera remains free until the web Start LCC button is pressed.",
            flush=True,
        )
        try:
            self._httpd.serve_forever()
        finally:
            self.shutdown()

    def shutdown(self):
        if self._httpd is not None:
            httpd = self._httpd
            self._httpd = None
            threading.Thread(target=httpd.shutdown, daemon=True).start()
        self.manager.stop("LCC Web 服务关闭")
