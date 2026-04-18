import json
import os
import threading
import time
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .camera import CameraStream
from .controller import VehicleController
from .settings import CAMERA_CONFIG, SERVER_CONFIG


class VehicleControlServer:
    def __init__(self, controller=None, camera=None, server_config=SERVER_CONFIG):
        self.controller = controller or VehicleController()
        self.camera = camera or CameraStream()
        self.server_config = server_config
        self._httpd = None

    def build_handler(self):
        controller = self.controller
        camera = self.camera
        static_dir = os.path.join(os.path.dirname(__file__), "static")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._serve_file(os.path.join(static_dir, "index.html"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/state":
                    self._send_json(controller.get_state())
                    return
                if parsed.path == "/stream.mjpg":
                    self._serve_stream()
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

            def do_POST(self):
                parsed = urlparse(self.path)
                if parsed.path != "/api/control":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                    return
                params = parse_qs(parsed.query)
                action = params.get("action", [None])[0]
                if not action:
                    self._send_json({"error": "missing action"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    controller.execute(action)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(controller.get_state())

            def log_message(self, fmt, *args):
                return

            def _serve_file(self, path, content_type):
                try:
                    with open(path, "rb") as handle:
                        payload = handle.read()
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(self, payload, status=HTTPStatus.OK):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_stream(self):
                self.send_response(HTTPStatus.OK)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = camera.get_jpeg_bytes()
                        if not frame:
                            time.sleep(0.1)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return

        return Handler

    def start(self):
        self.camera.start()
        try:
            self._httpd = ThreadingHTTPServer((self.server_config.host, self.server_config.port), self.build_handler())
        except OSError as exc:
            if exc.errno == 98:
                raise RuntimeError(
                    f"Port {self.server_config.port} is already in use. "
                    f"If the previous service is still running, open http://127.0.0.1:{self.server_config.port} directly, "
                    "or restart with another port."
                ) from exc
            raise
        print(f"Vehicle control server running on http://{self.server_config.host}:{self.server_config.port}")
        if self.camera.active_camera_index is not None:
            print(f"Camera stream ready on index {self.camera.active_camera_index}")
        try:
            self._httpd.serve_forever()
        finally:
            self.shutdown()

    def shutdown(self):
        if self._httpd:
            threading.Thread(target=self._httpd.shutdown, daemon=True).start()
            self._httpd = None
        self.controller.stop()
        self.camera.stop()


def run_server(host=None, port=None, camera_index=None):
    server_config = SERVER_CONFIG
    camera = None

    if host is not None or port is not None:
        server_config = replace(
            SERVER_CONFIG,
            host=host if host is not None else SERVER_CONFIG.host,
            port=port if port is not None else SERVER_CONFIG.port,
        )

    if camera_index is not None:
        camera = CameraStream(config=replace(CAMERA_CONFIG, camera_index=camera_index))

    server = VehicleControlServer(camera=camera, server_config=server_config)
    server.start()
