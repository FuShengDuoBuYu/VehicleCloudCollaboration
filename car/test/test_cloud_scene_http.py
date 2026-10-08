"""Real loopback HTTP and CLI tests; all credentials and responses are synthetic."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_client import CloudAPIError, CloudClient
from cloud_client.cli import main
from test_cloud_scene_client import completion, scene


class SceneHTTPTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "frame.jpg"
        self.image.write_bytes(b"\xff\xd8\xfftest-image")
        self.env_file = self.root / "empty.env"
        self.env_file.write_text("", encoding="utf-8")
        self.key = "synthetic-test-key-only"
        self.requests = []
        self.response = completion().encode()
        self.status = 200
        self.delay = 0
        self.malformed_chunked = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                       "payload": json.loads(self.rfile.read(int(self.headers["Content-Length"])) )})
                if owner.delay:
                    time.sleep(owner.delay)
                self.send_response(owner.status)
                if owner.malformed_chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(b"8\r\nshort")
                    return
                if owner.status == 302:
                    self.send_header("Location", owner.url + "/redirected")
                self.send_header("Content-Length", str(len(owner.response)))
                self.end_headers()
                try:
                    self.wfile.write(owner.response)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = "http://127.0.0.1:" + str(self.server.server_port) + "/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def client(self, **overrides):
        return CloudClient(env_file=self.env_file, url=self.url, api_key=self.key, **overrides)

    def test_auth_post_scene_envelope_usage_and_image_provenance(self):
        result = self.client().request_scene(self.image, {"run_id": "offline-001", "vehicle_stopped": True})
        self.assertEqual(self.requests[0]["auth"], "Bearer " + self.key)
        self.assertEqual(self.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(result.scene, scene())
        self.assertEqual(result.context["run_id"], "offline-001")
        self.assertEqual(len(result.input_manifest[0]["sha256"]), 64)
        self.assertTrue(result.request_id)
        self.assertTrue(result.started_at)
        self.assertTrue(result.finished_at)
        self.assertGreaterEqual(result.timings_ms["total"], result.timings_ms["http"])
        self.assertNotIn(self.key, json.dumps(asdict(result)))
        self.assertEqual(result.usage["completion_tokens"], 150)
        self.assertEqual(result.request_config["endpoint"], self.url + "/chat/completions")
        self.assertEqual(result.request_config["reasoning_effort"], "none")
        self.assertEqual(result.request_config["max_tokens"], 1024)

    def test_http_error_never_exposes_key_or_provider_body(self):
        self.status = 401
        self.response = ("provider echoed " + self.key).encode()
        with self.assertRaisesRegex(CloudAPIError, "HTTP 401") as caught:
            self.client().request_scene(self.image)
        self.assertNotIn(self.key, str(caught.exception))
        self.assertEqual(len(self.requests), 1)

    def test_rejects_redirect_without_following_or_retrying(self):
        self.status = 302
        with self.assertRaisesRegex(CloudAPIError, "HTTP 302"):
            self.client().request_scene(self.image)
        self.assertEqual(len(self.requests), 1)

    def test_socket_timeout_fails_without_retry(self):
        self.delay = 0.1
        with self.assertRaisesRegex(CloudAPIError, "timed out"):
            self.client(timeout=0.02).request_scene(self.image)
        self.assertEqual(len(self.requests), 1)

    def test_incomplete_scene_becomes_safe_error(self):
        self.response = completion(finish="length").encode()
        with self.assertRaisesRegex(CloudAPIError, "validation"):
            self.client().request_scene(self.image)

    def test_echoed_key_is_redacted_from_preserved_response(self):
        value = scene(); value["reason"] += self.key
        self.response = completion(value).encode()
        result = self.client().request_scene(self.image, {"test_data": self.key})
        self.assertNotIn(self.key, json.dumps(asdict(result)))
        self.assertIn("[REDACTED]", result.scene["reason"])

    def test_cli_dry_run_never_calls_network_or_shows_key(self):
        out = io.StringIO()
        with patch.dict(os.environ, {"CAR_CLOUD_API_KEY": self.key}, clear=True), redirect_stdout(out), patch("urllib.request.OpenerDirector.open") as network:
            status = main(["--env-file", str(self.env_file), "--dry-run", "--image", str(self.image)])
        self.assertEqual(status, 0)
        network.assert_not_called()
        self.assertNotIn(self.key, out.getvalue())
        self.assertFalse(json.loads(out.getvalue())["network_called"])

    def test_cli_live_creates_complete_record_and_refuses_overwrite(self):
        output = self.root / "scene.json"
        argv = ["--env-file", str(self.env_file), "--url", self.url, "--image", str(self.image), "--output", str(output)]
        with patch.dict(os.environ, {"CAR_CLOUD_API_KEY": self.key}, clear=True), redirect_stdout(io.StringIO()):
            self.assertEqual(main(argv), 0)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(record["scene"]["recommendation"], "stop")
        self.assertNotIn(self.key, output.read_text(encoding="utf-8"))
        before = output.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(argv)
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(len(self.requests), 1)

    def test_cli_requires_output_before_network(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--image", str(self.image)])
        self.assertEqual(len(self.requests), 0)

    def test_cli_failed_http_preserves_failure_without_key(self):
        self.status = 429
        self.response = self.key.encode()
        output = self.root / "failure.json"
        with patch.dict(os.environ, {"CAR_CLOUD_API_KEY": self.key}, clear=True), redirect_stderr(io.StringIO()):
            status = main(["--env-file", str(self.env_file), "--url", self.url,
                           "--image", str(self.image), "--output", str(output)])
        self.assertEqual(status, 1)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(record["mode"], "failed-request")
        self.assertTrue(record["request"]["request_id"])
        self.assertEqual(record["request"]["http_status"], 429)
        self.assertTrue(record["request"]["input_manifest"])
        self.assertNotIn(self.key, output.read_text(encoding="utf-8"))

    def test_cli_malformed_http_body_preserves_failure_record(self):
        self.malformed_chunked = True
        output = self.root / "malformed.json"
        with patch.dict(os.environ, {"CAR_CLOUD_API_KEY": self.key}, clear=True), redirect_stderr(io.StringIO()):
            status = main(["--env-file", str(self.env_file), "--url", self.url,
                           "--image", str(self.image), "--output", str(output)])
        self.assertEqual(status, 1)
        record = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(record["mode"], "failed-request")
        self.assertTrue(record["request"]["request_id"])


if __name__ == "__main__":
    unittest.main()
