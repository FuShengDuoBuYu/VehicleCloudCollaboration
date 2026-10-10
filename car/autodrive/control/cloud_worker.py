"""Bounded asynchronous scene requests, with all motion decisions on the caller."""
import json
import hashlib
from pathlib import Path
import queue
import threading
import time

import cv2

from .cloud_arbitration import SceneArbiter
from .lane_centering import DifferentialDriveCommand


class CloudCoordinator:
    # The real CloudClient redacts these fields. Raw provider responses and
    # unvalidated response text are deliberately excluded from this archive.
    _METADATA_FIELDS = ("provider", "requested_model", "schema_version", "prompt_version",
                        "request_id", "response_id", "started_at", "finished_at",
                        "timings_ms", "elapsed_ms", "input_manifest", "request_config", "http_status", "usage")
    def __init__(self, config, client, output_dir):
        self.arbiter = SceneArbiter(config)
        self.client = client
        self.output_dir = Path(output_dir)
        self._jobs = queue.Queue(maxsize=1)
        self._results = queue.Queue(maxsize=2)
        self._closed = threading.Event()
        self._thread = None
        if config.enabled:
            if client is None:
                raise ValueError("enabled cloud arbitration requires a scene client")
            self._thread = threading.Thread(target=self._run, name="cloud-scene", daemon=True)
            self._thread.start()

    @staticmethod
    def _replace_pending(channel, value):
        try:
            channel.put_nowait(value)
        except queue.Full:
            try:
                channel.get_nowait()
            except queue.Empty:
                pass
            channel.put_nowait(value)

    def filter(self, command, observation, local_safe, reason, now=None):
        if self._closed.is_set():
            return self._stop(command, "cloud coordinator closed")
        if not self.arbiter.config.enabled:
            return command
        now = time.monotonic() if now is None else now
        while True:
            try:
                event_id, scene, error = self._results.get_nowait()
            except queue.Empty:
                break
            self.arbiter.receive(event_id, scene, now, error=error)
        event = self.arbiter.observe(observation, local_safe, reason, now)
        if event is not None:
            frame = observation.get("frame")
            if frame is None:
                self.arbiter.receive(event["event_id"], None, now, error="missing bound image")
            else:
                self._replace_pending(self._jobs, (event, frame.copy()))
        if not local_safe:
            return self._stop(command, "local perception: " + reason)
        if not self.arbiter.allowed:
            return self._stop(command, "cloud arbitration: " + self.arbiter.reason)
        return command

    @staticmethod
    def _stop(command, reason):
        return DifferentialDriveCommand("stop", 0., 0., 0., command.confidence, reason)

    @staticmethod
    def _write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def _run(self):
        while not self._closed.is_set():
            try:
                job = self._jobs.get(timeout=.1)
            except queue.Empty:
                continue
            if job is None or self._closed.is_set():
                return
            event, frame = job
            scene = None
            error = None
            client_entered = False
            previous_metadata = getattr(self.client, "last_request_metadata", None)
            folder = self.output_dir / event["event_id"]
            response = {"event_id": event["event_id"], "metadata": {},
                        "worker_started_monotonic": time.monotonic()}
            try:
                folder.mkdir(parents=True, exist_ok=False)
                image_path = folder / "input.png"
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError("image archive failed")
                response["input_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
                context = dict(event, task="outer-loop; route suggestions are recorded only")
                self._write(folder / "request.json", context)
                client_entered = True
                result = self.client.request_scene([image_path], context=context)
                scene = result.scene
                response["metadata"] = {name: getattr(result, name) for name in self._METADATA_FIELDS
                                        if hasattr(result, name)}
                response.update(scene=scene, response_model=result.response_model,
                                request_id=result.request_id)
            except Exception as exc:
                # Provider exception text or repr can contain credentials.
                error = type(exc).__name__
                response["error_type"] = error
                metadata = getattr(self.client, "last_request_metadata", {})
                # CloudClient replaces its metadata when a request begins.
                # A pre-request archive/key failure must not inherit the prior event.
                if client_entered and metadata is not previous_metadata:
                    response["metadata"] = {name: metadata[name] for name in self._METADATA_FIELDS
                                            if name in metadata}
            try:
                response["worker_finished_monotonic"] = time.monotonic()
                self._write(folder / "response.json", response)
            except Exception:
                error = "EvidenceWriteError"
            if not self._closed.is_set():
                self._replace_pending(self._results, (event["event_id"], scene, error))

    def get_state(self):
        return dict(self.arbiter.get_state(), pending_jobs=self._jobs.qsize(),
                    motion_allowed=self.arbiter.allowed and not self._closed.is_set(),
                    worker_alive=self._thread is not None and self._thread.is_alive(),
                    closed=self._closed.is_set())

    def close(self):
        self._closed.set()
        if self._thread is not None:
            self._replace_pending(self._jobs, None)
            # The driver must be stopped before shutdown; a bounded in-flight
            # HTTP request may finish later, but it has no actuator reference.
            self._thread.join(timeout=.2)
