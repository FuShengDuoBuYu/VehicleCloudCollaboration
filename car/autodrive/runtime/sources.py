"""Camera and recorded-video input adapters; hardware opens only on construction."""
from __future__ import annotations

from autodrive.camera.transform import CameraTransformConfig
from autodrive.camera.transform import transform_frame
from pathlib import Path
import csv
import cv2
import time


class VideoSource:
    def __init__(self, path, sample_every):
        self.path = Path(path).expanduser().resolve()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open video: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.sample_every = max(1, int(sample_every))
        self.frame_index = -1
        self.timestamps = self._load_archive_timestamps()

    def _load_archive_timestamps(self):
        """Use the per-frame capture clock stored beside archived videos.

        Hardware capture may run below the MP4 writer's nominal frame rate.
        The CSV has one row per archived frame and preserves the real capture
        timestamps, which are required to reproduce time-bounded control gates.
        """
        sidecar = self.path.with_name("onboard_log.csv")
        if not sidecar.is_file():
            return None
        try:
            with sidecar.open("r", encoding="utf-8", newline="") as handle:
                timestamps = [
                    float(row["timestamp_s"])
                    for row in csv.DictReader(handle)
                ]
        except (KeyError, TypeError, ValueError, OSError):
            return None
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0 or len(timestamps) != frame_count:
            return None
        return timestamps

    def read(self):
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return None, None, None
            self.frame_index += 1
            if self.frame_index % self.sample_every == 0:
                timestamp = (
                    self.timestamps[self.frame_index]
                    if self.timestamps is not None
                    else self.frame_index / self.fps
                )
                return frame, timestamp, 0.0

    def close(self):
        self.capture.release()


class CameraSource:
    def __init__(self, camera_config):
        from dataclasses import replace as dc_replace
        from vehicle_control.camera import CameraStream
        from vehicle_control.settings import CAMERA_CONFIG

        self.startup_timeout = float(camera_config.get("startup_timeout", 10.0))
        self.stale_timeout = float(camera_config.get("stale_timeout", 1.0))
        self.frame_transform = CameraTransformConfig.from_mapping(camera_config)
        config = dc_replace(
            CAMERA_CONFIG,
            camera_index=int(camera_config.get("index", 0)),
            width=int(camera_config.get("width", 640)),
            height=int(camera_config.get("height", 480)),
            fps=int(camera_config.get("fps", 20)),
        )
        self.camera = CameraStream(config)
        self.camera.start()
        self.started_at = time.monotonic()
        self.last_sequence = 0

    def read(self):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            frame, captured_at, sequence = self.camera.get_frame_packet()
            age = (
                None
                if captured_at is None
                else time.monotonic() - captured_at
            )
            if (
                frame is not None
                and age is not None
                and age <= self.stale_timeout
                and sequence != self.last_sequence
            ):
                self.last_sequence = sequence
                return (
                    transform_frame(frame, self.frame_transform),
                    captured_at - self.started_at,
                    age,
                )
            time.sleep(0.05)
        return None, None, None

    def close(self):
        self.camera.stop()

