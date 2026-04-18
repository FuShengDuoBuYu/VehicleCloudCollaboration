import threading
import time

import cv2

from .settings import CAMERA_CONFIG


class CameraStream:
    def __init__(self, config=CAMERA_CONFIG):
        self.config = config
        self.capture = None
        self.active_camera_index = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return

        for camera_index in self._candidate_indices():
            capture = cv2.VideoCapture(camera_index)
            if not capture or not capture.isOpened():
                if capture:
                    capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)

            ok, frame = capture.read()
            if ok:
                self.capture = capture
                self.active_camera_index = camera_index
                with self._lock:
                    self._latest_frame = frame
                break

            capture.release()

        if self.capture is None:
            print(
                f"Camera unavailable. Tried indices {self._candidate_indices()}. "
                "The web page will use a placeholder frame."
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _candidate_indices(self):
        preferred = int(self.config.camera_index)
        candidates = [preferred]
        for fallback in [0, 1, 2, 3]:
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    def _capture_loop(self):
        interval = 1.0 / max(self.config.fps, 1)
        while self._running:
            ok, frame = self.capture.read() if self.capture else (False, None)
            if ok:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.1)
            time.sleep(interval)

    def get_jpeg_bytes(self):
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        if frame is None:
            frame = self._placeholder_frame()

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
        )
        if not ok:
            return b""
        return encoded.tobytes()

    def _placeholder_frame(self):
        frame = 255 * self._blank_image()
        cv2.putText(
            frame,
            "Camera unavailable",
            (40, self.config.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _blank_image(self):
        import numpy as np

        return np.ones((self.config.height, self.config.width, 3), dtype=np.uint8)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.capture:
            self.capture.release()
            self.capture = None
