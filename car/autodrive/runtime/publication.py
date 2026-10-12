"""Best-effort live diagnostics; publication never grants motion permission."""
from __future__ import annotations

from autodrive.runtime.io_utils import _offer_latest
from autodrive.runtime.io_utils import write_status
from pathlib import Path
import cv2
import numpy as np
import queue
import threading


class LivePublisher:
    """Publish replaceable web snapshots/status outside motor control."""

    def __init__(
        self,
        status_path,
        archive_status_path,
        latest_frame_path,
        latest_birdeye_path,
        save_latest,
    ):
        self.status_path = Path(status_path)
        self.archive_status_path = (
            None
            if archive_status_path is None
            else Path(archive_status_path)
        )
        self.latest_frame_path = Path(latest_frame_path)
        self.latest_birdeye_path = Path(latest_birdeye_path)
        self.save_latest = bool(save_latest)
        self._queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._published = 0
        self._dropped = 0
        self._error = None
        self._thread = threading.Thread(
            target=self._write_loop,
            name="lcc-live-publisher",
            daemon=True,
        )
        self._thread.start()

    def _write_loop(self):
        try:
            while True:
                record = self._queue.get()
                try:
                    if record is None:
                        break
                    payload, annotated, birdeye = record
                    if self.save_latest:
                        self._write_image(self.latest_frame_path, annotated)
                        if birdeye is not None:
                            self._write_image(self.latest_birdeye_path, birdeye)
                    write_status(self.status_path, payload)
                    if self.archive_status_path is not None:
                        write_status(self.archive_status_path, payload)
                    with self._lock:
                        self._published += 1
                finally:
                    self._queue.task_done()
        except Exception as exc:
            with self._lock:
                self._error = exc

    @staticmethod
    def _write_image(path, frame):
        # Readers see either complete old JPEG or complete new JPEG.
        temporary = path.with_name(path.stem + '.pending' + path.suffix)
        if not cv2.imwrite(str(temporary), frame):
            raise RuntimeError('failed to write live image')
        temporary.replace(path)

    def publish(self, payload, annotated, birdeye):
        with self._lock:
            failed = self._error is not None
        if failed:
            with self._lock:
                self._dropped += 1
            return False
        accepted, dropped = _offer_latest(
            self._queue,
            (dict(payload), annotated, birdeye),
        )
        with self._lock:
            self._dropped += dropped + int(not accepted)
        return accepted

    def get_state(self):
        with self._lock:
            return {
                "queue_depth": self._queue.qsize(),
                "published_updates": self._published,
                "dropped_updates": self._dropped,
                "error": None if self._error is None else str(self._error),
            }

    def close(self):
        if self._thread is None:
            return
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self._thread = None


class DebugFrameWriter:
    """Best-effort debug image writer that cannot stall vehicle control."""

    def __init__(self, output_dir, queue_frames=8):
        self.output_dir = None if output_dir is None else Path(output_dir)
        self._queue = None
        self._thread = None
        self._lock = threading.Lock()
        self._written = 0
        self._dropped = 0
        self._error = None
        if self.output_dir is None:
            return
        queue_frames = int(queue_frames)
        if queue_frames < 1:
            raise ValueError("debug queue_frames must be at least 1")
        self._queue = queue.Queue(maxsize=queue_frames)
        self._thread = threading.Thread(
            target=self._write_loop,
            name="lcc-debug-frames",
            daemon=True,
        )
        self._thread.start()

    def _write_loop(self):
        try:
            while True:
                record = self._queue.get()
                try:
                    if record is None:
                        break
                    sample, raw, annotated, birdeye = record
                    stem = f"{sample:04d}"
                    if not cv2.imwrite(
                        str(self.output_dir / f"{stem}_raw.jpg"), raw
                    ):
                        raise RuntimeError("failed to write raw debug frame")
                    if not cv2.imwrite(
                        str(self.output_dir / f"{stem}_annotated.jpg"),
                        annotated,
                    ):
                        raise RuntimeError("failed to write annotated debug frame")
                    if birdeye is not None and not cv2.imwrite(
                        str(self.output_dir / f"{stem}_birdeye.png"), birdeye
                    ):
                        raise RuntimeError("failed to write bird's-eye debug frame")
                    with self._lock:
                        self._written += 1
                finally:
                    self._queue.task_done()
        except Exception as exc:
            with self._lock:
                self._error = exc

    def submit(self, sample, raw, annotated, birdeye):
        if self._queue is None:
            return False
        with self._lock:
            failed = self._error is not None
        if failed:
            with self._lock:
                self._dropped += 1
            return False
        accepted, dropped = _offer_latest(
            self._queue,
            (int(sample), raw, annotated, birdeye),
        )
        with self._lock:
            self._dropped += dropped + int(not accepted)
        return accepted

    def get_state(self):
        with self._lock:
            return {
                "enabled": self._queue is not None,
                "queue_depth": 0 if self._queue is None else self._queue.qsize(),
                "written_frames": self._written,
                "dropped_frames": self._dropped,
                "error": None if self._error is None else str(self._error),
            }

    def close(self):
        if self._thread is None:
            return
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self._thread = None


def render_birdeye_debug(boundary_result, estimate):
    """Render the exact corridor geometry consumed by the controller."""
    corridor = (np.asarray(boundary_result.corridor_mask) > 0).astype(np.uint8)
    height, width = corridor.shape
    output = np.full((height, width, 3), 28, dtype=np.uint8)
    output[corridor > 0] = (40, 120, 40)
    historical = boundary_result.source in {"history", "visible-history"}
    for curve, inferred in (
        (
            boundary_result.left_curve,
            historical or boundary_result.source == "inner+width",
        ),
        (
            boundary_result.right_curve,
            historical or boundary_result.source == "outer+width",
        ),
    ):
        if curve.size == height:
            points = np.column_stack(
                [np.clip(curve, 0, width - 1), np.arange(height)]
            ).astype(np.int32)
            if inferred:
                for start in range(0, height - 1, 12):
                    end = min(height - 1, start + 6)
                    cv2.line(
                        output,
                        tuple(map(int, points[start])),
                        tuple(map(int, points[end])),
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                cv2.polylines(output, [points], False, (255, 160, 0), 2)
    if estimate.centerline.size:
        cv2.polylines(output, [estimate.centerline], False, (0, 255, 255), 3)
    ego = (width // 2, int(height * 0.92))
    cv2.circle(output, ego, 5, (255, 255, 255), -1)
    if estimate.lookahead_point is not None:
        cv2.arrowedLine(
            output,
            ego,
            tuple(map(int, estimate.lookahead_point)),
            (255, 255, 255),
            2,
        )
    cv2.putText(
        output,
        f"{boundary_result.source}: cyan=measured magenta=inferred",
        (5, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output

