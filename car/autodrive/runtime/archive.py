"""Run evidence archives and bounded asynchronous CSV output."""
from __future__ import annotations

from autodrive.runtime.io_utils import _offer_latest
from autodrive.runtime.io_utils import write_status
from datetime import datetime
from pathlib import Path
import csv
import cv2
import json
import numpy as np
import queue
import shutil
import threading
import yaml


class RunArchive:
    """Persist replay-aligned videos without blocking the control loop."""

    def __init__(self, output_dir, enabled, fps, mode, queue_frames=32):
        self.enabled = bool(enabled)
        self.fps = max(1.0, float(fps))
        self.run_dir = None
        self.status_path = None
        self.log_path = None
        self._writers = {}
        self._frame_queue = None
        self._writer_thread = None
        self._writer_error = None
        self._stats_lock = threading.Lock()
        self._enqueued_frames = 0
        self._written_frames = 0
        self._dropped_frames = 0
        self._archive_log_handle = None
        self._archive_log_writer = None
        self._archived_semantic_sequences = set()
        if not self.enabled:
            return
        queue_frames = int(queue_frames)
        if queue_frames < 1:
            raise ValueError("archive queue_frames must be at least 1")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{mode}"
        self.run_dir = Path(output_dir) / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.status_path = self.run_dir / "status.json"
        self.log_path = self.run_dir / "onboard_log.csv"
        self._frame_queue = queue.Queue(maxsize=queue_frames)
        self._writer_thread = threading.Thread(
            target=self._write_loop,
            name="lcc-run-archive",
            daemon=True,
        )
        self._writer_thread.start()

    def snapshot_file(self, source, target_name):
        if not self.enabled or source is None:
            return
        source = Path(source)
        if source.is_file():
            shutil.copy2(source, self.run_dir / target_name)

    def snapshot_mapping(self, mapping, target_name):
        if not self.enabled:
            return
        (self.run_dir / target_name).write_text(
            yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _writer(self, name, frame):
        writer = self._writers.get(name)
        if writer is not None:
            return writer
        height, width = frame.shape[:2]
        path = self.run_dir / name
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"unable to create archived video: {path}")
        self._writers[name] = writer
        return writer

    def _write_frames_now(self, raw, annotated, birdeye):
        self._writer("raw.mp4", raw).write(raw)
        self._writer("annotated.mp4", annotated).write(annotated)
        if birdeye is not None:
            self._writer("birdeye.mp4", birdeye).write(birdeye)

    def _write_record_now(self, raw, annotated, birdeye, row, semantic_observation=None):
        if semantic_observation is not None:
            sequence = int(semantic_observation['sequence'])
            if sequence not in self._archived_semantic_sequences:
                folder = self.run_dir / 'semantic_inputs'
                folder.mkdir(exist_ok=True)
                # Lossless actual model input and outputs; no JPEG/MP4 roundtrip
                # when reproducing a live decision. Written on archive thread.
                # Compression measured ~71 ms per observation on this Jetson,
                # overflowing the 20 Hz video queue. Uncompressed NPZ is still
                # lossless (~6.5 ms), with explicit disk/write failure stopping.
                np.savez(folder / ('%08d.npz' % sequence),
                    frame=semantic_observation['frame'], road=semantic_observation['mask'],
                    lane=semantic_observation['lane_mask'],
                    captured_at=semantic_observation['captured_at'],
                    sequence=sequence,
                    detections_json=json.dumps(semantic_observation.get('detections', [])))
                self._archived_semantic_sequences.add(sequence)
        self._write_frames_now(raw, annotated, birdeye)
        # Keep the archive sidecar exactly aligned with the frames that were
        # actually encoded.  If backpressure forced a frame drop, deterministic
        # replay still receives the correct timestamp for every retained frame.
        if self._archive_log_writer is None:
            self._archive_log_handle = self.log_path.open(
                "w", encoding="utf-8", newline=""
            )
            self._archive_log_writer = csv.DictWriter(
                self._archive_log_handle,
                fieldnames=list(row.keys()),
            )
            self._archive_log_writer.writeheader()
        self._archive_log_writer.writerow(row)
        self._archive_log_handle.flush()

    def _write_loop(self):
        try:
            while True:
                record = self._frame_queue.get()
                try:
                    if record is None:
                        break
                    if self._writer_error is None:
                        self._write_record_now(*record)
                        with self._stats_lock:
                            self._written_frames += 1
                    else:
                        with self._stats_lock:
                            self._dropped_frames += 1
                except Exception as exc:
                    self._writer_error = exc
                    with self._stats_lock:
                        self._dropped_frames += 1
                finally:
                    self._frame_queue.task_done()
        finally:
            for writer in self._writers.values():
                writer.release()
            self._writers.clear()
            if self._archive_log_handle is not None:
                self._archive_log_handle.close()
                self._archive_log_handle = None
                self._archive_log_writer = None

    def write_frames(self, raw, annotated, birdeye, row, semantic_observation=None):
        if not self.enabled:
            return False
        if self._writer_error is not None:
            with self._stats_lock:
                self._dropped_frames += 1
            return False
        accepted, dropped = _offer_latest(
            self._frame_queue,
            (raw, annotated, birdeye, dict(row), semantic_observation),
        )
        with self._stats_lock:
            self._enqueued_frames += int(accepted)
            self._dropped_frames += dropped + int(not accepted)
        return accepted

    def get_state(self):
        with self._stats_lock:
            state = {
                "enabled": self.enabled,
                "queue_capacity": (
                    0 if self._frame_queue is None else self._frame_queue.maxsize
                ),
                "queue_depth": (
                    0 if self._frame_queue is None else self._frame_queue.qsize()
                ),
                "enqueued_frames": self._enqueued_frames,
                "written_frames": self._written_frames,
                "dropped_frames": self._dropped_frames,
                "error": (
                    None
                    if self._writer_error is None
                    else str(self._writer_error)
                ),
            }
        return state

    def write_status(self, payload):
        if self.enabled:
            write_status(self.status_path, payload)

    def close(self):
        if not self.enabled or self._writer_thread is None:
            return
        self._frame_queue.put(None)
        self._writer_thread.join()
        self._writer_thread = None


class AsyncCsvLogger:
    """Write control records in the background with bounded memory."""

    def __init__(self, path, fieldnames, queue_rows=256):
        queue_rows = int(queue_rows)
        if queue_rows < 1:
            raise ValueError("CSV queue_rows must be at least 1")
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._queue = queue.Queue(maxsize=queue_rows)
        self._lock = threading.Lock()
        self._enqueued = 0
        self._written = 0
        self._dropped = 0
        self._error = None
        self._thread = threading.Thread(
            target=self._write_loop,
            name="lcc-csv-log",
            daemon=True,
        )
        self._thread.start()

    def _write_loop(self):
        try:
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                while True:
                    row = self._queue.get()
                    try:
                        if row is None:
                            break
                        writer.writerow(row)
                        handle.flush()
                        with self._lock:
                            self._written += 1
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            with self._lock:
                self._error = exc

    def submit(self, row):
        with self._lock:
            failed = self._error is not None
        if failed:
            with self._lock:
                self._dropped += 1
            return False
        accepted, dropped = _offer_latest(self._queue, dict(row))
        with self._lock:
            self._enqueued += int(accepted)
            self._dropped += dropped + int(not accepted)
        return accepted

    def get_state(self):
        with self._lock:
            return {
                "queue_capacity": self._queue.maxsize,
                "queue_depth": self._queue.qsize(),
                "enqueued_rows": self._enqueued,
                "written_rows": self._written,
                "dropped_rows": self._dropped,
                "error": None if self._error is None else str(self._error),
            }

    def close(self):
        if self._thread is None:
            return
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self._thread = None

