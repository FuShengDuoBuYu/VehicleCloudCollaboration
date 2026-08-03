"""Asynchronous YOLOPv2 masks and conservative LCC corridor fusion."""

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class YOLOPv2FusionConfig:
    """Runtime settings for sharing a slow semantic model with the fast LCC."""

    enabled: bool = False
    weights: str = ""
    device: str = "cpu"
    img_size: int = 320
    fast_mask: bool = True
    optimize_for_inference: bool = True
    drivable_only: bool = True
    torch_num_threads: int = 1
    torch_interop_threads: int = 1
    asynchronous: bool = True
    fusion_mode: str = "intersection"
    max_result_age_seconds: float = 2.5
    minimum_overlap_ratio: float = 0.30
    drivable_dilate_kernel: int = 9
    confidence_weight: float = 0.20
    required_for_motion: bool = False

    def __post_init__(self):
        if self.img_size < 32:
            raise ValueError("YOLOPv2 img_size must be at least 32")
        if self.torch_num_threads < 0 or self.torch_interop_threads < 0:
            raise ValueError("YOLOPv2 torch thread counts must not be negative")
        if self.fusion_mode not in {"intersection", "validate"}:
            raise ValueError("YOLOPv2 fusion_mode must be intersection or validate")
        if self.max_result_age_seconds <= 0.0:
            raise ValueError("YOLOPv2 max_result_age_seconds must be positive")
        if not 0.0 <= self.minimum_overlap_ratio <= 1.0:
            raise ValueError("YOLOPv2 minimum_overlap_ratio must be in [0, 1]")
        if self.drivable_dilate_kernel < 1:
            raise ValueError("YOLOPv2 drivable_dilate_kernel must be positive")
        if not 0.0 <= self.confidence_weight <= 1.0:
            raise ValueError("YOLOPv2 confidence_weight must be in [0, 1]")


@dataclass(frozen=True)
class _MaskResult:
    sequence: int
    captured_at: float
    completed_at: float
    inference_seconds: float
    drivable_mask: np.ndarray
    lane_mask: np.ndarray


class YOLOPv2FusionDetector:
    """Run YOLOPv2 off the control thread and fuse its latest safe result.

    YOLOPv2 is substantially slower than the 20 Hz classical boundary loop on
    the Raspberry Pi.  After one stopped warm-up inference, subsequent calls
    enqueue only the newest frame and immediately return the latest completed
    masks.  The semantic mask may only validate or shrink the classical LCC
    corridor; it can never enlarge it into a YOLO false-positive region.
    """

    source_name = "yolopv2-fusion"

    def __init__(
        self,
        config: YOLOPv2FusionConfig,
        output_width: int = 320,
        output_height: int = 180,
        model: Optional[Any] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        if self.output_width < 16 or self.output_height < 16:
            raise ValueError("YOLOPv2 output mask dimensions must be at least 16")
        self._clock = clock
        if model is None:
            weights = Path(config.weights).expanduser()
            if not weights.is_file():
                raise FileNotFoundError(f"YOLOPv2 weights not found: {weights}")
            # Keep the heavyweight torch/transformers dependency out of module
            # import so classical-only tests and tools remain lightweight.
            import torch

            if config.torch_num_threads > 0:
                torch.set_num_threads(int(config.torch_num_threads))
            if config.torch_interop_threads > 0:
                try:
                    torch.set_num_interop_threads(
                        int(config.torch_interop_threads)
                    )
                except RuntimeError:
                    # Another torch consumer may already have initialized its
                    # inter-op pool. Intra-op limiting still protects LCC.
                    pass
            from longtail.detectors.yolopv2_detector import YOLOPv2Detector

            model = YOLOPv2Detector(
                {
                    "weights_path": str(weights),
                    "device": config.device,
                    "img_size": int(config.img_size),
                    "fast_mask": bool(config.fast_mask),
                    "optimize_for_inference": bool(
                        config.optimize_for_inference
                    ),
                    "drivable_only": bool(config.drivable_only),
                    "use_full_model": True,
                }
            )
        self.model = model
        self._lock = threading.Lock()
        self._latest: Optional[_MaskResult] = None
        self._consumer_result: Optional[_MaskResult] = None
        self._last_fusion = self._empty_fusion_state("unavailable")
        self._error: Optional[str] = None
        self._sequence = 0
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._closed = False
        self._queue = queue.Queue(maxsize=1) if config.asynchronous else None
        self._thread = None
        if self._queue is not None:
            self._thread = threading.Thread(
                target=self._worker,
                name="yolopv2-fusion",
                daemon=True,
            )
            self._thread.start()

    @staticmethod
    def _empty_fusion_state(source: str) -> dict:
        return {
            "active": False,
            "motion_allowed": True,
            "source": source,
            "result_age_seconds": None,
            "overlap_ratio": None,
            "drivable_ratio": None,
            "inference_seconds": None,
            "confidence_scale": 1.0,
        }

    def _next_sequence(self) -> int:
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
            return sequence

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if binary.ndim != 2:
            raise ValueError("YOLOPv2 masks must be two-dimensional")
        if binary.shape != (self.output_height, self.output_width):
            binary = cv2.resize(
                binary,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return (binary > 0).astype(np.uint8)

    def _infer(self, sequence: int, frame: np.ndarray, captured_at: float) -> _MaskResult:
        started = self._clock()
        drivable, lane = self.model.predict_masks(frame)
        completed = self._clock()
        return _MaskResult(
            sequence=sequence,
            captured_at=float(captured_at),
            completed_at=float(completed),
            inference_seconds=float(completed - started),
            drivable_mask=self._resize_mask(drivable),
            lane_mask=self._resize_mask(lane),
        )

    def _publish_result(self, result: _MaskResult) -> None:
        with self._lock:
            self._latest = result
            self._completed += 1
            self._error = None

    def _run_job(self, job) -> None:
        sequence, frame, captured_at = job
        try:
            self._publish_result(self._infer(sequence, frame, captured_at))
        except Exception as exc:  # surfaced through state; old safe result expires
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._run_job(job)
            finally:
                self._queue.task_done()

    def _submit_latest(self, job) -> None:
        try:
            self._queue.put_nowait(job)
            with self._lock:
                self._submitted += 1
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self._queue.task_done()
            with self._lock:
                self._dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(job)
            with self._lock:
                self._submitted += 1
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def predict_masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("YOLOPv2 fusion detector is closed")
        if frame is None or np.asarray(frame).ndim != 3:
            raise ValueError("YOLOPv2 fusion detector requires a BGR frame")
        sequence = self._next_sequence()
        captured_at = self._clock()

        if self._queue is None:
            result = self._infer(sequence, frame, captured_at)
            self._publish_result(result)
        else:
            with self._lock:
                result = self._latest
            if result is None:
                # The runtime calls this once while the wheels are held stopped.
                # A synchronous first result prevents an uninitialized semantic
                # mask when the actual control loop starts.
                result = self._infer(sequence, frame, captured_at)
                self._publish_result(result)
            else:
                self._submit_latest((sequence, frame.copy(), captured_at))

        with self._lock:
            result = self._latest
        self._consumer_result = result
        return result.drivable_mask.copy(), result.lane_mask.copy()

    def fuse_corridor(
        self,
        classical_corridor: np.ndarray,
        semantic_drivable: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Return a corridor no larger than the classical LCC result."""
        classical = (np.asarray(classical_corridor) > 0).astype(np.uint8)
        semantic = (np.asarray(semantic_drivable) > 0).astype(np.uint8)
        if classical.ndim != 2 or semantic.ndim != 2:
            raise ValueError("fusion masks must be two-dimensional")
        if semantic.shape != classical.shape:
            semantic = cv2.resize(
                semantic,
                (classical.shape[1], classical.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        result = self._consumer_result
        if result is None:
            source = (
                "required-unavailable"
                if self.config.required_for_motion
                else "fallback-unavailable"
            )
            state = self._fallback_state(source)
            return self._finish_fusion(classical, state)

        age = max(0.0, self._clock() - result.captured_at)
        base_state = {
            "active": False,
            "motion_allowed": True,
            "source": "unavailable",
            "result_age_seconds": age,
            "overlap_ratio": None,
            "drivable_ratio": float(np.mean(semantic > 0)),
            "inference_seconds": result.inference_seconds,
            "confidence_scale": 1.0,
        }
        if age > self.config.max_result_age_seconds:
            source = "required-stale" if self.config.required_for_motion else "fallback-stale"
            state = dict(base_state, source=source)
            if self.config.required_for_motion:
                state["motion_allowed"] = False
                return self._finish_fusion(np.zeros_like(classical), state)
            return self._finish_fusion(classical, state)

        kernel_size = max(1, int(self.config.drivable_dilate_kernel))
        if kernel_size > 1:
            semantic = cv2.dilate(
                semantic,
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
            )
        corridor_pixels = int(np.count_nonzero(classical))
        overlap_pixels = int(np.count_nonzero((classical > 0) & (semantic > 0)))
        overlap = overlap_pixels / max(1, corridor_pixels)
        base_state["overlap_ratio"] = float(overlap)
        if corridor_pixels == 0 or overlap < self.config.minimum_overlap_ratio:
            source = (
                "required-low-overlap"
                if self.config.required_for_motion
                else "fallback-low-overlap"
            )
            state = dict(base_state, source=source)
            if self.config.required_for_motion:
                state["motion_allowed"] = False
                return self._finish_fusion(np.zeros_like(classical), state)
            return self._finish_fusion(classical, state)

        confidence_scale = 1.0 - self.config.confidence_weight * (1.0 - overlap)
        state = dict(
            base_state,
            active=True,
            source=f"fused-{self.config.fusion_mode}",
            confidence_scale=float(np.clip(confidence_scale, 0.0, 1.0)),
        )
        fused = classical if self.config.fusion_mode == "validate" else classical & semantic
        if not np.any(fused):
            state["source"] = (
                "required-empty"
                if self.config.required_for_motion
                else "fallback-empty"
            )
            state["active"] = False
            if self.config.required_for_motion:
                state["motion_allowed"] = False
                return self._finish_fusion(np.zeros_like(classical), state)
            return self._finish_fusion(classical, state)
        return self._finish_fusion(fused, state)

    def _fallback_state(self, source: str) -> dict:
        state = self._empty_fusion_state(source)
        state["motion_allowed"] = not self.config.required_for_motion
        return state

    def _finish_fusion(self, corridor: np.ndarray, state: dict) -> tuple[np.ndarray, dict]:
        self._last_fusion = dict(state)
        return (np.asarray(corridor) > 0).astype(np.uint8), dict(state)

    def get_state(self) -> dict:
        with self._lock:
            latest = self._latest
            state = {
                "enabled": True,
                "asynchronous": self._queue is not None,
                "optimized_for_inference": bool(
                    self.config.optimize_for_inference
                ),
                "drivable_only": bool(self.config.drivable_only),
                "submitted_frames": self._submitted,
                "completed_frames": self._completed,
                "dropped_frames": self._dropped,
                "queue_depth": 0 if self._queue is None else self._queue.qsize(),
                "error": self._error,
                "latest_sequence": None if latest is None else latest.sequence,
            }
        state["fusion"] = dict(self._last_fusion)
        return state

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is None:
            return
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._queue.put(None)
        self._thread.join()
        self._thread = None
