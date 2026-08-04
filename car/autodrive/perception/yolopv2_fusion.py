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
    backend: str = "torchscript"
    weights: str = ""
    device: str = "cpu"
    img_size: int = 320
    fast_mask: bool = True
    optimize_for_inference: bool = True
    drivable_only: bool = True
    torch_num_threads: int = 1
    torch_interop_threads: int = 1
    onnx_intra_op_threads: int = 1
    adaptive_precision: bool = False
    int8_weights: str = ""
    straight_enter_frames: int = 8
    straight_max_abs_steering: float = 0.08
    straight_min_confidence: float = 0.55
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
        if self.backend not in {"torchscript", "onnxruntime"}:
            raise ValueError("YOLOPv2 backend must be torchscript or onnxruntime")
        if self.torch_num_threads < 0 or self.torch_interop_threads < 0:
            raise ValueError("YOLOPv2 torch thread counts must not be negative")
        if self.onnx_intra_op_threads < 1:
            raise ValueError("YOLOPv2 ONNX thread count must be positive")
        if self.adaptive_precision and self.backend != "onnxruntime":
            raise ValueError("YOLOPv2 adaptive precision requires onnxruntime")
        if self.straight_enter_frames < 1:
            raise ValueError("YOLOPv2 straight enter frames must be positive")
        if not 0.0 <= self.straight_max_abs_steering <= 1.0:
            raise ValueError("YOLOPv2 straight steering limit must be in [0, 1]")
        if not 0.0 <= self.straight_min_confidence <= 1.0:
            raise ValueError("YOLOPv2 straight confidence must be in [0, 1]")
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
    precision: str
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
        int8_model: Optional[Any] = None,
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
            if config.backend == "onnxruntime":
                from .yolopv2_onnx import YOLOPv2ONNXDetector

                model = YOLOPv2ONNXDetector(
                    weights=str(weights),
                    img_size=int(config.img_size),
                    intra_op_threads=int(config.onnx_intra_op_threads),
                    fast_mask=bool(config.fast_mask),
                )
            else:
                # Keep torch out of module import so classical-only tools stay
                # lightweight and ONNX deployment does not allocate PyTorch.
                import torch

                if config.torch_num_threads > 0:
                    torch.set_num_threads(int(config.torch_num_threads))
                if config.torch_interop_threads > 0:
                    try:
                        torch.set_num_interop_threads(
                            int(config.torch_interop_threads)
                        )
                    except RuntimeError:
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
        if config.adaptive_precision and int8_model is None:
            int8_weights = Path(config.int8_weights).expanduser()
            if not int8_weights.is_file():
                raise FileNotFoundError(
                    f"YOLOPv2 INT8 weights not found: {int8_weights}"
                )
            from .yolopv2_onnx import YOLOPv2ONNXDetector

            int8_model = YOLOPv2ONNXDetector(
                weights=str(int8_weights),
                img_size=int(config.img_size),
                intra_op_threads=int(config.onnx_intra_op_threads),
                fast_mask=bool(config.fast_mask),
            )
        self.model = model
        self.int8_model = int8_model
        self._models = {"fp32": model}
        if int8_model is not None:
            self._models["int8"] = int8_model
        self._lock = threading.Lock()
        self._latest: Optional[_MaskResult] = None
        self._consumer_result: Optional[_MaskResult] = None
        self._last_fusion = self._empty_fusion_state("unavailable")
        self._error: Optional[str] = None
        self._sequence = 0
        self._submitted = 0
        self._completed = 0
        self._submitted_by_precision = {"fp32": 0, "int8": 0}
        self._completed_by_precision = {"fp32": 0, "int8": 0}
        self._dropped = 0
        self._requested_precision = "fp32"
        self._straight_valid_frames = 0
        self._precision_switches = 0
        self._last_switch_reason = "startup defaults to FP32"
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
            "precision": None,
            "requested_precision": "fp32",
            "confidence_scale": 1.0,
        }

    def _set_requested_precision(self, precision: str, reason: str) -> None:
        if precision not in self._models:
            precision = "fp32"
            reason = f"{reason}; requested model unavailable"
        if precision != self._requested_precision:
            self._precision_switches += 1
        self._requested_precision = precision
        self._last_switch_reason = reason

    def observe_boundary(self, boundary_result: Any) -> None:
        """Immediately reject INT8 when current boundary evidence is not straight."""
        if not self.config.adaptive_precision:
            return
        valid = bool(boundary_result is not None and boundary_result.valid)
        source = None if boundary_result is None else boundary_result.source
        confidence = (
            0.0 if boundary_result is None else float(boundary_result.confidence)
        )
        yellow_hazard = bool(
            boundary_result is not None and boundary_result.yellow_hazard
        )
        if (
            not valid
            or source != "both"
            or confidence < self.config.straight_min_confidence
            or yellow_hazard
        ):
            with self._lock:
                self._straight_valid_frames = 0
                self._set_requested_precision(
                    "fp32",
                    f"boundary={source or 'missing'} confidence={confidence:.3f}",
                )

    def update_route_state(self, boundary_result: Any, command: Any) -> None:
        """Enter INT8 only after a stable straight; leave it in one frame."""
        if not self.config.adaptive_precision:
            return
        confidence = (
            0.0 if boundary_result is None else float(boundary_result.confidence)
        )
        steering = 1.0 if command is None else abs(float(command.steering))
        eligible = bool(
            boundary_result is not None
            and boundary_result.valid
            and boundary_result.source == "both"
            and not boundary_result.yellow_hazard
            and confidence >= self.config.straight_min_confidence
            and command is not None
            and command.action == "forward"
            and steering <= self.config.straight_max_abs_steering
        )
        with self._lock:
            if eligible:
                self._straight_valid_frames += 1
                if (
                    self._straight_valid_frames
                    >= self.config.straight_enter_frames
                ):
                    self._set_requested_precision(
                        "int8",
                        (
                            f"stable straight for "
                            f"{self._straight_valid_frames} frames"
                        ),
                    )
            else:
                self._straight_valid_frames = 0
                action = "missing" if command is None else command.action
                self._set_requested_precision(
                    "fp32",
                    f"action={action} steering={steering:.3f}",
                )

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

    def _infer(
        self,
        sequence: int,
        frame: np.ndarray,
        captured_at: float,
        precision: str,
    ) -> _MaskResult:
        started = self._clock()
        drivable, lane = self._models[precision].predict_masks(frame)
        completed = self._clock()
        return _MaskResult(
            sequence=sequence,
            captured_at=float(captured_at),
            completed_at=float(completed),
            inference_seconds=float(completed - started),
            precision=precision,
            drivable_mask=self._resize_mask(drivable),
            lane_mask=self._resize_mask(lane),
        )

    def _publish_result(self, result: _MaskResult) -> None:
        with self._lock:
            self._latest = result
            self._completed += 1
            self._completed_by_precision[result.precision] += 1
            self._error = None

    def _run_job(self, job) -> None:
        sequence, frame, captured_at, precision = job
        try:
            self._publish_result(
                self._infer(sequence, frame, captured_at, precision)
            )
        except Exception as exc:  # surfaced through state; old safe result expires
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                if precision == "int8":
                    self._straight_valid_frames = 0
                    self._set_requested_precision(
                        "fp32", "INT8 inference failed"
                    )

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
                self._submitted_by_precision[job[3]] += 1
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
                self._submitted_by_precision[job[3]] += 1
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
        with self._lock:
            precision = self._requested_precision

        if self._queue is None:
            result = self._infer(sequence, frame, captured_at, precision)
            self._publish_result(result)
        else:
            with self._lock:
                result = self._latest
            if result is None:
                # The runtime calls this once while the wheels are held stopped.
                # A synchronous first result prevents an uninitialized semantic
                # mask when the actual control loop starts.
                result = self._infer(sequence, frame, captured_at, precision)
                self._publish_result(result)
            else:
                self._submit_latest(
                    (sequence, frame.copy(), captured_at, precision)
                )

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

        with self._lock:
            requested_precision = self._requested_precision
        age = max(0.0, self._clock() - result.captured_at)
        base_state = {
            "active": False,
            "motion_allowed": True,
            "source": "unavailable",
            "result_age_seconds": age,
            "overlap_ratio": None,
            "drivable_ratio": float(np.mean(semantic > 0)),
            "inference_seconds": result.inference_seconds,
            "precision": result.precision,
            "requested_precision": requested_precision,
            "confidence_scale": 1.0,
        }
        if result.precision == "int8" and requested_precision == "fp32":
            source = (
                "required-precision-transition"
                if self.config.required_for_motion
                else "fallback-precision-transition"
            )
            state = dict(base_state, source=source)
            if self.config.required_for_motion:
                state["motion_allowed"] = False
                return self._finish_fusion(np.zeros_like(classical), state)
            return self._finish_fusion(classical, state)
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
        with self._lock:
            state["requested_precision"] = self._requested_precision
        return state

    def _finish_fusion(self, corridor: np.ndarray, state: dict) -> tuple[np.ndarray, dict]:
        self._last_fusion = dict(state)
        return (np.asarray(corridor) > 0).astype(np.uint8), dict(state)

    def get_state(self) -> dict:
        with self._lock:
            latest = self._latest
            state = {
                "enabled": True,
                "backend": self.config.backend,
                "adaptive_precision": self.config.adaptive_precision,
                "straight_enter_frames": self.config.straight_enter_frames,
                "straight_max_abs_steering": (
                    self.config.straight_max_abs_steering
                ),
                "straight_min_confidence": (
                    self.config.straight_min_confidence
                ),
                "asynchronous": self._queue is not None,
                "optimized_for_inference": bool(
                    self.config.optimize_for_inference
                ),
                "drivable_only": bool(self.config.drivable_only),
                "submitted_frames": self._submitted,
                "completed_frames": self._completed,
                "submitted_by_precision": dict(
                    self._submitted_by_precision
                ),
                "completed_by_precision": dict(
                    self._completed_by_precision
                ),
                "dropped_frames": self._dropped,
                "queue_depth": 0 if self._queue is None else self._queue.qsize(),
                "error": self._error,
                "latest_sequence": None if latest is None else latest.sequence,
                "latest_precision": (
                    None if latest is None else latest.precision
                ),
                "requested_precision": self._requested_precision,
                "straight_valid_frames": self._straight_valid_frames,
                "precision_switches": self._precision_switches,
                "last_switch_reason": self._last_switch_reason,
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
