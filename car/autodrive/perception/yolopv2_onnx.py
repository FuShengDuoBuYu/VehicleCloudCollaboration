"""CPU-only ONNX Runtime inference for the YOLOPv2 drivable-area head."""

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


class YOLOPv2ONNXDetector:
    """Run a fixed-size, drivable-only YOLOPv2 ONNX graph on CPU."""

    def __init__(
        self,
        weights: str,
        img_size: int = 320,
        intra_op_threads: int = 1,
        fast_mask: bool = True,
        session: Optional[Any] = None,
    ):
        self.weights = str(weights)
        self.img_size = int(img_size)
        self.fast_mask = bool(fast_mask)
        if self.img_size < 32:
            raise ValueError("YOLOPv2 ONNX img_size must be at least 32")
        if intra_op_threads < 1:
            raise ValueError("YOLOPv2 ONNX intra-op threads must be positive")

        if session is None:
            model_path = Path(self.weights).expanduser()
            if not model_path.is_file():
                raise FileNotFoundError(f"YOLOPv2 ONNX weights not found: {model_path}")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "ONNX Runtime is required for backend=onnxruntime"
                ) from exc
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(intra_op_threads)
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session = ort.InferenceSession(
                str(model_path),
                options,
                providers=["CPUExecutionProvider"],
            )
        self.session = session
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, tuple]:
        height, width = image.shape[:2]
        scale = min(self.img_size / height, self.img_size / width)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        pad_width = (self.img_size - resized_width) % 32
        pad_height = (self.img_size - resized_height) % 32
        half_width = pad_width / 2
        half_height = pad_height / 2
        if (width, height) != (resized_width, resized_height):
            image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
        top = int(round(half_height - 0.1))
        bottom = int(round(half_height + 0.1))
        left = int(round(half_width - 0.1))
        right = int(round(half_width + 0.1))
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return image, (top, bottom, left, right)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, tuple]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("YOLOPv2 ONNX requires a BGR image")
        # Match the preprocessing embedded in the existing TorchScript path.
        image = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        image, padding = self._letterbox(image)
        tensor = np.ascontiguousarray(
            image[:, :, ::-1].transpose(2, 0, 1),
            dtype=np.float32,
        )
        tensor /= 255.0
        return tensor[None], padding

    def predict_masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, padding = self._preprocess(frame)
        segmentation = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )[0]
        if self.fast_mask:
            top, bottom, left, right = padding
            height_end = segmentation.shape[2] - bottom if bottom else segmentation.shape[2]
            width_end = segmentation.shape[3] - right if right else segmentation.shape[3]
            segmentation = segmentation[
                :, :, top:height_end, left:width_end
            ]
        mask = np.argmax(segmentation, axis=1).astype(np.uint8).squeeze()
        return mask, np.zeros_like(mask)
