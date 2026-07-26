"""Propagate YOLOPv2 masks between semantic keyframes using optical flow."""

import cv2
import numpy as np


class TemporalMaskPropagator:
    def __init__(
        self,
        max_steps: int = 3,
        confidence_decay: float = 0.90,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if not 0 < confidence_decay <= 1:
            raise ValueError("confidence_decay must be in (0, 1]")
        self.max_steps = int(max_steps)
        self.confidence_decay = float(confidence_decay)
        self._previous_gray = None
        self._drivable_mask = None
        self._lane_mask = None
        self.steps = 0

    @property
    def initialized(self):
        return self._previous_gray is not None

    @property
    def needs_keyframe(self):
        return not self.initialized or self.steps >= self.max_steps

    @staticmethod
    def _gray_at_mask_size(frame, mask_shape):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = mask_shape
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    def reset(self, frame, drivable_mask, lane_mask):
        drivable = (np.asarray(drivable_mask) > 0).astype(np.uint8)
        lane = (np.asarray(lane_mask) > 0).astype(np.uint8)
        if drivable.ndim != 2 or lane.shape != drivable.shape:
            raise ValueError("drivable and lane masks must be matching 2D arrays")
        self._previous_gray = self._gray_at_mask_size(frame, drivable.shape)
        self._drivable_mask = drivable
        self._lane_mask = lane
        self.steps = 0

    def propagate(self, frame):
        if not self.initialized:
            raise RuntimeError("propagator needs a YOLO keyframe first")
        if self.needs_keyframe:
            raise RuntimeError("maximum propagation steps reached")

        current_gray = self._gray_at_mask_size(
            frame, self._drivable_mask.shape
        )
        backward_flow = cv2.calcOpticalFlowFarneback(
            current_gray,
            self._previous_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        height, width = current_gray.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = grid_x + backward_flow[..., 0]
        map_y = grid_y + backward_flow[..., 1]
        drivable = cv2.remap(
            self._drivable_mask,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        lane = cv2.remap(
            self._lane_mask,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        drivable = cv2.morphologyEx(
            (drivable > 0).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        lane = (lane > 0).astype(np.uint8)

        self._previous_gray = current_gray
        self._drivable_mask = drivable
        self._lane_mask = lane
        self.steps += 1
        confidence = self.confidence_decay ** self.steps
        return drivable.copy(), lane.copy(), float(confidence)
