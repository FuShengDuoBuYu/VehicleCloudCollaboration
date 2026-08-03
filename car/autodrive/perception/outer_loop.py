"""Outer-loop corridor tracking for the fixed yellow-boundary test field."""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class OuterLoopBoundaryConfig:
    enabled: bool = False
    navigation_mode: str = "boundary"
    direction: str = "clockwise"
    outer_boundary_side: str = "left"
    top_ratio: float = 0.24
    bottom_ratio: float = 0.98
    scan_step: int = 3
    band_half_height: int = 2
    minimum_run_width: int = 2
    center_exclusion_ratio: float = 0.06
    expected_lane_width_ratio: float = 0.70
    minimum_lane_width_ratio: float = 0.24
    maximum_lane_width_ratio: float = 0.72
    adaptive_lane_width: bool = True
    maximum_boundary_jump_ratio: float = 0.24
    minimum_observations: int = 10
    polynomial_degree: int = 2
    history_blend: float = 0.20
    maximum_missing_frames: int = 4
    missing_confidence_decay: float = 0.78
    minimum_visible_boundary_ratio: float = 0.01
    boundary_margin_ratio: float = 0.01
    include_lane_mask: bool = True
    yellow_hsv_lower: tuple[int, int, int] = (18, 20, 150)
    yellow_hsv_upper: tuple[int, int, int] = (42, 255, 255)
    yellow_open_kernel: int = 3
    # Raw-camera footprint safety zone. Keep this substantially narrower than
    # the route corridor so a legitimate curved boundary is not mistaken for
    # a line underneath the chassis.
    ego_boundary_top_ratio: float = 0.84
    ego_boundary_half_width_ratio: float = 0.14
    maximum_ego_yellow_ratio: float = 0.020
    green_hsv_lower: tuple[int, int, int] = (45, 55, 35)
    green_hsv_upper: tuple[int, int, int] = (105, 255, 255)
    green_lab_a_max: int = 118
    green_lab_b_min: int = 123
    green_close_kernel: int = 5
    green_dilate_kernel: int = 3

    def __post_init__(self):
        if self.navigation_mode not in {"boundary", "surface"}:
            raise ValueError("navigation_mode must be boundary or surface")
        if self.direction not in {"clockwise", "counterclockwise"}:
            raise ValueError("outer-loop direction must be clockwise or counterclockwise")
        if self.outer_boundary_side not in {"left", "right"}:
            raise ValueError("outer_boundary_side must be left or right")
        if not 0.0 <= self.top_ratio < self.bottom_ratio <= 1.0:
            raise ValueError("expected 0 <= top_ratio < bottom_ratio <= 1")
        if self.scan_step < 1 or self.band_half_height < 0:
            raise ValueError("scan settings must be non-negative")
        if self.minimum_run_width < 1 or self.minimum_observations < 3:
            raise ValueError("boundary observation settings are too small")
        if not 0.0 < self.expected_lane_width_ratio < 1.0:
            raise ValueError("expected_lane_width_ratio must be in (0, 1)")
        if not (
            0.0
            < self.minimum_lane_width_ratio
            < self.maximum_lane_width_ratio
            < 1.0
        ):
            raise ValueError("lane-width limits must be ordered inside (0, 1)")
        if not 0.0 <= self.history_blend < 1.0:
            raise ValueError("history_blend must be in [0, 1)")
        if self.maximum_missing_frames < 0:
            raise ValueError("maximum_missing_frames must not be negative")
        if not 0.0 <= self.minimum_visible_boundary_ratio <= 1.0:
            raise ValueError("minimum_visible_boundary_ratio must be in [0, 1]")
        if not 0.0 <= self.ego_boundary_top_ratio < 1.0:
            raise ValueError("ego_boundary_top_ratio must be in [0, 1)")
        if not 0.0 < self.ego_boundary_half_width_ratio < 0.5:
            raise ValueError("ego_boundary_half_width_ratio must be in (0, 0.5)")
        if not 0.0 <= self.maximum_ego_yellow_ratio <= 1.0:
            raise ValueError("maximum_ego_yellow_ratio must be in [0, 1]")


@dataclass
class BoundaryTrackResult:
    valid: bool
    confidence: float
    corridor_mask: np.ndarray
    source: str
    lane_width_ratio: float = 0.0
    observed_left_rows: int = 0
    observed_right_rows: int = 0
    reason: str = ""
    left_curve: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    right_curve: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    ego_yellow_ratio: float = 0.0
    yellow_hazard: bool = False
    boundary_visible_ratio: float = 0.0


class OuterLoopBoundaryTracker:
    """Build a safe road corridor between the outer and inner boundaries.

    In clockwise mode the continuous outer field boundary stays on the left.
    The inner boundary stays on the right and may contain intentional gaps at
    intersections. Row-wise boundary observations and a temporal lane-width
    prior bridge those gaps instead of allowing the drivable mask to leak into
    an inner branch.
    """

    def __init__(self, config: OuterLoopBoundaryConfig = OuterLoopBoundaryConfig()):
        self.config = config
        self._left_curve: Optional[np.ndarray] = None
        self._right_curve: Optional[np.ndarray] = None
        self._width_curve: Optional[np.ndarray] = None
        self._confidence = 0.0
        self._missing_frames = 0

    def reset(self) -> None:
        self._left_curve = None
        self._right_curve = None
        self._width_curve = None
        self._confidence = 0.0
        self._missing_frames = 0

    def yellow_mask(self, frame: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Extract the field's yellow boundary paint at model-mask resolution."""
        if frame is None or np.asarray(frame).ndim != 3:
            return np.zeros(target_shape, dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.asarray(self.config.yellow_hsv_lower, dtype=np.uint8)
        upper = np.asarray(self.config.yellow_hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel_size = max(1, int(self.config.yellow_open_kernel))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        height, width = target_shape
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    def road_surface_mask(
        self, frame: np.ndarray, target_shape: tuple[int, int]
    ) -> np.ndarray:
        """Return pixels that are not part of the field's green islands."""
        if frame is None or np.asarray(frame).ndim != 3:
            return np.ones(target_shape, dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.asarray(self.config.green_hsv_lower, dtype=np.uint8)
        upper = np.asarray(self.config.green_hsv_upper, dtype=np.uint8)
        green = cv2.inRange(hsv, lower, upper)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_green = (
            (lab[..., 1] <= int(self.config.green_lab_a_max))
            & (lab[..., 2] >= int(self.config.green_lab_b_min))
        ).astype(np.uint8) * 255
        green = cv2.bitwise_or(green, lab_green)
        close_size = max(1, int(self.config.green_close_kernel))
        if close_size > 1:
            green = cv2.morphologyEx(
                green,
                cv2.MORPH_CLOSE,
                np.ones((close_size, close_size), dtype=np.uint8),
            )
        dilate_size = max(1, int(self.config.green_dilate_kernel))
        if dilate_size > 1:
            green = cv2.dilate(
                green,
                np.ones((dilate_size, dilate_size), dtype=np.uint8),
            )
        height, width = target_shape
        resized = cv2.resize(green, (width, height), interpolation=cv2.INTER_NEAREST)
        return (resized == 0).astype(np.uint8)

    def yellow_under_ego(self, yellow_mask: np.ndarray) -> tuple[bool, float]:
        """Detect a yellow boundary entering the raw-image vehicle footprint."""
        yellow = (np.asarray(yellow_mask) > 0).astype(np.uint8)
        if yellow.ndim != 2 or yellow.size == 0:
            return False, 0.0
        height, width = yellow.shape
        top = int(height * self.config.ego_boundary_top_ratio)
        half_width = int(width * self.config.ego_boundary_half_width_ratio)
        left = max(0, width // 2 - half_width)
        right = min(width, width // 2 + half_width)
        region = yellow[top:, left:right]
        ratio = float(np.mean(region > 0)) if region.size else 0.0
        return ratio > self.config.maximum_ego_yellow_ratio, ratio

    @staticmethod
    def _runs(row: np.ndarray, minimum_width: int) -> list[tuple[int, int]]:
        xs = np.flatnonzero(row)
        if xs.size == 0:
            return []
        split_points = np.flatnonzero(np.diff(xs) > 1) + 1
        return [
            (int(run[0]), int(run[-1]))
            for run in np.split(xs, split_points)
            if run.size >= minimum_width
        ]

    def _expected_at(self, curve: Optional[np.ndarray], y: int) -> Optional[float]:
        if curve is None or not 0 <= y < curve.size:
            return None
        return float(curve[y])

    def _scan_boundaries(self, boundary_mask: np.ndarray):
        height, width = boundary_mask.shape
        top = int(height * self.config.top_ratio)
        bottom = min(height - 1, int(height * self.config.bottom_ratio))
        exclusion = width * self.config.center_exclusion_ratio
        max_jump = width * self.config.maximum_boundary_jump_ratio

        left_observations = []
        right_observations = []
        for y in range(bottom, top - 1, -self.config.scan_step):
            y0 = max(0, y - self.config.band_half_height)
            y1 = min(height, y + self.config.band_half_height + 1)
            band = np.any(boundary_mask[y0:y1] > 0, axis=0).astype(np.uint8)
            runs = self._runs(band, self.config.minimum_run_width)
            if not runs:
                continue

            expected_left = self._expected_at(self._left_curve, y)
            expected_right = self._expected_at(self._right_curve, y)
            if expected_left is not None and expected_right is not None:
                split = (expected_left + expected_right) * 0.5
            else:
                split = width * 0.5

            left_candidates = [right for left, right in runs if right < split - exclusion]
            right_candidates = [left for left, right in runs if left > split + exclusion]

            if left_candidates:
                if expected_left is None:
                    left_x = max(left_candidates)
                else:
                    left_x = min(left_candidates, key=lambda x: abs(x - expected_left))
                    if abs(left_x - expected_left) > max_jump:
                        left_x = None
                if left_x is not None:
                    left_observations.append((y, float(left_x)))

            if right_candidates:
                if expected_right is None:
                    right_x = min(right_candidates)
                else:
                    right_x = min(right_candidates, key=lambda x: abs(x - expected_right))
                    if abs(right_x - expected_right) > max_jump:
                        right_x = None
                if right_x is not None:
                    right_observations.append((y, float(right_x)))

        return left_observations, right_observations

    def _fit_curve(self, observations, height: int, width: int):
        if len(observations) < self.config.minimum_observations:
            return None, 0.0
        values = np.asarray(observations, dtype=np.float32)
        ys, xs = values.T
        normalized_y = (height - 1 - ys) / max(1.0, float(height - 1))
        degree = min(self.config.polynomial_degree, len(observations) - 1)
        inliers = np.ones(len(observations), dtype=bool)
        coefficients = None
        for _ in range(3):
            if np.count_nonzero(inliers) < self.config.minimum_observations:
                return None, 0.0
            coefficients = np.polyfit(normalized_y[inliers], xs[inliers], degree)
            predicted = np.polyval(coefficients, normalized_y)
            residuals = np.abs(xs - predicted)
            residual_limit = max(3.0, float(np.median(residuals[inliers]) * 3.0 + 1.0))
            next_inliers = residuals <= residual_limit
            if np.array_equal(next_inliers, inliers):
                break
            inliers = next_inliers

        all_y = np.arange(height, dtype=np.float32)
        all_normalized = (height - 1 - all_y) / max(1.0, float(height - 1))
        curve = np.polyval(coefficients, all_normalized).astype(np.float32)
        curve = np.clip(curve, 0.0, float(width - 1))
        predicted = np.polyval(coefficients, normalized_y[inliers])
        mean_residual = float(np.mean(np.abs(xs[inliers] - predicted)))
        scan_rows = max(
            1,
            int(
                (self.config.bottom_ratio - self.config.top_ratio)
                * height
                / self.config.scan_step
            ),
        )
        coverage = min(1.0, np.count_nonzero(inliers) / scan_rows)
        fit_quality = float(np.exp(-mean_residual / max(2.0, width * 0.025)))
        return curve, float(np.clip(0.55 * coverage + 0.45 * fit_quality, 0.0, 1.0))

    def _blend(self, current: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
        if previous is None or previous.shape != current.shape:
            return current
        previous_weight = self.config.history_blend
        return (
            current * (1.0 - previous_weight) + previous * previous_weight
        ).astype(np.float32)

    def _default_width(self, height: int, width: int) -> np.ndarray:
        return np.full(
            height,
            width * self.config.expected_lane_width_ratio,
            dtype=np.float32,
        )

    def _make_corridor(
        self,
        drivable_mask: np.ndarray,
        left_curve: np.ndarray,
        right_curve: np.ndarray,
    ) -> np.ndarray:
        height, width = drivable_mask.shape
        top = int(height * self.config.top_ratio)
        bottom = min(height - 1, int(height * self.config.bottom_ratio))
        margin = width * self.config.boundary_margin_ratio
        corridor = np.zeros((height, width), dtype=np.uint8)
        for y in range(top, bottom + 1):
            left = int(np.ceil(left_curve[y] + margin))
            right = int(np.floor(right_curve[y] - margin))
            left = max(0, min(width - 1, left))
            right = max(0, min(width - 1, right))
            if right > left:
                corridor[y, left : right + 1] = 1
        return corridor & (np.asarray(drivable_mask) > 0).astype(np.uint8)

    def _history_result(
        self,
        drivable_mask: np.ndarray,
        reason: str,
        boundary_visible_ratio: float = 0.0,
    ):
        boundary_still_visible = bool(
            boundary_visible_ratio
            >= self.config.minimum_visible_boundary_ratio
        )
        if (
            self._left_curve is None
            or self._right_curve is None
            or (
                not boundary_still_visible
                and self._missing_frames >= self.config.maximum_missing_frames
            )
        ):
            self._missing_frames += 1
            return BoundaryTrackResult(
                False,
                0.0,
                np.zeros_like(drivable_mask, dtype=np.uint8),
                "missing",
                reason=reason,
                boundary_visible_ratio=boundary_visible_ratio,
            )
        self._missing_frames += 1
        self._confidence *= self.config.missing_confidence_decay
        corridor = self._make_corridor(
            drivable_mask, self._left_curve, self._right_curve
        )
        lane_width = float(np.median(self._right_curve - self._left_curve))
        return BoundaryTrackResult(
            bool(np.any(corridor)),
            self._confidence,
            corridor,
            "visible-history" if boundary_still_visible else "history",
            lane_width / max(1.0, drivable_mask.shape[1]),
            reason=(
                "boundary remains visible outside x(y) fit"
                if boundary_still_visible
                else "using recent boundary history"
            ),
            left_curve=self._left_curve.copy(),
            right_curve=self._right_curve.copy(),
            boundary_visible_ratio=boundary_visible_ratio,
        )

    def update(
        self,
        drivable_mask: np.ndarray,
        lane_mask: np.ndarray,
        yellow_mask: Optional[np.ndarray] = None,
    ) -> BoundaryTrackResult:
        drivable = (np.asarray(drivable_mask) > 0).astype(np.uint8)
        lane = (np.asarray(lane_mask) > 0).astype(np.uint8)
        if drivable.ndim != 2 or lane.shape != drivable.shape:
            raise ValueError("drivable and lane masks must be same-shaped 2D arrays")
        boundary = (
            lane.copy()
            if self.config.include_lane_mask
            else np.zeros_like(lane, dtype=np.uint8)
        )
        if yellow_mask is not None:
            yellow = (np.asarray(yellow_mask) > 0).astype(np.uint8)
            if yellow.shape != drivable.shape:
                raise ValueError("yellow mask must match the model masks")
            boundary |= yellow

        height, width = drivable.shape
        boundary_visible_ratio = float(np.mean(boundary > 0))
        left_obs, right_obs = self._scan_boundaries(boundary)
        left_curve, left_confidence = self._fit_curve(left_obs, height, width)
        right_curve, right_confidence = self._fit_curve(right_obs, height, width)
        if left_curve is None and right_curve is None:
            return self._history_result(
                drivable,
                "outer-loop boundaries are missing",
                boundary_visible_ratio,
            )

        source = "both"
        width_curve = self._width_curve
        if width_curve is None:
            width_curve = self._default_width(height, width)

        if left_curve is not None and right_curve is not None:
            measured_width = right_curve - left_curve
            valid_width = (
                (measured_width >= width * self.config.minimum_lane_width_ratio)
                & (measured_width <= width * self.config.maximum_lane_width_ratio)
            )
            if np.count_nonzero(valid_width) >= height // 4:
                replacement = float(np.median(measured_width[valid_width]))
                measured_width = np.where(valid_width, measured_width, replacement)
                if self.config.adaptive_lane_width:
                    width_curve = self._blend(
                        measured_width.astype(np.float32),
                        width_curve,
                    )
                else:
                    # Preserve the calibrated prior for a later one-edge
                    # dropout, but do not replace either freshly fitted edge.
                    # The real pair and its midpoint are the authoritative
                    # straight-road geometry for the current frame.
                    width_curve = self._default_width(height, width)
            else:
                # A pair outside the calibrated physical width is not trusted:
                # retain only the continuous outer boundary and infer the
                # missing side below.
                if self.config.outer_boundary_side == "left":
                    right_curve = None
                else:
                    left_curve = None

        if left_curve is None:
            source = "inner+width"
            right_curve = self._blend(right_curve, self._right_curve)
            left_curve = right_curve - width_curve
            confidence = right_confidence * 0.68
        elif right_curve is None:
            source = "outer+width"
            left_curve = self._blend(left_curve, self._left_curve)
            right_curve = left_curve + width_curve
            confidence = left_confidence * 0.78
        else:
            left_curve = self._blend(left_curve, self._left_curve)
            right_curve = self._blend(right_curve, self._right_curve)
            confidence = 0.5 * (left_confidence + right_confidence)

        if not self.config.adaptive_lane_width:
            # Keep real two-boundary geometry untouched. Re-anchor only the
            # inferred side of a one-boundary result so temporal smoothing and
            # image-edge clipping cannot shrink the calibrated fallback width.
            width_curve = self._default_width(height, width)
            if source == "outer+width":
                right_curve = left_curve + width_curve
            elif source == "inner+width":
                left_curve = right_curve - width_curve

        minimum_width = width * self.config.minimum_lane_width_ratio
        maximum_width = width * self.config.maximum_lane_width_ratio
        center_curve = (left_curve + right_curve) * 0.5
        lane_width = np.clip(right_curve - left_curve, minimum_width, maximum_width)
        left_curve = np.clip(center_curve - lane_width * 0.5, 0.0, width - 1.0)
        right_curve = np.clip(center_curve + lane_width * 0.5, 0.0, width - 1.0)
        corridor = self._make_corridor(drivable, left_curve, right_curve)
        if not np.any(corridor):
            return self._history_result(
                drivable,
                "outer-loop corridor is empty",
                boundary_visible_ratio,
            )

        self._left_curve = left_curve.astype(np.float32)
        self._right_curve = right_curve.astype(np.float32)
        self._width_curve = (
            lane_width.astype(np.float32)
            if self.config.adaptive_lane_width
            else self._default_width(height, width)
        )
        self._confidence = float(np.clip(confidence, 0.0, 1.0))
        self._missing_frames = 0
        middle_start = int(height * self.config.top_ratio)
        middle_end = int(height * self.config.bottom_ratio) + 1
        width_ratio = float(
            np.median(lane_width[middle_start:middle_end]) / max(1.0, width)
        )
        return BoundaryTrackResult(
            True,
            self._confidence,
            corridor,
            source,
            width_ratio,
            len(left_obs),
            len(right_obs),
            reason="outer-loop corridor tracked",
            left_curve=self._left_curve.copy(),
            right_curve=self._right_curve.copy(),
            boundary_visible_ratio=boundary_visible_ratio,
        )
