"""Road-center estimation and lane-centering control without hardware dependencies."""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class LaneEstimate:
    valid: bool
    confidence: float
    lateral_error: float = 0.0
    heading_error: float = 0.0
    lookahead_point: Optional[tuple[int, int]] = None
    centerline: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    left_boundary: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    right_boundary: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    reason: str = ""


@dataclass(frozen=True)
class LCCConfig:
    """Normalized differential-drive controller settings.

    Positive steering means turning right. Wheel speeds are normalized to
    ``[-1, 1]`` and can later be converted to the vehicle PWM range.
    """

    base_speed: float = 0.34
    min_confidence: float = 0.30
    lateral_gain: float = 0.72
    heading_gain: float = 0.92
    derivative_gain: float = 0.08
    steering_limit: float = 0.75
    steering_speed_gain: float = 0.72
    turn_slowdown: float = 0.38
    maximum_lateral_error: float = 1.0
    maximum_heading_error: float = 1.0
    steering_smoothing: float = 0.0


@dataclass(frozen=True)
class DifferentialDriveCommand:
    action: str
    steering: float
    left_speed: float
    right_speed: float
    confidence: float
    reason: str = ""

    def as_pwm(self, maximum: int = 100) -> tuple[int, int]:
        """Scale the normalized proposal; ``maximum`` must be calibrated on-car."""
        maximum = max(1, int(maximum))
        left = int(round(np.clip(self.left_speed, -1.0, 1.0) * maximum))
        right = int(round(np.clip(self.right_speed, -1.0, 1.0) * maximum))
        return left, right


class RoadCenterlineEstimator:
    """Extract a locally continuous road centerline from a drivable-area mask."""

    def __init__(
        self,
        top_ratio: float = 0.48,
        bottom_ratio: float = 0.94,
        lookahead_ratio: float = 0.64,
        sample_count: int = 24,
        minimum_width_ratio: float = 0.06,
        route_hint_bias: float = 0.15,
    ):
        if not 0.0 < top_ratio < lookahead_ratio < bottom_ratio <= 1.0:
            raise ValueError("expected top_ratio < lookahead_ratio < bottom_ratio")
        if not 0.0 <= route_hint_bias < 0.5:
            raise ValueError("route_hint_bias must be in [0, 0.5)")
        self.top_ratio = float(top_ratio)
        self.bottom_ratio = float(bottom_ratio)
        self.lookahead_ratio = float(lookahead_ratio)
        self.sample_count = max(8, int(sample_count))
        self.minimum_width_ratio = float(minimum_width_ratio)
        self.route_hint_bias = float(route_hint_bias)

    @staticmethod
    def _segments(row: np.ndarray, minimum_width: int) -> list[tuple[int, int]]:
        xs = np.flatnonzero(row)
        if xs.size == 0:
            return []
        split_points = np.flatnonzero(np.diff(xs) > 1) + 1
        runs = np.split(xs, split_points)
        return [
            (int(run[0]), int(run[-1]))
            for run in runs
            if run.size >= minimum_width
        ]

    @staticmethod
    def _select_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
        binary = (mask > 0).astype(np.uint8)
        h, w = binary.shape
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count <= 1:
            return binary, 0.0

        bottom_start = max(0, int(h * 0.82))
        center_left, center_right = int(w * 0.38), int(w * 0.62)
        best_label = 0
        best_score = -1.0
        best_anchor = 0.0
        for label in range(1, count):
            component = labels == label
            area_ratio = float(stats[label, cv2.CC_STAT_AREA]) / float(h * w)
            lower_ratio = float(np.mean(component[bottom_start:, :]))
            anchor_ratio = float(
                np.mean(component[bottom_start:, center_left:center_right])
            )
            score = 8.0 * anchor_ratio + 2.0 * lower_ratio + area_ratio
            if score > best_score:
                best_score = score
                best_label = label
                best_anchor = anchor_ratio
        return (labels == best_label).astype(np.uint8), best_anchor

    def _choose_segment(
        self,
        segments: list[tuple[int, int]],
        previous_center: float,
        width: int,
        route_hint: str,
    ) -> tuple[int, int]:
        centers = np.array([(left + right) * 0.5 for left, right in segments])
        distance = np.abs(centers - previous_center)
        continuity_weight = 1.0 - self.route_hint_bias
        if route_hint == "left":
            score = continuity_weight * distance + self.route_hint_bias * centers
        elif route_hint == "right":
            score = (
                continuity_weight * distance
                + self.route_hint_bias * (width - centers)
            )
        else:
            score = distance
        return segments[int(np.argmin(score))]

    def estimate(
        self,
        drivable_mask: np.ndarray,
        lane_mask: Optional[np.ndarray] = None,
        route_hint: str = "center",
    ) -> LaneEstimate:
        if route_hint not in {"left", "center", "right"}:
            raise ValueError("route_hint must be left, center, or right")
        if drivable_mask is None or np.asarray(drivable_mask).ndim != 2:
            return LaneEstimate(False, 0.0, reason="invalid drivable-area mask")

        raw_mask = (np.asarray(drivable_mask) > 0).astype(np.uint8)
        h, w = raw_mask.shape
        if h < 16 or w < 16 or not np.any(raw_mask):
            return LaneEstimate(False, 0.0, reason="drivable area is empty")

        road, anchor_ratio = self._select_component(raw_mask)
        minimum_width = max(3, int(w * self.minimum_width_ratio))
        y_values = np.linspace(
            int(h * self.bottom_ratio),
            int(h * self.top_ratio),
            self.sample_count,
        ).astype(np.int32)

        previous_center = w * 0.5
        samples = []
        for y in y_values:
            y0, y1 = max(0, int(y) - 1), min(h, int(y) + 2)
            band = np.any(road[y0:y1] > 0, axis=0).astype(np.uint8)
            segments = self._segments(band, minimum_width)
            if not segments:
                continue
            left, right = self._choose_segment(
                segments, previous_center, w, route_hint
            )
            center = (left + right) * 0.5
            samples.append((float(y), float(left), float(right), center))
            previous_center = center

        minimum_samples = max(5, self.sample_count // 3)
        if len(samples) < minimum_samples:
            return LaneEstimate(
                False,
                min(0.2, len(samples) / self.sample_count),
                reason="too few valid road rows",
            )

        values = np.asarray(samples, dtype=np.float32)
        ys, lefts, rights, centers = values.T
        widths = rights - lefts
        normalized_y = (h - ys) / max(1.0, float(h))
        degree = 2 if len(samples) >= 8 else 1
        weights = np.linspace(1.7, 0.8, len(samples))
        coefficients = np.polyfit(normalized_y, centers, degree, w=weights)
        fitted_centers = np.polyval(coefficients, normalized_y)

        residual = np.abs(centers - fitted_centers)
        residual_limit = max(4.0, float(np.median(residual) * 3.0 + 2.0))
        inliers = residual <= residual_limit
        if np.count_nonzero(inliers) >= minimum_samples:
            coefficients = np.polyfit(
                normalized_y[inliers],
                centers[inliers],
                degree,
                w=weights[inliers],
            )
            fitted_centers = np.polyval(coefficients, normalized_y)

        near_y = int(h * 0.88)
        lookahead_y = int(h * self.lookahead_ratio)

        def fitted_x(y: int) -> float:
            return float(np.polyval(coefficients, (h - y) / max(1.0, float(h))))

        near_x = fitted_x(near_y)
        lookahead_x = fitted_x(lookahead_y)
        lateral_error = (near_x - w * 0.5) / max(1.0, w * 0.5)
        heading_error = (lookahead_x - near_x) / max(1.0, w * 0.5)

        coverage = len(samples) / self.sample_count
        fit_quality = float(
            np.exp(-np.mean(np.abs(centers - fitted_centers)) / max(3.0, w * 0.06))
        )
        width_stability = float(
            np.exp(-np.std(widths) / max(2.0, float(np.mean(widths))))
        )
        anchor_quality = float(np.clip(anchor_ratio / 0.45, 0.0, 1.0))
        confidence = float(
            np.clip(
                0.35 * coverage
                + 0.25 * fit_quality
                + 0.20 * width_stability
                + 0.20 * anchor_quality,
                0.0,
                1.0,
            )
        )

        fitted_int = np.column_stack(
            [np.clip(fitted_centers, 0, w - 1), ys]
        ).astype(np.int32)
        left_points = np.column_stack([lefts, ys]).astype(np.int32)
        right_points = np.column_stack([rights, ys]).astype(np.int32)
        return LaneEstimate(
            valid=True,
            confidence=confidence,
            lateral_error=float(np.clip(lateral_error, -1.0, 1.0)),
            heading_error=float(np.clip(heading_error, -1.0, 1.0)),
            lookahead_point=(
                int(np.clip(lookahead_x, 0, w - 1)),
                lookahead_y,
            ),
            centerline=fitted_int,
            left_boundary=left_points,
            right_boundary=right_points,
            reason="ok",
        )


class LaneCenteringController:
    """Convert road-center errors into normalized differential wheel speeds."""

    def __init__(self, config: LCCConfig = LCCConfig()):
        self.config = config
        self._previous_lateral_error: Optional[float] = None
        self._previous_steering: Optional[float] = None

    def reset(self) -> None:
        self._previous_lateral_error = None
        self._previous_steering = None

    def update(self, estimate: LaneEstimate, dt: Optional[float] = None) -> DifferentialDriveCommand:
        if not estimate.valid:
            self.reset()
            return DifferentialDriveCommand(
                "stop", 0.0, 0.0, 0.0, estimate.confidence, estimate.reason
            )
        if estimate.confidence < self.config.min_confidence:
            self.reset()
            return DifferentialDriveCommand(
                "stop",
                0.0,
                0.0,
                0.0,
                estimate.confidence,
                "road confidence below safety threshold",
            )
        if abs(estimate.lateral_error) > self.config.maximum_lateral_error:
            self.reset()
            return DifferentialDriveCommand(
                "stop",
                0.0,
                0.0,
                0.0,
                estimate.confidence,
                "lateral error exceeds recovery limit",
            )
        if abs(estimate.heading_error) > self.config.maximum_heading_error:
            self.reset()
            return DifferentialDriveCommand(
                "stop",
                0.0,
                0.0,
                0.0,
                estimate.confidence,
                "heading error exceeds recovery limit",
            )

        derivative = 0.0
        if self._previous_lateral_error is not None and dt and dt > 1e-3:
            derivative = (
                estimate.lateral_error - self._previous_lateral_error
            ) / dt
        self._previous_lateral_error = estimate.lateral_error

        raw_steering = (
            self.config.lateral_gain * estimate.lateral_error
            + self.config.heading_gain * estimate.heading_error
            + self.config.derivative_gain * derivative
        )
        steering = float(
            np.clip(
                raw_steering,
                -self.config.steering_limit,
                self.config.steering_limit,
            )
        )
        if self._previous_steering is not None:
            previous_weight = float(
                np.clip(self.config.steering_smoothing, 0.0, 0.95)
            )
            steering = float(
                previous_weight * self._previous_steering
                + (1.0 - previous_weight) * steering
            )
        self._previous_steering = steering
        base_speed = self.config.base_speed * (
            1.0 - self.config.turn_slowdown * abs(steering)
        )
        differential = steering * self.config.steering_speed_gain
        left_speed = float(np.clip(base_speed + differential, -1.0, 1.0))
        right_speed = float(np.clip(base_speed - differential, -1.0, 1.0))

        if steering > 0.08:
            action = "turn-right"
        elif steering < -0.08:
            action = "turn-left"
        else:
            action = "forward"
        return DifferentialDriveCommand(
            action,
            steering,
            left_speed,
            right_speed,
            estimate.confidence,
            "ok",
        )
