"""Road-center estimation and lane-centering control without hardware dependencies."""

from __future__ import annotations

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
    near_heading_error: float = 0.0
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
    tight_turn_near_heading_start: float = 0.40
    tight_turn_near_heading_full: float = 0.65

    def __post_init__(self):
        if not (
            0.0
            <= self.tight_turn_near_heading_start
            < self.tight_turn_near_heading_full
            <= 1.0
        ):
            raise ValueError(
                "tight-turn near-heading thresholds must be ordered in [0, 1]"
            )


@dataclass(frozen=True)
class DifferentialDriveCommand:
    action: str
    steering: float
    left_speed: float
    right_speed: float
    confidence: float
    reason: str = ""
    tight_turn_factor: Optional[float] = None

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
        tight_turn_lookahead_ratio: float = 0.72,
        sample_count: int = 24,
        minimum_width_ratio: float = 0.06,
        route_hint_bias: float = 0.15,
    ):
        if not (
            0.0
            < top_ratio
            < lookahead_ratio
            < tight_turn_lookahead_ratio
            < bottom_ratio
            <= 1.0
        ):
            raise ValueError(
                "expected top_ratio < lookahead_ratio < "
                "tight_turn_lookahead_ratio < bottom_ratio"
            )
        if not 0.0 <= route_hint_bias < 0.5:
            raise ValueError("route_hint_bias must be in [0, 0.5)")
        self.top_ratio = float(top_ratio)
        self.bottom_ratio = float(bottom_ratio)
        self.lookahead_ratio = float(lookahead_ratio)
        self.tight_turn_lookahead_ratio = float(tight_turn_lookahead_ratio)
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
    def _select_component(mask: np.ndarray, preserve_exclusions: bool = False) -> tuple[np.ndarray, float]:
        binary = (mask > 0).astype(np.uint8)
        h, w = binary.shape
        kernel = np.ones((3, 3), dtype=np.uint8)
        if not preserve_exclusions:
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

    def estimate_from_boundaries(
        self,
        left_curve: np.ndarray,
        right_curve: np.ndarray,
        image_width: int,
    ) -> LaneEstimate:
        """Build the centerline directly from a freshly tracked lane pair.

        ``OuterLoopBoundaryTracker`` has already fitted and continuity-checked
        these curves. Requiring their surface-clipped corridor to form one
        bottom-centred connected component repeats the same geometry test and
        can falsely stop on dark seams or lighting changes in the road mat.
        This path retains the fitted yellow-boundary geometry while the
        tracker remains responsible for rejecting an empty/unsafe corridor.
        """
        left = np.asarray(left_curve, dtype=np.float32).reshape(-1)
        right = np.asarray(right_curve, dtype=np.float32).reshape(-1)
        height = int(left.size)
        width = int(image_width)
        if (
            height < 16
            or width < 16
            or right.size != height
            or not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
        ):
            return LaneEstimate(False, 0.0, reason="invalid boundary curves")

        y_values = np.linspace(
            int(height * self.bottom_ratio),
            int(height * self.top_ratio),
            self.sample_count,
        ).astype(np.int32)
        y_values = np.clip(y_values, 0, height - 1)
        lefts = left[y_values]
        rights = right[y_values]
        widths = rights - lefts
        minimum_width = max(3.0, width * self.minimum_width_ratio)
        valid_rows = (
            np.isfinite(lefts)
            & np.isfinite(rights)
            & (widths >= minimum_width)
        )
        minimum_samples = max(5, self.sample_count // 3)
        if np.count_nonzero(valid_rows) < minimum_samples:
            return LaneEstimate(
                False,
                min(0.2, np.count_nonzero(valid_rows) / self.sample_count),
                reason="too few valid boundary rows",
            )

        ys = y_values[valid_rows].astype(np.float32)
        lefts = lefts[valid_rows]
        rights = rights[valid_rows]
        widths = widths[valid_rows]
        centers = (lefts + rights) * 0.5
        near_y = int(height * 0.88)
        lookahead_y = int(height * self.lookahead_ratio)
        tight_turn_y = int(height * self.tight_turn_lookahead_ratio)
        center_curve = (left + right) * 0.5
        near_x = float(center_curve[np.clip(near_y, 0, height - 1)])
        lookahead_x = float(
            center_curve[np.clip(lookahead_y, 0, height - 1)]
        )
        tight_turn_x = float(
            center_curve[np.clip(tight_turn_y, 0, height - 1)]
        )
        lateral_error = (near_x - width * 0.5) / max(1.0, width * 0.5)
        heading_error = (lookahead_x - near_x) / max(1.0, width * 0.5)
        near_heading_error = (
            (tight_turn_x - near_x) / max(1.0, width * 0.5)
        )

        coverage = float(np.count_nonzero(valid_rows) / self.sample_count)
        width_stability = float(
            np.exp(-np.std(widths) / max(2.0, float(np.mean(widths))))
        )
        confidence = float(
            np.clip(0.75 * coverage + 0.25 * width_stability, 0.0, 1.0)
        )
        center_points = np.column_stack(
            [np.clip(centers, 0, width - 1), ys]
        ).astype(np.int32)
        left_points = np.column_stack(
            [np.clip(lefts, 0, width - 1), ys]
        ).astype(np.int32)
        right_points = np.column_stack(
            [np.clip(rights, 0, width - 1), ys]
        ).astype(np.int32)
        return LaneEstimate(
            valid=True,
            confidence=confidence,
            lateral_error=float(np.clip(lateral_error, -1.0, 1.0)),
            heading_error=float(np.clip(heading_error, -1.0, 1.0)),
            near_heading_error=float(
                np.clip(near_heading_error, -1.0, 1.0)
            ),
            lookahead_point=(
                int(np.clip(lookahead_x, 0, width - 1)),
                lookahead_y,
            ),
            centerline=center_points,
            left_boundary=left_points,
            right_boundary=right_points,
            reason="ok (tracked boundary curves)",
        )

    def _contained_bend_path(self, road, anchor_ratio, preview_point=None, reference_x=None):
        """Find a local path when a global polynomial cuts an exterior bend.

        Operates only on the current already-selected semantic corridor. It
        cannot fill holes or jump a missing road row. Erosion provides room
        for every adjacent-row transition; it is image-space clearance only.
        The path ends at the requested preview, not beyond the observed road.
        """
        h, w = road.shape
        bottom = min(h - 1, int(h * self.bottom_ratio))
        goal = int(h * self.lookahead_ratio)
        near = int(h * .88)
        tight = int(h * self.tight_turn_lookahead_ratio)
        if preview_point is not None:
            target = np.asarray(preview_point)
            if (target.shape != (2,) or not np.isfinite(target).all()
                    or np.any(target != target.astype(int))):
                return None
            target_x, goal = map(int,target)
            if not (0 <= target_x < w and int(h*.5) < goal < near):
                return None
            tight = min(near-1, max(goal+1,tight))
            # Do not turn this local road planner into an obstacle bypass:
            # a closed exclusion on the direct target ray remains a veto.
            _, bg = cv2.connectedComponents((road==0).astype(np.uint8),8)
            exterior = set(np.concatenate((bg[0],bg[-1],bg[:,0],bg[:,-1])))
            ray_y = np.arange(goal,bottom+1)
            ray_x = np.rint(np.linspace(target_x,w*.5,len(ray_y))).astype(int)
            for i in range(len(ray_y)-1):
                crossed = bg[ray_y[i]:ray_y[i+1]+1,
                             min(ray_x[i],ray_x[i+1]):max(ray_x[i],ray_x[i+1])+1]
                if any(label and label not in exterior for label in np.unique(crossed)):
                    return None
        if not goal < tight <= near <= bottom:
            return None
        step = max(1, int(w * .02))
        allowed = cv2.erode(road, np.ones((3, 2 * step + 1), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        clearance = cv2.distanceTransform(road, cv2.DIST_L2, 3)
        x = np.arange(w)
        cost = np.full(w, np.inf)
        start = allowed[bottom] & (np.abs(x - w * .5) <= max(2, w * .08))
        cost[start] = 5 * ((x[start] - w * .5) / max(1, w * .08)) ** 2
        if not np.isfinite(cost).any():
            return None
        parents = []
        offsets = np.arange(-step, step + 1)
        parent_x = x[None, :] - offsets[:, None]
        parent_valid = (parent_x >= 0) & (parent_x < w)
        parent_index = np.clip(parent_x, 0, w-1)
        turn_cost = .12 * (offsets[:, None] / step) ** 2
        for y in range(bottom - 1, goal - 1, -1):
            row_allowed = allowed[y].copy()
            wide = np.zeros(w, bool)
            for left, right in self._segments(road[y], max(3, int(w * self.minimum_width_ratio))):
                wide[left:right + 1] = True
            row_allowed &= wide
            if preview_point is not None and y == goal:
                row_allowed &= np.abs(x-target_x) <= max(2,int(w*.025))
            # Same adjacent-row choices/costs; build the indices once instead
            # of allocating one shifted width-sized array for every offset.
            candidates = cost[parent_index] + turn_cost
            candidates[~parent_valid] = np.inf
            choice = candidates.argmin(axis=0)
            cost = candidates[choice, x] + 2. / (clearance[y] + 1.)
            if reference_x is not None:
                # Keep the same geometric reference across polynomial/contained
                # path transitions. This is a soft cost only: the current road
                # mask, erosion and final transition checks remain mandatory.
                cost += .04 * ((x-reference_x[y]) / max(1,w*.05))**2
            cost[~row_allowed] = np.inf
            if not np.isfinite(cost).any():
                return None
            parents.append(x - offsets[choice])
        chosen = int(cost.argmin())
        path = [(chosen, goal)]
        for index in range(len(parents) - 1, -1, -1):
            chosen = int(parents[index][chosen])
            path.append((chosen, bottom - index))
        points = np.asarray(path[::-1], dtype=np.int32)
        # Explicitly validate every transition in the ORIGINAL current mask.
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            if not np.all(road[y1:y0 + 1, min(x0,x1):max(x0,x1) + 1]):
                return None
        by_y = {int(y): int(px) for px, y in points}
        near_x, goal_x, tight_x = by_y[near], by_y[goal], by_y[tight]
        confidence = float(min(.9, .7 + .2 * min(1., anchor_ratio / .45)))
        return LaneEstimate(
            True, confidence,
            lateral_error=float(np.clip((near_x-w*.5)/(w*.5), -1, 1)),
            heading_error=float(np.clip((goal_x-near_x)/(w*.5), -1, 1)),
            near_heading_error=float(np.clip((tight_x-near_x)/(w*.5), -1, 1)),
            lookahead_point=(goal_x, goal), centerline=points,
            reason=('current-semantic front-corner path' if preview_point is not None
                    else 'contained current-semantic bend path'),
        )

    def estimate(
        self,
        drivable_mask: np.ndarray,
        lane_mask: Optional[np.ndarray] = None,
        route_hint: str = "center",
        preserve_exclusions: bool = False,
        semantic_preview_point=None,
    ) -> LaneEstimate:
        if route_hint not in {"left", "center", "right"}:
            raise ValueError("route_hint must be left, center, or right")
        if drivable_mask is None or np.asarray(drivable_mask).ndim != 2:
            return LaneEstimate(False, 0.0, reason="invalid drivable-area mask")

        raw_mask = (np.asarray(drivable_mask) > 0).astype(np.uint8)
        h, w = raw_mask.shape
        if h < 16 or w < 16 or not np.any(raw_mask):
            return LaneEstimate(False, 0.0, reason="drivable area is empty")

        if preserve_exclusions and lane_mask is not None:
            raw_mask &= (np.asarray(lane_mask) == 0).astype(np.uint8)
        road, anchor_ratio = self._select_component(raw_mask, preserve_exclusions)
        if semantic_preview_point is not None:
            if not preserve_exclusions:
                return LaneEstimate(False,0.,reason='semantic preview requires current exclusions')
            contained = self._contained_bend_path(road,anchor_ratio,semantic_preview_point)
            return contained if contained is not None else LaneEstimate(
                False,0.,reason='current-semantic corner target has no contained path')
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
            band = (road[int(y)].copy() if preserve_exclusions else
                    np.any(road[y0:y1] > 0, axis=0).astype(np.uint8))
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
        tight_turn_y = int(h * self.tight_turn_lookahead_ratio)

        def fitted_x(y: int) -> float:
            return float(np.polyval(coefficients, (h - y) / max(1.0, float(h))))

        near_x = fitted_x(near_y)
        lookahead_x = fitted_x(lookahead_y)
        tight_turn_x = fitted_x(tight_turn_y)
        lateral_error = (near_x - w * 0.5) / max(1.0, w * 0.5)
        heading_error = (lookahead_x - near_x) / max(1.0, w * 0.5)
        near_heading_error = (
            (tight_turn_x - near_x) / max(1.0, w * 0.5)
        )

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

        if preserve_exclusions:
            # A polynomial can cut across a bend or excluded line even when
            # every sampled row midpoint was inside the selected component.
            check_y = np.arange(
                min(int(ys.min()), near_y, lookahead_y, tight_turn_y),
                max(int(ys.max()), near_y, lookahead_y, tight_turn_y) + 1,
            )
            check_x = np.polyval(coefficients, (h - check_y) / float(h))
            if (not np.isfinite(check_x).all() or np.any(check_x < 0)
                    or np.any(check_x >= w)
                    or np.any(road[check_y, np.clip(check_x, 0, w - 1).astype(int)] == 0)):
                # Internal holes on the proposed route remain an unconditional
                # veto; this fallback only corrects exterior-boundary cutting.
                if np.isfinite(check_x).all():
                    clipped = np.clip(check_x, 0, w - 1).astype(int)
                    bad = road[check_y, clipped] == 0
                    _, background = cv2.connectedComponents((road == 0).astype(np.uint8), 8)
                    exterior = set(np.concatenate((background[0], background[-1],
                                                   background[:,0], background[:,-1])))
                    if all(label in exterior for label in background[check_y[bad], clipped[bad]]):
                        reference_x = np.polyval(coefficients, (h-np.arange(h))/float(h))
                        contained = self._contained_bend_path(road, anchor_ratio,
                                                              reference_x=reference_x)
                        if contained is not None:
                            return contained
                return LaneEstimate(False, 0.0, reason="fitted centerline leaves semantic corridor")

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
            near_heading_error=float(
                np.clip(near_heading_error, -1.0, 1.0)
            ),
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
        tight_turn_factor = 0.0
        near_heading = float(estimate.near_heading_error)
        if steering * near_heading > 0.0:
            start = float(self.config.tight_turn_near_heading_start)
            full = float(self.config.tight_turn_near_heading_full)
            tight_turn_factor = float(
                np.clip((abs(near_heading) - start) / (full - start), 0.0, 1.0)
            )
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
            tight_turn_factor,
        )
