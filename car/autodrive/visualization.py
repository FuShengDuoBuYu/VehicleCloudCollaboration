"""Visualization helpers for the onboard outer-loop LCC."""

import cv2
import numpy as np

from .lane_centering import DifferentialDriveCommand, LaneEstimate


def _scale_points(points: np.ndarray, mask_shape: tuple[int, int], frame_shape) -> np.ndarray:
    if points.size == 0:
        return points
    mask_h, mask_w = mask_shape
    frame_h, frame_w = frame_shape[:2]
    scaled = points.astype(np.float32).copy()
    scaled[:, 0] *= frame_w / mask_w
    scaled[:, 1] *= frame_h / mask_h
    return np.rint(scaled).astype(np.int32)


def render_debug_frame(
    frame: np.ndarray,
    drivable_mask: np.ndarray,
    lane_mask: np.ndarray,
    estimate: LaneEstimate,
    command: DifferentialDriveCommand,
    inference_ms: float,
    latency_label: str = "boundary",
) -> np.ndarray:
    output = frame.copy()
    frame_h, frame_w = output.shape[:2]
    mask_h, mask_w = drivable_mask.shape
    drivable = cv2.resize(
        (drivable_mask > 0).astype(np.uint8),
        (frame_w, frame_h),
        interpolation=cv2.INTER_NEAREST,
    )
    lane = cv2.resize(
        (lane_mask > 0).astype(np.uint8),
        (frame_w, frame_h),
        interpolation=cv2.INTER_NEAREST,
    )

    color_layer = np.zeros_like(output)
    color_layer[drivable > 0] = (30, 150, 30)
    color_layer[lane > 0] = (20, 20, 230)
    visible = (drivable > 0) | (lane > 0)
    if np.any(visible):
        blended = (
            output[visible].astype(np.float32) * 0.55
            + color_layer[visible].astype(np.float32) * 0.45
        )
        output[visible] = np.clip(blended, 0, 255).astype(np.uint8)

    left = _scale_points(estimate.left_boundary, (mask_h, mask_w), frame.shape)
    right = _scale_points(estimate.right_boundary, (mask_h, mask_w), frame.shape)
    center = _scale_points(estimate.centerline, (mask_h, mask_w), frame.shape)
    if left.size:
        cv2.polylines(output, [left], False, (255, 160, 0), 2)
    if right.size:
        cv2.polylines(output, [right], False, (255, 160, 0), 2)
    if center.size:
        cv2.polylines(output, [center], False, (0, 255, 255), 4)

    ego = (frame_w // 2, int(frame_h * 0.92))
    cv2.circle(output, ego, 8, (255, 255, 255), -1)
    if estimate.lookahead_point is not None:
        lookahead = np.asarray([estimate.lookahead_point], dtype=np.int32)
        lookahead = _scale_points(lookahead, (mask_h, mask_w), frame.shape)[0]
        lookahead_tuple = (int(lookahead[0]), int(lookahead[1]))
        cv2.circle(output, lookahead_tuple, 10, (0, 255, 255), -1)
        cv2.arrowedLine(output, ego, lookahead_tuple, (255, 255, 255), 3)

    panel_height = 112
    panel = output[:panel_height].copy()
    output[:panel_height] = cv2.addWeighted(
        panel, 0.30, np.zeros_like(panel), 0.70, 0
    )
    left_pwm, right_pwm = command.as_pwm()
    lines = [
        f"LCC: {command.action}  steer={command.steering:+.3f}",
        (
            f"error: lateral={estimate.lateral_error:+.3f}  "
            f"heading={estimate.heading_error:+.3f}  conf={estimate.confidence:.2f}"
        ),
        (
            f"wheel proposal: L={command.left_speed:+.2f} ({left_pwm:+d})  "
            f"R={command.right_speed:+.2f} ({right_pwm:+d})  "
            f"{latency_label}={inference_ms:.0f}ms"
        ),
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (14, 30 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output
