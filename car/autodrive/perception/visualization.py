"""Visualization helpers for the onboard outer-loop LCC."""

import cv2
import numpy as np

from ..control.lane_centering import DifferentialDriveCommand, LaneEstimate


def _scale_points(points: np.ndarray, mask_shape: tuple[int, int], frame_shape) -> np.ndarray:
    if points.size == 0:
        return points
    mask_h, mask_w = mask_shape
    frame_h, frame_w = frame_shape[:2]
    scaled = points.astype(np.float32).copy()
    scaled[:, 0] *= frame_w / mask_w
    scaled[:, 1] *= frame_h / mask_h
    return np.rint(scaled).astype(np.int32)


def _draw_boundary(
    image: np.ndarray,
    points: np.ndarray,
    inferred: bool,
) -> None:
    """Draw measured edges solid cyan and inferred edges dashed magenta."""
    if points.size == 0:
        return
    if not inferred:
        cv2.polylines(image, [points], False, (255, 160, 0), 2)
        return
    for index in range(0, len(points) - 1, 2):
        cv2.line(
            image,
            tuple(map(int, points[index])),
            tuple(map(int, points[index + 1])),
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )


def render_debug_frame(
    frame: np.ndarray,
    drivable_mask: np.ndarray,
    lane_mask: np.ndarray,
    estimate: LaneEstimate,
    command: DifferentialDriveCommand,
    inference_ms: float,
    latency_label: str = "boundary",
    boundary_source: str = "both",
    semantic_mask: np.ndarray = None,
    semantic_label: str = "",
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
    semantic = None
    if semantic_mask is not None:
        semantic = cv2.resize(
            (np.asarray(semantic_mask) > 0).astype(np.uint8),
            (frame_w, frame_h),
            interpolation=cv2.INTER_NEAREST,
        )

    color_layer = np.zeros_like(output)
    color_layer[drivable > 0] = (30, 150, 30)
    color_layer[lane > 0] = (20, 20, 230)
    visible = (drivable > 0) | (lane > 0)
    if np.any(visible):
        # OpenCV performs the full-frame blend in optimized native code. This
        # is materially faster than float-converting a broad YOLO mask through
        # NumPy fancy indexing on the Raspberry Pi.
        blended = cv2.addWeighted(output, 0.55, color_layer, 0.45, 0)
        output[visible] = blended[visible]
    if semantic is not None:
        # Drawing only the proposal outline keeps the broad semantic mask
        # interpretable without alpha-blending most of every camera frame.
        contours, _ = cv2.findContours(
            semantic,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, (255, 100, 0), 2, cv2.LINE_AA)

    left = _scale_points(estimate.left_boundary, (mask_h, mask_w), frame.shape)
    right = _scale_points(estimate.right_boundary, (mask_h, mask_w), frame.shape)
    center = _scale_points(estimate.centerline, (mask_h, mask_w), frame.shape)
    historical = boundary_source in {"history", "visible-history"}
    _draw_boundary(
        output,
        left,
        inferred=historical or boundary_source == "inner+width",
    )
    _draw_boundary(
        output,
        right,
        inferred=historical or boundary_source == "outer+width",
    )
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
            f"heading={estimate.heading_error:+.3f}  "
            f"near={estimate.near_heading_error:+.3f}  conf={estimate.confidence:.2f}"
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
    if semantic is not None:
        cv2.putText(
            output,
            semantic_label or "blue outline=YOLOPv2  green=fused LCC",
            (8, frame_h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output
