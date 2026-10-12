"""Current-frame perception and local proposal computation."""
from __future__ import annotations

from autodrive.control.lane_centering import DifferentialDriveCommand
from autodrive.control.lane_centering import LaneEstimate
from autodrive.perception.outer_loop import BoundaryTrackResult
from autodrive.perception.visualization import render_debug_frame
from dataclasses import replace
import numpy as np
import time


class SurfaceOnlyDetector:
    """Provide mask geometry without running an unused semantic model.

    The classical outer-loop controller derives its corridor directly from
    every fresh camera frame. It only needs empty, correctly sized semantic
    masks so the common visualization and perspective code can stay shared.
    """

    source_name = "surface-only"

    def __init__(self, width=320, height=180):
        self.width = int(width)
        self.height = int(height)
        if self.width < 16 or self.height < 16:
            raise ValueError("surface mask dimensions must be at least 16 pixels")

    def predict_masks(self, _frame):
        shape = (self.height, self.width)
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=np.uint8)


def fuse_tracking_confidence(
    centerline_confidence: float,
    boundary_confidence: float,
) -> float:
    """Combine two confidence stages without double-penalizing one corridor.

    The centerline estimate is computed from the boundary tracker's corridor,
    so multiplying both values treats the same weak observation as two
    independent failures.  The weaker value is still a conservative joint
    confidence: either stage can stop the car, while a valid single-boundary
    corner does not lose confidence a second time.
    """
    return float(
        np.clip(
            min(float(centerline_confidence), float(boundary_confidence)),
            0.0,
            1.0,
        )
    )


def analyze(
    detector,
    estimator,
    controller,
    mapper,
    frame,
    route_hint,
    dt,
    maximum_inference_time,
    boundary_tracker=None,
    captured_at=None,
    deferred_render=None,
    renderer=render_debug_frame,
):
    started = time.monotonic()
    if hasattr(detector, "semantic_corridor"):
        drivable_mask, lane_mask = detector.predict_masks(frame, captured_at=captured_at)
    else:
        drivable_mask, lane_mask = detector.predict_masks(frame)
    inference_time = time.monotonic() - started
    control_drivable = (
        mapper.warp_mask(drivable_mask) if mapper is not None else drivable_mask
    )
    control_lane = mapper.warp_mask(lane_mask) if mapper is not None else lane_mask
    boundary_started = time.monotonic()
    boundary_result = None
    yellow_hazard = False
    ego_yellow_ratio = 0.0
    display_drivable = drivable_mask
    display_lane = lane_mask
    semantic_fusion = None
    if hasattr(detector, "semantic_corridor"):
        boundary_result = detector.semantic_corridor(control_drivable, control_lane)
        if getattr(boundary_result,'semantic_exclusion_mask',None) is not None:
            control_lane = boundary_result.semantic_exclusion_mask
            display_lane = control_lane
        yellow_hazard = boundary_result.yellow_hazard
        ego_yellow_ratio = boundary_result.ego_yellow_ratio
        control_drivable = boundary_result.corridor_mask
        display_drivable = (mapper.camera_mask(control_drivable)
                            if mapper is not None else control_drivable)
    elif boundary_tracker is not None:
        yellow_mask = boundary_tracker.yellow_mask(frame, lane_mask.shape)
        # The calibration trapezoid can end where the lane boundaries leave
        # the image, well ahead of the physical chassis. The actual vehicle
        # footprint therefore stays in the raw image's narrow bottom-centre
        # zone; only boundary fitting uses the bird's-eye yellow mask.
        yellow_hazard, ego_yellow_ratio = boundary_tracker.yellow_under_ego(
            yellow_mask
        )
        road_surface_mask = boundary_tracker.road_surface_mask(
            frame, lane_mask.shape
        )
        control_yellow = (
            mapper.warp_mask(yellow_mask) if mapper is not None else yellow_mask
        )
        control_surface = (
            mapper.warp_mask(road_surface_mask)
            if mapper is not None
            else road_surface_mask
        )
        control_surface &= (control_yellow == 0).astype(np.uint8)
        if boundary_tracker.config.navigation_mode == "surface":
            boundary_result = BoundaryTrackResult(
                bool(np.any(control_surface)),
                1.0 if np.any(control_surface) else 0.0,
                control_surface,
                "surface",
                reason="non-green outer-route surface",
            )
        else:
            boundary_result = boundary_tracker.update(
                control_surface,
                control_lane,
                control_yellow,
            )
        # Publish the current raw-image safety evidence before adaptive
        # precision selection.  In particular, a yellow-under-ego event must
        # reject an in-flight INT8 result in this same control frame.
        boundary_result.ego_yellow_ratio = ego_yellow_ratio
        boundary_result.yellow_hazard = yellow_hazard
        if hasattr(detector, "fuse_corridor"):
            if hasattr(detector, "observe_boundary"):
                detector.observe_boundary(boundary_result)
            fused_corridor, semantic_fusion = detector.fuse_corridor(
                boundary_result.corridor_mask,
                control_drivable,
            )
            boundary_result.corridor_mask = fused_corridor
            boundary_result.semantic_fusion_source = semantic_fusion["source"]
            boundary_result.semantic_result_age_seconds = semantic_fusion[
                "result_age_seconds"
            ]
            boundary_result.semantic_overlap_ratio = semantic_fusion[
                "overlap_ratio"
            ]
            boundary_result.semantic_drivable_ratio = semantic_fusion[
                "drivable_ratio"
            ]
            boundary_result.semantic_inference_seconds = semantic_fusion[
                "inference_seconds"
            ]
            boundary_result.semantic_precision = semantic_fusion["precision"]
            boundary_result.semantic_requested_precision = semantic_fusion[
                "requested_precision"
            ]
            boundary_result.confidence *= semantic_fusion["confidence_scale"]
            if not semantic_fusion["motion_allowed"]:
                boundary_result.valid = False
                boundary_result.confidence = 0.0
                boundary_result.reason = (
                    "YOLOPv2 fusion required for motion: "
                    f"{semantic_fusion['source']}"
                )
        control_drivable = boundary_result.corridor_mask
        display_drivable = (
            mapper.camera_mask(control_drivable)
            if mapper is not None
            else control_drivable
        )
    boundary_time = time.monotonic() - boundary_started
    if yellow_hazard:
        estimate = LaneEstimate(
            False,
            0.0,
            reason="yellow boundary entered the vehicle safety zone",
        )
    elif boundary_result is not None and not boundary_result.valid:
        estimate = LaneEstimate(
            False,
            boundary_result.confidence,
            reason=boundary_result.reason,
        )
    else:
        fresh_boundary_sources = {"both", "outer+width", "inner+width"}
        if hasattr(detector, "semantic_corridor"):
            estimate = estimator.estimate(
                control_drivable, control_lane, route_hint, preserve_exclusions=True,
                semantic_preview_point=boundary_result.semantic_preview_point,
            )
        elif (
            boundary_result is not None
            and boundary_result.source in fresh_boundary_sources
            and boundary_result.left_curve.size
            and boundary_result.right_curve.size
        ):
            estimate = estimator.estimate_from_boundaries(
                boundary_result.left_curve,
                boundary_result.right_curve,
                boundary_result.corridor_mask.shape[1],
            )
            # Retain the surface-mask estimator as a defensive fallback for a
            # malformed fitted curve. Fresh valid curves are the primary LCC
            # geometry; history/dropout sources still use the corridor path so
            # the bounded corner-continuation safety gate remains in control.
            if not estimate.valid:
                estimate = estimator.estimate(
                    control_drivable, control_lane, route_hint
                )
        else:
            estimate = estimator.estimate(
                control_drivable, control_lane, route_hint
            )
        if boundary_result is not None:
            estimate.confidence = fuse_tracking_confidence(
                estimate.confidence,
                boundary_result.confidence,
            )
    estimate.confidence = float(np.clip(estimate.confidence, 0.0, 1.0))
    control_estimate = estimate
    if getattr(getattr(detector, 'outer_selector', None), 'visual_feedback', False):
        control_estimate = replace(estimate, heading_error=estimate.near_heading_error)
    command = controller.update(control_estimate, dt)
    if inference_time + boundary_time > maximum_inference_time:
        command = DifferentialDriveCommand(
            "stop",
            0.0,
            0.0,
            0.0,
            estimate.confidence,
            "inference exceeded safety limit",
        )
    display_estimate = (
        mapper.camera_estimate(estimate, drivable_mask.shape)
        if mapper is not None
        else estimate
    )
    display_frame = frame
    if hasattr(detector, 'semantic_corridor') and hasattr(detector, 'get_consumed_frame'):
        consumed_frame = detector.get_consumed_frame()
        if consumed_frame is not None:
            display_frame = consumed_frame
    def render():
        return renderer(
            display_frame,
            display_drivable,
            display_lane,
            display_estimate,
            command,
            ((boundary_result.semantic_inference_seconds or 0.0) * 1000
             if hasattr(detector, "semantic_corridor") else inference_time * 1000),
            control_label=("YOLOPv2 -> control" if hasattr(detector, "semantic_corridor") else "LCC"),
            latency_label=(
                "YOLOPv2" if hasattr(detector, "semantic_corridor") else "boundary"
                if boundary_result is None
                else f"boundary/{boundary_result.source}"
            ),
            boundary_source=(
                "missing" if boundary_result is None else boundary_result.source
            ),
            semantic_mask=(
                drivable_mask if hasattr(detector, "fuse_corridor") else None
            ),
            semantic_label=(
                ("YOLOPv2 matched frame; raw road blue; filtered corridor green; "
                 + ("white allowed; yellow/unknown excluded red; "
                    if getattr(boundary_result,'semantic_track_colors',False) else "")
                 +
                 f"age={float(boundary_result.semantic_result_age_seconds or 0):.2f}s")
                if hasattr(detector, "semantic_corridor") else ""
                if semantic_fusion is None
                else (
                    "blue outline=YOLOPv2 green=fused LCC  "
                    f"{semantic_fusion['source']} "
                    f"{semantic_fusion['precision'] or 'none'}->"
                    f"{semantic_fusion['requested_precision']} "
                    f"overlap={float(semantic_fusion['overlap_ratio'] or 0.0):.2f} "
                    f"age={float(semantic_fusion['result_age_seconds'] or 0.0):.2f}s"
                )
            ),
        )
    if deferred_render is None:
        annotated = render()
    else:
        deferred_render(render)
        annotated = None
    return (
        estimate,
        command,
        annotated,
        inference_time,
        boundary_time,
        boundary_result,
    )

