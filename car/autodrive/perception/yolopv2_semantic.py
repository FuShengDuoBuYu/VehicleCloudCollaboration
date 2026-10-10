"""Primary YOLO road perception with an optional, track-specific paint policy."""
import cv2
import numpy as np

from .outer_loop import BoundaryTrackResult
from .yolopv2_fusion import YOLOPv2FusionDetector
from .semantic_stability import SemanticStabilizer


class YOLOPv2SemanticDetector(YOLOPv2FusionDetector):
    source_name = "yolopv2-primary"

    def __init__(self, config, outer_loop_config=None, stability_config=None,
                 track_color_config=None, **kwargs):
        if not config.required_for_motion or config.drivable_only or not config.detect_objects:
            raise ValueError("primary YOLOPv2 requires road/lane/object heads and required_for_motion=true")
        if config.backend != "torchscript" or config.adaptive_precision:
            raise ValueError("primary YOLOPv2 currently requires full TorchScript without adaptive precision")
        self.outer_selector = None
        colors=track_color_config or {}
        if (set(colors)-{'enabled','surface_assist'} or type(colors.get('enabled',False)) is not bool
                or type(colors.get('surface_assist',False)) is not bool
                or colors.get('surface_assist',False) and not colors.get('enabled',False)):
            raise ValueError('track_colors requires boolean enabled/surface_assist flags')
        self.track_colors=colors.get('enabled',False)
        self.surface_assist=colors.get('surface_assist',False)
        self.stabilizer = SemanticStabilizer(**(stability_config or {'enabled': False}))
        if outer_loop_config and outer_loop_config.get('enabled', False):
            from .semantic_outer_route import SemanticOuterRoute
            self.outer_selector = SemanticOuterRoute(**outer_loop_config)
        super().__init__(config, **kwargs)

    def semantic_corridor(self, drivable, lane):
        """Exclude model lane pixels; fail closed on missing or old evidence.

        Confidence here measures geometric support, not model calibration.
        The near-centre anchor and all-image rejection are conservative initial
        deployment checks; they do not establish obstacle clearance in metres.
        """
        road = np.asarray(drivable)
        lines = np.asarray(lane)
        if (road.ndim != 2 or road.shape != lines.shape or min(road.shape) < 16
                or not np.all(np.isfinite(road)) or not np.all(np.isfinite(lines))):
            raise ValueError("semantic masks must be finite, same-shaped 2D arrays")
        corridor = ((road > 0) & (lines == 0)).astype(np.uint8)
        result = self._consumer_result
        with self._lock:
            error = self._error
        age = None if result is None else max(0., self._clock() - result.captured_at)
        # Plausibility and physical near-camera support must be checked in
        # camera coordinates: a warp's zero padding can hide an all-road error.
        raw_road = road if result is None else result.drivable_mask
        raw_lane = lines if result is None else result.lane_mask
        exclusion=None;paint=None;color_error=False;yellow_hazard=False;ego_yellow_ratio=0.
        surface=dict(observed=False,mask=np.zeros_like(road,dtype=np.uint8),front_ratio=None)
        supported_road=raw_road
        if self.track_colors:
            from .track_colors import classify_track_paint, observe_track_surface, PARAMETERS
            try:
                if (result is None or not np.array_equal(road,raw_road)
                        or not np.array_equal(lines,raw_lane)):
                    raise ValueError('track colors require the consumed camera-coordinate masks')
                if self.surface_assist:
                    surface=observe_track_surface(raw_road,result.source_frame)
                    supported_road=((raw_road>0)|(surface['mask']>0)).astype(np.uint8)
                paint=classify_track_paint(supported_road,raw_lane,result.source_frame)
                exclusion=paint.exclusion_mask
                if self.surface_assist:
                    # This course defines only yellow paint as forbidden;
                    # lane-head detections of observed floor texture are not
                    # physical barriers. Detected objects still veto below.
                    # Current model road plus current gray/white floor also
                    # distinguishes traversable texture from yellow paint.
                    # This never adds missing model road without a boundary.
                    floor=(surface['mask']>0)|(surface['model_floor_mask']>0)
                    exclusion=((exclusion>0)&~floor|
                               (paint.yellow_mask>0)).astype(np.uint8)
                corridor=((supported_road>0)&(exclusion==0)).astype(np.uint8)
                ego_yellow_ratio=paint.ego_yellow_ratio
                yellow_hazard=(paint.rotation_yellow_ratio if self.surface_assist else ego_yellow_ratio)>PARAMETERS['maximum_ego_yellow_ratio']
            except (ValueError,cv2.error):
                color_error=True
        ratio = float(np.mean(raw_road > 0))
        rh, rw = raw_road.shape
        raw_corridor = (supported_road > 0) & ((raw_lane if exclusion is None else exclusion) == 0)
        raw_anchor = float(np.mean(raw_corridor[int(.80*rh):int(.96*rh), int(.42*rw):int(.58*rw)]))
        h, w = corridor.shape
        anchor = float(np.mean(corridor[int(.80*h):int(.96*h), int(.42*w):int(.58*w)]))
        from .track_colors import rotation_origin_region
        rotation_anchor=float(np.mean(rotation_origin_region(raw_corridor)))
        reason = "fresh YOLOPv2 road/lane corridor"
        valid = True
        preview_point = None
        # Image-space veto only; this is not a calibrated stopping distance.
        near_obstacle = False
        if result is not None:
            for detection in result.detections:
                x1, y1, x2, y2 = detection['box']
                if (detection['confidence'] >= .3 and y2 > .65
                        and x1 < .60 and x2 > .40):
                    near_obstacle = True
                    break
        _, _, components, _ = cv2.connectedComponentsWithStats(raw_corridor.astype(np.uint8), 8)
        pivot_support = (float(np.max(components[1:,cv2.CC_STAT_AREA])) / raw_corridor.size
                         if len(components) > 1 else 0.)
        # Rotation can retain IMU angle while the near-centre anchor leaves the
        # view. A failed mask containing only tiny fragments cannot authorize
        # it. This initial plausibility bound is not a physical clearance test.
        hard_safe = (not self._closed and result is not None and not error
                     and age <= self.config.max_result_age_seconds
                     and (ratio>0 or surface['observed']) and ratio<=.95 and pivot_support >= .05 and not near_obstacle
                     and not color_error and not yellow_hazard)
        if self._closed or result is None:
            valid, reason = False, "YOLOPv2 unavailable"
        elif error:
            valid, reason = False, "YOLOPv2 inference error"
        elif age > self.config.max_result_age_seconds:
            valid, reason = False, "YOLOPv2 stale result"
        elif color_error:
            valid, reason = False, 'track colors missing matched source frame'
        elif yellow_hazard:
            valid, reason = False, 'yellow boundary entered the vehicle safety zone'
        elif (ratio == 0 and not surface['observed']) or ratio > .95:
            valid, reason = False, "YOLOPv2 empty or implausible all-image road mask"
        elif (raw_anchor < .60 or anchor < .60) and not (surface['observed'] and rotation_anchor>=.90):
            valid, reason = False, "YOLOPv2 missing near-centre road support"
        elif near_obstacle:
            valid, reason = False, "YOLOPv2 near obstacle overlaps forward corridor"
        # Select/validate the outer branch on RAW current evidence: smoothing
        # must not manufacture a visible edge when road reaches image x=0.
        selector_used = valid and self.outer_selector is not None
        if selector_used:
            corridor, valid, reason = self.outer_selector.select(corridor,surface_observed=surface['observed'])
            preview_point = self.outer_selector.preview_point if valid else None
        # Smooth only the already selected route; no historical re-routing.
        if valid:
            corridor = self.stabilizer.update(corridor, result.sequence, result.captured_at)
        else:
            self.stabilizer.reset()
        if not valid:
            corridor[:] = 0
            self.stabilizer.reset()
        confidence = min(1., max(anchor,rotation_anchor if surface['observed'] else 0.) / .75) if valid else 0.
        state = {
            "active": valid, "motion_allowed": valid,
            "source": ('yolopv2+track-surface' if surface['observed'] else 'yolopv2') if valid else reason,
            "result_age_seconds": age, "overlap_ratio": None, "drivable_ratio": ratio,
            "inference_seconds": None if result is None else result.inference_seconds,
            "precision": None if result is None else result.precision,
            "requested_precision": self._requested_precision, "confidence_scale": confidence,
            "stability": dict(self.stabilizer.state),
            'track_colors':dict(enabled=self.track_colors,
                surface_assist=self.surface_assist,surface_observed=surface['observed'],
                surface_pixels=int(surface['mask'].sum()),
                model_floor_pixels=int(surface.get('model_floor_mask',np.zeros_like(road)).sum()),
                allowed_white_pixels=0 if paint is None else paint.allowed_white_pixels,
                yellow_pixels=0 if paint is None else paint.yellow_pixels,
                ego_yellow_ratio=ego_yellow_ratio,
                parameters=PARAMETERS if self.track_colors else None),
        }
        self._last_fusion = state
        rotation_mask=raw_corridor.astype(np.uint8) if surface['observed'] and hard_safe else None
        from .visual_clearance import assess_rotation_clearance
        rotation=assess_rotation_clearance(rotation_mask,None if paint is None else paint.yellow_mask)
        return BoundaryTrackResult(
            valid, confidence, corridor, "yolopv2", reason=reason,
            semantic_exclusion_mask=exclusion,
            semantic_track_colors=self.track_colors,
            semantic_allowed_white_pixels=0 if paint is None else paint.allowed_white_pixels,
            semantic_yellow_pixels=0 if paint is None else paint.yellow_pixels,
            semantic_yellow_mask=None if paint is None else paint.yellow_mask,
            semantic_track_surface_pixels=int(surface['mask'].sum()),
            semantic_turn_entry_ready=(self.outer_selector.turn_entry_ready if selector_used else False),
            semantic_exit_heading_error=(self.outer_selector.exit_heading_error if selector_used else None),
            semantic_rotation_mask=rotation_mask,
            semantic_rotation_heading_error=rotation['heading_error'],
            semantic_forward_road_valid=bool(raw_anchor>=.60 and anchor>=.60),
            semantic_rotation_yellow_ratio=None if paint is None else paint.rotation_yellow_ratio,
            yellow_hazard=yellow_hazard,ego_yellow_ratio=ego_yellow_ratio,
            semantic_preview_point=preview_point if valid else None,
            semantic_front_boundary_ratio=(self.outer_selector.front_boundary_ratio
                if valid and self.outer_selector is not None else None),
            semantic_right_exit_observed=(self.outer_selector.right_exit_observed
                if valid and self.outer_selector is not None else False),
            semantic_hard_safe=bool(hard_safe),
            semantic_front_observed=(self.outer_selector.front_observed if selector_used else False),
            semantic_observed_front_ratio=(self.outer_selector.observed_front_ratio if selector_used else None),
            semantic_front_spread_ratio=(self.outer_selector.front_spread_ratio if selector_used else None),
            semantic_right_exit_support_ratio=(self.outer_selector.right_exit_support_ratio if selector_used else None),
            semantic_corner_reject_reason=(self.outer_selector.corner_reject_reason if selector_used else ''),
            semantic_pivot_road_support_ratio=pivot_support,
            semantic_sequence=None if result is None else result.sequence,
            semantic_captured_at=None if result is None else result.captured_at,
            semantic_fusion_source=state["source"], semantic_result_age_seconds=age,
            semantic_drivable_ratio=ratio,
            semantic_inference_seconds=state["inference_seconds"],
            semantic_precision=state["precision"], semantic_requested_precision=self._requested_precision)
