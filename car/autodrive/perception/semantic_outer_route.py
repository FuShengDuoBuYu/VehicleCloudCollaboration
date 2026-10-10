"""Select the outer-left route using only current YOLO road/lane pixels.

The row-width prior comes from this fixed camera's recorded straight road.
It constrains an opening into the inner course; it never creates road pixels
or supplies a missing current outer edge. This is image-space route selection,
not a metric vehicle-footprint or obstacle-clearance guarantee.
"""
import cv2
import numpy as np


def route_options(config):
    """Same derived route geometry/policy in live and recorded replay."""
    options=config.get('perception',{}).get('semantic_outer_loop')
    if options and options.get('visual_feedback'):
        options=dict(options,path_start_ratio=config.get('centerline',{}).get('bottom_ratio',.94))
        corner=config.get('stationary_corner',{})
        if corner.get('visual_corner_guard'):
            options.update(guarded_corner_approach=True,corner_approach_ratio=corner.get('front_near_ratio',.74))
            if config.get('perception',{}).get('track_colors',{}).get('surface_assist') is True:
                # The observed paint frontier replaces the model dropout edge.
                # Enter a turn when it reaches the current path lookahead.
                options['corner_approach_ratio']=config.get('centerline',{}).get('lookahead_ratio',.62)
    return options


class SemanticOuterRoute:
    def __init__(self, width_profile, preview_ratio=.62, enabled=True, stationary_corners=False,
                 visual_feedback=False, path_start_ratio=.94,guarded_corner_approach=False,
                 corner_approach_ratio=.74):
        values = np.asarray(width_profile, dtype=float)
        if (type(enabled) is not bool or not enabled or values.ndim != 2
                or values.shape[1] != 2 or len(values) < 2
                or not np.isfinite(values).all()
                or np.any(np.diff(values[:, 0]) <= 0)
                or np.any(values[:, 0] < 0) or np.any(values[:, 0] > 1)
                or np.any(values[:, 1] <= 0) or np.any(values[:, 1] > 1)):
            raise ValueError('outer route requires a finite ordered row-width profile')
        if not .5 <= preview_ratio <= .8:
            raise ValueError('outer preview must be within [.5, .8]')
        if type(stationary_corners) is not bool:
            raise ValueError('stationary_corners must be boolean')
        self.width_profile = values
        self.preview_ratio = float(preview_ratio)
        self.preview_point = None
        self.stationary_corners = stationary_corners
        if type(visual_feedback) is not bool or visual_feedback and not stationary_corners:
            raise ValueError('visual feedback requires stationary_corners=true')
        self.visual_feedback = visual_feedback
        if (type(path_start_ratio) not in (int,float) or not np.isfinite(path_start_ratio)
                or not .88 < path_start_ratio < 1.):
            raise ValueError('path start must be below the near-heading row and inside the image')
        self.path_start_ratio = float(path_start_ratio)
        if (type(guarded_corner_approach) is not bool
                or guarded_corner_approach and not visual_feedback):
            raise ValueError('guarded approach requires visual feedback')
        if (type(corner_approach_ratio) not in (int,float) or not np.isfinite(corner_approach_ratio)
                or not .5 < corner_approach_ratio <= .85):
            raise ValueError('corner approach ratio must be in (.5,.85]')
        self.guarded_corner_approach=guarded_corner_approach
        self.corner_approach_ratio=corner_approach_ratio
        self._reset_observation()

    def _reset_observation(self):
        self.preview_point = None
        self.front_boundary_ratio = None
        self.right_exit_observed = False
        self.front_observed = False
        self.observed_front_ratio = None
        self.front_spread_ratio = None
        self.right_exit_support_ratio = None
        self.corner_reject_reason = 'front-not-observed'
        self.turn_entry_ready = False
        self.exit_heading_error = None

    def _front_corner_preview(self, road):
        """Observe the front and independently confirm a right exit.

        Used only for the requested clockwise outer route. All evidence is
        from the current ego-connected model component; all-road masks and
        Stationary mode can approach on current centre road while an exit is
        uncertain. Only a confirmed exit supplies permission for rotation.
        """
        h, w = road.shape
        columns = np.arange(int(w*.30), int(w*.85))
        present = np.any(road[:, columns], axis=0)
        front = np.argmax(road[:, columns], axis=0)
        usable = present & (front > h*.50) & (front < h*(.85 if self.stationary_corners else .74))
        if np.mean(usable) < .9:
            return None
        heights = front[usable]
        self.front_observed = True
        self.observed_front_ratio = float(np.median(heights)/h)
        self.front_spread_ratio = float((np.percentile(heights,90)-np.percentile(heights,10))/h)
        if np.percentile(heights,90)-np.percentile(heights,10) > h*.12:
            self.corner_reject_reason = 'front-spread-too-large'
            return None
        if self.stationary_corners:
            # The current transverse edge remains observed when the exit head
            # drops pixels. Exit confirmation is a separate condition for pivot.
            self.front_boundary_ratio = self.observed_front_ratio
        # A transverse boundary must not bypass the outer-left branch rule.
        # Reject a split anywhere along the approach, including thin separators
        # attached to the exterior. Tiny enclosed exclusions remain excluded in
        # the returned road; this support is only for recognizing row extents.
        row_support = self._row_extent_support(road)
        split_rows = 0
        for y in ([] if self.visual_feedback else range(int(h*.50), int(h*.94)+1)):
            xs = np.flatnonzero(row_support[y])
            runs = np.split(xs, np.flatnonzero(np.diff(xs)>1)+1)
            if sum(r.size >= max(3,int(w*.04)) for r in runs) > 1:
                split_rows += 1
                if split_rows >= max(2,int(h*.025)):
                    self.corner_reject_reason = 'approach-has-split-branches'
                    return None
            else:
                split_rows = 0
        approach_point = None
        if self.stationary_corners:
            # Approach on observed road toward the transverse edge. Rotation
            # is commanded separately after a stop and fresh IMU confirmation.
            forward_x = int(w*.5)
            support = np.flatnonzero(road[:,forward_x])
            forward_y = max(int(h*.74),int(support[0]+h*.06)) if len(support) else h
            if forward_y >= int(h*.88) or not road[forward_y,forward_x]:
                self.corner_reject_reason = 'forward-approach-target-missing'
                return None
            approach_point = (forward_x,forward_y)
        # This column probes exit support; visual mode chooses its actual
        # steering target from the current observed road pixels below.
        target_x = int(w*.82)
        support = np.flatnonzero(road[:, target_x])
        if not support.size:
            self.corner_reject_reason = 'right-exit-column-missing'
            return approach_point
        target_y = max(int(h*.64), int(support[0] + h*.10))
        if target_y >= int(h*(.94 if self.stationary_corners else .82)) or not road[target_y,target_x]:
            self.corner_reject_reason = 'right-exit-target-missing'
            return approach_point
        # During a visual pivot the inner curved marking enters the lower
        # right image. Its remote branch must not veto a clear upper exit.
        # Confirm current exit support beside the target; the exact ego-to-
        # target ray below independently prevents crossing that marking.
        exit_bottom = (target_y + max(2, int(h*.025)) if self.visual_feedback
                       else int(h*(.96 if self.stationary_corners else .90)))
        right_exit = road[target_y:exit_bottom, int(w*.90):]
        self.right_exit_support_ratio = float(np.mean(right_exit)) if right_exit.size else 0.
        if not right_exit.size or np.mean(right_exit) < .85:
            self.corner_reject_reason = 'right-exit-support-insufficient'
            return approach_point
        if self.stationary_corners:
            self.right_exit_observed = True
            self.corner_reject_reason = ''
            if self.visual_feedback:
                xs=np.flatnonzero(road[target_y,int(w*.60):])+int(w*.60)
                target_x=int(xs[len(xs)//2])
                self.exit_heading_error=float((target_x-w*.5)/(w*.5))
                self.turn_entry_ready=bool(self.guarded_corner_approach
                    and self.observed_front_ratio>=self.corner_approach_ratio)
                if self.guarded_corner_approach and self.observed_front_ratio < self.corner_approach_ratio:
                    return approach_point
                # The ego branch is already selected row by row. A turn may
                # require a curved path around the inner boundary: the path
                # estimator validates each transition in this original mask.
                return target_x,target_y
            return approach_point
        self.corner_reject_reason = ''
        return target_x, target_y

    @staticmethod
    def _row_extent_support(road):
        """Do not confuse tiny enclosed exclusions with a branch boundary.

        This temporary mask is ONLY for choosing the row interval. The returned
        corridor still uses original road pixels, including every excluded
        pixel. Large islands, long lane separators and exterior-connected gaps
        remain boundaries. Bounds scale with mask resolution (at 320x240:
        at most 65 pixels, 24 wide, 6 high).
        """
        h,w = road.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats((road==0).astype(np.uint8),8)
        exterior = set(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
        eligible = ((stats[:,cv2.CC_STAT_AREA] <= max(1,int(h*w*.00085)))
                    & (stats[:,cv2.CC_STAT_WIDTH] <= max(1,int(w*.075)))
                    & (stats[:,cv2.CC_STAT_HEIGHT] <= max(1,int(h*.025))))
        eligible[0] = False
        eligible[list(exterior)] = False
        # One label lookup, not an entire image scan for each tiny fragment.
        return road | eligible[labels].astype(np.uint8)

    def select(self, corridor, *, surface_observed=False):
        self._reset_observation()
        road = (np.asarray(corridor) > 0).astype(np.uint8)
        if road.ndim != 2 or min(road.shape) < 16:
            raise ValueError('outer corridor must be a two-dimensional mask')
        h, w = road.shape
        # Ignore detached objects/islands; keep the model component underneath
        # the vehicle centre. No morphology may bridge a model-excluded line.
        count, labels, _, _ = cv2.connectedComponentsWithStats(road, 8)
        anchor = labels[int(.82*h):int(.96*h), int(.42*w):int(.58*w)]
        candidates = [(int(np.sum(anchor == label)), label) for label in range(1, count)]
        if not candidates or max(candidates)[0] == 0:
            return np.zeros_like(road), False, 'YOLO outer route has no ego-connected road'
        road = (labels == max(candidates)[1]).astype(np.uint8)
        # Visual mode must choose the reachable ego branch BEFORE looking for
        # its exit. A straight ray to a far target is not a curved-road path.
        self.preview_point = None if self.visual_feedback else self._front_corner_preview(road)
        if self.preview_point is not None:
            reason = ('current YOLO front boundary; right exit unconfirmed'
                      if self.stationary_corners and not self.right_exit_observed
                      else 'current YOLO front outer boundary and right exit')
            return road, True, reason
        row_support = self._row_extent_support(road)
        output = np.zeros_like(road)
        origin = min(h-1,int(h*self.path_start_ratio)) if self.visual_feedback else h-1
        # Choose the ego branch at the same row used by the path planner.
        # Bottom-border fragments are behind that origin and must not choose
        # the route ahead. Keep their original pixels for adjacent-row checks;
        # no exclusion or missing road is filled.
        if self.visual_feedback:
            output[origin+1:] = road[origin+1:]
        visible = []
        preview_y = self.preview_ratio * h
        previous_run = None
        branch_x = w*.5
        for y in range(origin, max(0, int(h*.45)) - 1, -1):
            xs = np.flatnonzero(row_support[y])
            runs = [r for r in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
                    if r.size >= max(3, int(.04*w))]
            if not runs:
                if abs(y-preview_y) <= h*.04:
                    visible.append(False)
                continue
            # Legacy arc mode follows the outer-left branch. Stationary mode
            # approaches on the current ego path until the independently
            # observed front boundary supplies the corner transition.
            run = runs[0]
            if self.stationary_corners:
                # A globally connected roadside sliver may reconnect only far
                # ahead. Keep the branch reachable from the current ego row,
                # rather than switching to that sliver merely because it is left.
                reachable = [r for r in runs if previous_run is None or
                             (r[-1] >= previous_run[0] and r[0] <= previous_run[1])]
                if not reachable:
                    if abs(y-preview_y) <= h*.04:
                        visible.append(False)
                    continue
                run = min(reachable, key=lambda r: (
                    max(float(r[0])-branch_x, branch_x-float(r[-1]), 0.),
                    abs((float(r[0])+float(r[-1]))*.5-branch_x), int(r[0])))
            # At the nearest rows the anchor is the vehicle, not a new branch
            # choice. Bottom-border segmentation fragments must not discard
            # currently observed road directly under the image centre. The
            # preview still uses the outer-left branch, and no gap is filled.
            if not self.stationary_corners and y >= int(h*.88):
                run = next((r for r in runs if r[0] <= w*.5 <= r[-1]), run)
            left, right = int(run[0]), int(run[-1])
            outer_visible = left > 0
            if abs(y-preview_y) <= h*.04:
                visible.append(outer_visible or self.visual_feedback and right < w-1)
            if outer_visible and not self.visual_feedback:
                width = np.interp(y/max(1, h-1), self.width_profile[:, 0],
                                  self.width_profile[:, 1]) * w
                right = min(right, left + max(3, int(round(width))) - 1)
            if self.stationary_corners:
                previous_run = (left, right)
                branch_x = (left+right)*.5
            # At the bottom the physical edges normally lie outside the image.
            # Preserve observed support there without inventing unseen geometry.
            output[y, left:right+1] = road[y, left:right+1]
        if self.visual_feedback:
            self.preview_point = self._front_corner_preview(output)
            if self.preview_point is not None:
                return output, True, ('current YOLO front outer boundary and right exit'
                    if self.right_exit_observed else 'current YOLO front boundary; right exit unconfirmed')
        observed_front = self.visual_feedback and self.front_observed
        if not observed_front and not surface_observed and (not visible or np.mean(visible) < .6):
            return np.zeros_like(road), False, 'YOLO outer left edge missing at preview'
        return output, True, ('current YOLO ego-connected corridor' if self.visual_feedback
                              else 'YOLO outer-left corridor with recorded width prior')
