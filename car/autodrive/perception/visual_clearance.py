"""Visible image-space road margins, never a metric wheel/body footprint.

No pixels are filled and no unseen boundary is extrapolated. A valid centre
point alone cannot grant motion beside a visible exclusion or missing road.
"""
import cv2
import numpy as np
from .track_colors import ROTATION_PARAMETERS,rotation_origin_region


def assess_rotation_clearance(corridor,yellow,margin_ratio=None):
    """Current near-origin support and direction, independent of forward fit."""
    result=dict(rotation_clear=False,minimum_margin_px=None,heading_error=None,
                required_margin_px=None,origin_x_px=None,origin_y_px=None)
    if corridor is None or yellow is None:
        return result
    road=np.asarray(corridor);paint=np.asarray(yellow)
    if (road.ndim!=2 or min(road.shape)<16 or road.shape!=paint.shape
            or not np.isfinite(road).all() or not np.isfinite(paint).all()):
        return result
    h,w=road.shape
    if margin_ratio is None:margin_ratio=ROTATION_PARAMETERS['minimum_margin_ratio']
    margin=max(2,int(np.ceil(w*margin_ratio)))
    ox=int(w*ROTATION_PARAMETERS['origin_x_ratio'])
    oy=min(h-1,int(h*ROTATION_PARAMETERS['origin_y_ratio']))
    result.update(required_margin_px=margin,origin_x_px=ox,origin_y_px=oy)
    allowed=((road>0)&(paint==0)).astype(np.uint8)
    # Direction must belong to the same current road as the near-car origin.
    # A detached model island cannot authorise a turn toward itself.
    _,labels=cv2.connectedComponents(allowed,8)
    origin=int(labels[oy,ox])
    if origin==0:
        return result
    allowed=(labels==origin).astype(np.uint8)
    distance=cv2.distanceTransform(np.pad(allowed,((0,0),(1,1))),cv2.DIST_L2,3)[:,1:-1]
    # Forward lookahead cannot represent in-place rotation. Require current
    # floor at the nearest visible origin and its own margin, without also
    # adding the forward rectangular footprint to that margin.
    clearance=float(distance[oy,ox])
    supported=float(np.mean(rotation_origin_region(allowed)))>=ROTATION_PARAMETERS['minimum_support_fraction']
    no_yellow=not np.any(rotation_origin_region(paint)>0)
    result.update(rotation_clear=bool(clearance>=margin and supported and no_yellow),minimum_margin_px=clearance)
    for y in range(int(h*.62),int(h*.88)+1):
        row=distance[y];maximum=float(row.max())
        if maximum<margin:continue
        choices=np.flatnonzero(row>=maximum-.1)
        target=int(choices[np.argmin(np.abs(choices-w*.5))])
        result['heading_error']=float((target-w*.5)/(w*.5))
        break
    return result


def assess_visible_clearance(corridor, yellow, centerline, margin_ratio=.06):
    result=dict(path_safe=False,forward_clear=False,minimum_margin_px=None,
                required_margin_px=None,reason='visual-clearance-evidence-missing')
    if corridor is None or yellow is None:
        return result
    road=np.asarray(corridor);marking=np.asarray(yellow);path=np.asarray(centerline)
    if (road.ndim!=2 or min(road.shape)<16 or road.shape!=marking.shape
            or path.ndim!=2 or path.shape[1]!=2 or not len(path)
            or not np.isfinite(road).all() or not np.isfinite(marking).all()
            or not np.isfinite(path).all() or np.any(path!=path.astype(int))):
        return result
    h,w=road.shape
    if np.any(path<0) or np.any(path[:,0]>=w) or np.any(path[:,1]>=h):
        return result
    margin=max(2,int(np.ceil(w*margin_ratio)))
    # Matched yellow stays forbidden even if a caller accidentally omits it
    # from the corridor. Border padding preserves finite image-space support.
    allowed=((road>0)&(marking==0)).astype(np.uint8)
    # Bottom image cutoff is behind the planning origin, not an observed
    # transverse obstacle. Lateral cutoffs still bound visible support.
    distance=cv2.distanceTransform(np.pad(allowed,((0,0),(1,1))),cv2.DIST_L2,3)[:,1:-1]
    bottom=int(path[:,1].max());near=int(h*.78)
    trace=np.zeros_like(allowed)
    cv2.polylines(trace,[path.astype(np.int32)],False,1,1)
    trace[:near]=0
    ys,xs=np.nonzero(trace)
    if not len(xs) or bottom<=near:
        return result
    clearance=float(distance[ys,xs].min())
    forward=allowed[near:bottom+1,w//2-margin:w//2+margin+1]
    result.update(path_safe=clearance>=margin,forward_clear=bool(forward.size and np.all(forward)),
        minimum_margin_px=clearance,required_margin_px=margin,
        reason='visual-clearance-ok' if clearance>=margin else 'visual-path-margin-insufficient')
    return result
