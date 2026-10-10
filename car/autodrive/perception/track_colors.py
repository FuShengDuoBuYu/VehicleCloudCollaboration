"""This track's white paint is traversable; yellow paint is excluded.

The optional bounded-track surface observation can support model dropouts.
Only the image actually consumed by the model may classify its markings.
"""
from dataclasses import dataclass
import cv2
import numpy as np


PARAMETERS = dict(white_hsv_lower=[0,0,170], white_hsv_upper=[179,55,255],
    yellow_hsv_lower=[18,85,135], yellow_hsv_upper=[42,255,255],
    pale_yellow_hsv_lower=[10,30,185], pale_yellow_hsv_upper=[42,255,255],
    minimum_yellow_component_pixels=4,
    minimum_pale_yellow_seed_pixels=4,
    minimum_white_fraction=.10, minimum_near_white_fraction=.45,
    minimum_road_fraction=.90, ego_top_ratio=.84, ego_half_width_ratio=.14,
    maximum_ego_yellow_ratio=.020)

SURFACE_PARAMETERS=dict(minimum_value=35,maximum_gray_value=185,maximum_saturation=110,
    blue_white_hue_lower=95,blue_white_hue_upper=140,blue_white_maximum_saturation=110,
    boundary_top_ratio=.30,boundary_bottom_ratio=.88,minimum_column_fraction=.55,
    maximum_side_span_ratio=.10)

# Nearest visible floor for an in-place step, separate from the forward path.
# These image-space trial settings are not a calibrated body footprint.
ROTATION_PARAMETERS=dict(origin_y_ratio=1.,origin_x_ratio=.50,
    support_top_ratio=.98,support_left_ratio=.49,support_right_ratio=.51,
    minimum_support_fraction=.90,minimum_margin_ratio=.05)


def rotation_origin_region(mask):
    h,w=mask.shape
    p=ROTATION_PARAMETERS
    return mask[int(h*p['support_top_ratio']):,
                int(w*p['support_left_ratio']):int(w*p['support_right_ratio'])+1]


def _yellow_mask(hsv,w,h):
    strong=cv2.inRange(hsv,tuple(PARAMETERS['yellow_hsv_lower']),tuple(PARAMETERS['yellow_hsv_upper']))
    # Bright glare can wash out this track's thin yellow frontier.  A second
    # range requires high brightness so dim warm-gray floor stays excluded.
    yellow=strong|cv2.inRange(hsv,tuple(PARAMETERS['pale_yellow_hsv_lower']),tuple(PARAMETERS['pale_yellow_hsv_upper']))
    # Filter only tiny isolated source-image colour specks, before dilation
    # magnifies them. A one-pixel-wide continuous line remains excluded.
    count,labels,stats,_=cv2.connectedComponentsWithStats(yellow,8)
    seeds=np.bincount(labels[strong>0],minlength=count)
    # Pale highlights alone can be warm white printing. Extend only a
    # currently observed yellow component, through actual pale pixels.
    keep=(stats[:,cv2.CC_STAT_AREA]>=PARAMETERS['minimum_yellow_component_pixels'])&(seeds>=PARAMETERS['minimum_pale_yellow_seed_pixels'])
    keep[0]=False
    yellow=keep[labels].astype(np.float32)
    yellow=cv2.resize(yellow,(w,h),interpolation=cv2.INTER_AREA)>0
    return cv2.dilate(yellow.astype(np.uint8),np.ones((3,3),np.uint8))>0


def observe_track_surface(road, frame):
    """Observe gray/white floor below a current saturated yellow boundary.

    No historical masks, missing-column extrapolation or hole closing. This
    is a track-specific visual observation, not model road or metric clearance.
    """
    road=np.asarray(road)
    if (road.ndim!=2 or min(road.shape)<16 or not np.isfinite(road).all()
            or frame is None or np.asarray(frame).ndim!=3 or frame.shape[2]!=3
            or frame.dtype!=np.uint8):
        raise ValueError('track surface requires matched finite masks and uint8 BGR frame')
    h,w=road.shape
    empty=dict(mask=np.zeros_like(road,dtype=np.uint8),observed=False,front_ratio=None)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    yellow=_yellow_mask(hsv,w,h)
    p=SURFACE_PARAMETERS
    ground=(hsv[:,:,1]<=p['maximum_saturation'])&(hsv[:,:,2]>=p['minimum_value'])
    # Camera white balance gives the observed white arrows a blue tint.
    # Retain that current colour evidence without admitting bright green.
    blue_white=((hsv[:,:,0]>=p['blue_white_hue_lower'])
        &(hsv[:,:,0]<=p['blue_white_hue_upper'])
        &(hsv[:,:,1]<=p['blue_white_maximum_saturation'])&(hsv[:,:,2]>=170))
    ground&=(hsv[:,:,2]<=p['maximum_gray_value'])|((hsv[:,:,1]<=55)&(hsv[:,:,2]>=170))|blue_white
    ground=cv2.resize(ground.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)>=.8
    model_floor=(ground&(road>0)).astype(np.uint8)
    empty['model_floor_mask']=model_floor
    probe=yellow.copy();probe[:int(h*p['boundary_top_ratio'])]=False
    probe[int(h*p['boundary_bottom_ratio']):]=False
    present=probe.any(axis=0)
    centre=present[int(w*.30):int(w*.85)]
    transverse=centre.mean()>=p['minimum_column_fraction'] and present.mean()>=.45
    def anchored_floor(observed):
        observed[:int(h*.50)]=False
        count,labels,stats,_=cv2.connectedComponentsWithStats(observed.astype(np.uint8),8)
        anchor=labels[int(h*.88):int(h*.96),int(w*.42):int(w*.58)]
        choices=[(int(np.sum(anchor==k)),k) for k in range(1,count)]
        if not choices or max(choices)[0]<anchor.size*.35:
            return None
        mask=(labels==max(choices)[1]).astype(np.uint8)
        return mask if mask.mean()>=.06 else None

    # Use the lowest observed yellow pixel in each column, so paint and its
    # far side remain excluded. Missing columns never receive extra road.
    edge=h-1-np.argmax(probe[::-1],axis=0)
    mask=(anchored_floor(ground&present[None,:]&(np.arange(h)[:,None]>edge[None,:])&~yellow)
          if transverse else None)
    if mask is None:
        # Two currently visible side markings plus existing model support
        # can explain texture/white-paint holes on a straight. No-model side
        # views alone cannot create floor, and a missing transverse column
        # never enters this alternative.
        left=probe[:, :int(w*.30)].sum();right=probe[:,int(w*.70):].sum()
        model_anchor=np.mean(road[int(h*.80):int(h*.96),int(w*.42):int(w*.58)]>0)
        if min(left,right)<h*.15 or model_anchor<.6:
            return empty
        # Add floor only between both side markings actually seen on this
        # row. Two blobs or the continuation beyond a line's end cannot
        # bound previously unrecognised floor.
        paired=0;interior=np.zeros_like(ground)
        for y in range(int(h*p['boundary_top_ratio']),int(h*p['boundary_bottom_ratio'])):
            left_x=np.flatnonzero(yellow[y,:int(w*.30)])
            right_x=np.flatnonzero(yellow[y,int(w*.70):])+int(w*.70)
            # A horizontal band's broad left/right fragments are not side
            # lines. Reject them instead of filling its missing centre columns.
            if (len(left_x) and len(right_x)
                    and left_x[-1]-left_x[0]<w*p['maximum_side_span_ratio']
                    and right_x[-1]-right_x[0]<w*p['maximum_side_span_ratio']):
                interior[y,left_x[-1]+1:right_x[0]]=True
                paired+=1
        if paired<h*.15:
            return empty
        mask=anchored_floor(ground&~yellow&((road>0)|interior))
    if mask is None:
        return empty
    return dict(mask=mask,observed=True,front_ratio=float(np.median(edge[present])/h),model_floor_mask=model_floor)


@dataclass(frozen=True)
class TrackPaintResult:
    exclusion_mask: np.ndarray
    allowed_white_pixels: int
    yellow_pixels: int
    ego_yellow_ratio: float
    yellow_mask: np.ndarray
    rotation_yellow_ratio: float


def classify_track_paint(road, lane, frame):
    road=np.asarray(road)>0;lane=np.asarray(lane)>0
    if road.ndim!=2 or road.shape!=lane.shape:
        raise ValueError('track paint requires matching road/lane masks')
    if (frame is None or np.asarray(frame).ndim!=3 or frame.shape[2]!=3
            or frame.dtype!=np.uint8):
        raise ValueError('track paint requires the consumed uint8 BGR frame')
    h,w=road.shape
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    white=cv2.inRange(hsv,tuple(PARAMETERS['white_hsv_lower']),tuple(PARAMETERS['white_hsv_upper']))
    # Area sampling keeps even a thin yellow pixel instead of losing it at
    # half resolution. No morphology closes a gap in the YOLO road mask.
    white=cv2.resize(white.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)>=127.5
    yellow=_yellow_mask(hsv,w,h)
    near_white=cv2.dilate(white.astype(np.uint8),np.ones((5,5),np.uint8))>0
    count,labels,stats,_=cv2.connectedComponentsWithStats(lane.astype(np.uint8),8)
    allowed=np.zeros_like(lane)
    for label in range(1,count):
        x,y,bw,bh,_=map(int,stats[label]);roi=np.s_[y:y+bh,x:x+bw];component=labels[roi]==label
        if (not np.any(yellow[roi][component])
                and np.mean(road[roi][component])>=PARAMETERS['minimum_road_fraction']
                and np.mean(white[roi][component])>=PARAMETERS['minimum_white_fraction']
                and np.mean(near_white[roi][component])>=PARAMETERS['minimum_near_white_fraction']):
            allowed[roi]|=component & road[roi]
    exclusion=((lane & ~allowed) | yellow).astype(np.uint8)
    half=int(w*PARAMETERS['ego_half_width_ratio'])
    ego=yellow[int(h*PARAMETERS['ego_top_ratio']):,w//2-half:w//2+half]
    rotation=rotation_origin_region(yellow)
    return TrackPaintResult(exclusion,int(allowed.sum()),int(yellow.sum()),float(np.mean(ego)),
                            yellow.astype(np.uint8),float(np.mean(rotation)))
