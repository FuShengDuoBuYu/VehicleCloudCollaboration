"""Current-evidence recovery. All mutation and motion decisions stay on the control thread."""
from collections import deque
from dataclasses import dataclass
import copy
import hashlib
import math
import uuid
import cv2
import numpy as np
from cloud_client.recovery_contract import validate_recovery
from .lane_centering import DifferentialDriveCommand
from .visual_feedback import VisualFeedbackController, finite


@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    frame_policy: str = 'sparse3'
    stall_seconds: float = 1.
    trigger_observations: int = 3
    resume_observations: int = 3
    observation_max_age_seconds: float = .6
    request_timeout_seconds: float = 30.
    response_max_age_seconds: float = 10.
    retry_seconds: float = 2.
    max_attempts: int = 3
    mask_change_threshold: float = .4
    step_seconds: float = .15
    step_yaw_degrees: float = 5.
    settle_seconds: float = .25
    yaw_sign: int = -1

    def __post_init__(self):
        if type(self.enabled) is not bool or self.frame_policy not in ('single','sparse3'):
            raise ValueError('invalid recovery enable/frame policy')
        for name in ('trigger_observations','resume_observations','max_attempts'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1: raise ValueError('invalid '+name)
        for name in ('stall_seconds','observation_max_age_seconds','request_timeout_seconds',
                     'response_max_age_seconds','retry_seconds','step_seconds','step_yaw_degrees','settle_seconds'):
            if not finite(getattr(self,name)) or getattr(self,name)<=0: raise ValueError('invalid '+name)
        if (self.step_seconds>.15 or self.step_yaw_degrees>5 or self.settle_seconds<.25
                or self.observation_max_age_seconds>.6 or self.max_attempts>3
                or not finite(self.mask_change_threshold) or not 0<self.mask_change_threshold<1
                or type(self.yaw_sign) is not int or self.yaw_sign not in (-1,1)):
            raise ValueError('recovery exceeds experimental bounds')


def stopped(reason):
    return DifferentialDriveCommand('stop',0.,0.,0.,0.,reason)


def build_evidence(observation,boundary,command,*,now,frame_age,external_allowed,
                   perception_valid,pivot_verified,imu_sample,local_state):
    """Recompute from matched raw masks, never the zeroed failed route fit.

    Gray/white floor must be supported by THIS model frame or the existing
    bounded-track observation. Pixel margins are not a metric body footprint.
    """
    evidence=dict(observation=observation,hard_safe=False,candidates=[],stop_code='hard_veto',imu={},
                  local_route_valid=local_state.get('local_safe') is True)
    if not isinstance(observation,dict) or boundary is None: return evidence
    sequence=observation.get('sequence');captured=observation.get('captured_at')
    if (type(sequence) is not int or not finite(captured) or not finite(now)
            or not 0<=now-captured<=.6 or sequence!=getattr(boundary,'semantic_sequence',None)
            or captured!=getattr(boundary,'semantic_captured_at',None)
            or not finite(frame_age) or not 0<=frame_age<=.25
            or external_allowed is not True or perception_valid is not True
            or getattr(boundary,'semantic_hard_safe',False) is not True
            or getattr(boundary,'yellow_hazard',False) is not False): return evidence
    from autodrive.perception.track_colors import observe_track_surface, classify_track_paint
    from autodrive.perception.visual_clearance import assess_visible_clearance, assess_rotation_clearance
    try:
        road=np.asarray(observation['mask']);lane=np.asarray(observation['lane_mask'])
        if (road.ndim!=2 or road.shape!=lane.shape or min(road.shape)<16
                or not np.isfinite(road).all() or not np.isfinite(lane).all()): return evidence
        # Broad current near-field detection veto covers both candidate types.
        for detection in observation['detections']:
            x1,y1,x2,y2=detection['box'];confidence=detection['confidence']
            if not all(finite(x) for x in (x1,y1,x2,y2,confidence)): return evidence
            if not (0<=x1<=x2<=1 and 0<=y1<=y2<=1 and 0<=confidence<=1): return evidence
            if confidence>=.3 and y2>.65 and x1<.8 and x2>.2: return evidence
        surface=observe_track_surface(road,observation['frame'])
        supported=((road>0)|(surface['mask']>0)).astype(np.uint8)
        paint=classify_track_paint(supported,lane,observation['frame'])
        # Only observed floor can reinterpret lane exclusions; all yellow remains.
        floor=(surface['model_floor_mask']>0)|(surface['mask']>0)
        allowed=(supported>0)&floor&(paint.yellow_mask==0)
        h,w=road.shape
        centerline=np.array([[w//2,int(h*.98)],[w//2,int(h*.78)]])
        forward=assess_visible_clearance(allowed,paint.yellow_mask,centerline,.06)
        rotation=assess_rotation_clearance(allowed,paint.yellow_mask)
    except (KeyError,ValueError,TypeError,cv2.error): return evidence
    sample=imu_sample if isinstance(imu_sample,dict) else {}
    radians=(sample.get('value') or {}).get('radians') if isinstance(sample.get('value'),dict) else None
    yaw=radians[2] if isinstance(radians,(tuple,list)) and len(radians)==3 else None
    stamp=sample.get('received_monotonic')
    imu_valid=(sample.get('valid') is True and sample.get('stale') is False
               and finite(yaw) and finite(stamp) and 0<=now-stamp<=.2)
    evidence.update(hard_safe=True,imu=dict(yaw_rad=yaw,imu_timestamp=stamp,imu_valid=bool(imu_valid)))
    if forward['path_safe'] and forward['forward_clear']:
        hsv=cv2.cvtColor(observation['frame'],cv2.COLOR_BGR2HSV)
        white=cv2.resize(((hsv[:,:,1]<=55)&(hsv[:,:,2]>=170)).astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)>.8
        near=(slice(int(h*.78),int(h*.98)+1),slice(int(w*.44),int(w*.56)+1))
        marking=bool(np.any(white[near] & (lane[near]>0)))
        evidence['candidates'].append(dict(candidate_id='forward-white' if marking else 'forward-gap',
            action='forward',assessment='white_marking' if marking else 'road_model_gap',
            description='当前近场中心短直行，当前road或有界地面支持，白色排除可重解释，黄线保留；范围图像高度78%-98%、中心左右6%',
            margin_px=forward['minimum_margin_px']))
    heading=rotation['heading_error']
    if pivot_verified is True and imu_valid and rotation['rotation_clear'] and finite(heading) and heading>.18:
        evidence['candidates'].append(dict(candidate_id='pivot-right',action='pivot-right',assessment='right_corner',
            description='当前位置小幅右旋；当前连通地面右方出口支持，近车旋转原点及黄线余量通过，需实时IMU',
            heading_error=heading,margin_px=rotation['minimum_margin_px']))
    controller=local_state.get('controller',{})
    stopped_at=controller.get('stopped_at')
    normal_wait=(local_state.get('visual_corner_phase')=='exit-confirm'
        or finite(stopped_at) and captured<=stopped_at+.25
        or local_state.get('local_safe') is True and local_state.get('stationary_startup_ready') is False)
    evidence['stop_code']=('local_driving' if command.action!='stop' else
                           'normal_wait' if normal_wait else 'perception_stall')
    return evidence


class RecoveryEngine:
    def __init__(self,config,run_id):
        self.config=config;self.run_id=run_id
        self.controller=VisualFeedbackController(config.step_seconds,math.radians(config.step_yaw_degrees),
            config.settle_seconds,config.observation_max_age_seconds,yaw_sign=config.yaw_sign)
        self.phase='LOCAL_DRIVING';self.reason='await local evidence';self.attempts=0
        self.request=None;self._event=None;self._reply=None;self._reply_at=None
        self._history=deque(maxlen=40);self._sequence=None;self._capture=None;self._now=None
        self._stall_at=None;self._stall_count=0;self._resume_count=0;self._retry_at=0.
        self._observe_after=None;self._selected=None;self._event_mask=None;self._event_capture=None
        self.transitions=[]

    @property
    def motion_deadline(self): return self.controller.get_state()['motion_deadline'] if self.phase=='PROBE_STEP' else None

    def _phase(self,phase,reason,now):
        if phase!=self.phase or reason!=self.reason:
            self.transitions.append(dict(at=now,phase=phase,reason=reason,attempts=self.attempts,event_id=self._event))
        self.phase=phase;self.reason=reason

    def veto(self,reason,now):
        was_probe=self.phase=='PROBE_STEP'
        observing=self.phase=='OBSERVE'
        self.controller.veto(reason,now=now)
        self._reply=None;self._selected=None;self._event=None;self.request=None
        self._stall_at=None;self._stall_count=0;self._resume_count=0
        self._retry_at=now+self.config.retry_seconds
        if was_probe:self._observe_after=now+self.config.settle_seconds
        self._phase('OBSERVE' if was_probe or observing else 'HOLD',reason,now)
        return stopped(reason)

    def _frames(self,current):
        selected=[current]
        if self.config.frame_policy=='sparse3':
            for seconds in (1.,2.):
                target=current['captured_at']-seconds
                choices=[x for x in self._history if x['sequence']<current['sequence'] and abs(x['captured_at']-target)<=.45]
                if choices:
                    nearest=min(choices,key=lambda x:abs(x['captured_at']-target))
                    if all(x['sequence']!=nearest['sequence'] for x in selected):selected.append(nearest)
        return sorted(selected,key=lambda x:x['sequence'])

    def _begin(self,evidence,now):
        obs=evidence['observation'];self.attempts+=1;self._event=str(uuid.uuid4())
        self._event_mask=np.array(obs['mask']>0,copy=True);self._event_capture=obs['captured_at']
        frames=self._frames(obs)
        context=dict(run_id=self.run_id,event_id=self._event,frame_id=str(obs['sequence']),
            candidate_version='local-recovery-v1',stop_code=evidence['stop_code'],
            stall_seconds=now-self._stall_at,attempt=self.attempts,frame_policy=self.config.frame_policy,
            candidates=copy.deepcopy(evidence['candidates']),
            frames=[dict(sequence=x['sequence'],captured_at=x['captured_at'],
                         relative_seconds=x['captured_at']-obs['captured_at']) for x in frames])
        import json
        context['candidate_sha256']=hashlib.sha256(json.dumps(context['candidates'],sort_keys=True,
            ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()
        context['road_mask_sha256']=hashlib.sha256(np.ascontiguousarray(obs['mask']).tobytes()).hexdigest()
        self.request=dict(context=context,frames=[x['frame'].copy() for x in frames],started=now)
        self._reply=None;self._retry_at=now+self.config.request_timeout_seconds
        self._phase('CLOUD_PENDING','request current stall',now)

    def receive(self,event_id,scene,now,error=None):
        if event_id!=self._event or self.phase!='CLOUD_PENDING':return
        if error or now>self._retry_at:
            self._reply=None;self._retry_at=now+self.config.retry_seconds
            self._phase('HOLD','cloud failed or timed out',now);return
        try:
            scene=validate_recovery(scene)
            selected=next((c for c in self.request['context']['candidates'] if c['candidate_id']==scene['candidate_id']),None)
            if (scene['recommendation']!='try_candidate' or selected is None
                    or selected['assessment']!=scene['assessment']):raise ValueError('selection mismatch/hold')
        except (ValueError,TypeError):
            self._retry_at=now+self.config.retry_seconds;self._phase('HOLD','cloud hold or invalid selection',now);return
        self._reply=copy.deepcopy(scene);self._reply_at=now
        self._phase('REVALIDATE','await newer current evidence',now)

    def tick(self,command,evidence,now,worker_busy=False):
        if not self.config.enabled:return command
        if not finite(now) or self._now is not None and now<self._now:
            return self.veto('recovery clock invalid',self._now or 0.)
        self._now=now
        obs=evidence.get('observation') or {};seq=obs.get('sequence');capture=obs.get('captured_at')
        fresh=(type(seq) is int and seq>=0 and finite(capture) and 0<=now-capture<=self.config.observation_max_age_seconds
               and (self._sequence is None or seq>=self._sequence) and (self._capture is None or capture>=self._capture))
        if not fresh or evidence.get('hard_safe') is not True:return self.veto('current evidence hard veto',now)
        new=(self._sequence is None or seq>self._sequence) and (self._capture is None or capture>self._capture)
        if new:
            self._sequence=seq;self._capture=capture
            if not self._history or capture-self._history[-1]['captured_at']>=.1:
                self._history.append(dict(sequence=seq,captured_at=capture,frame=obs['frame'].copy()))
            while self._history and capture-self._history[0]['captured_at']>3.:self._history.popleft()
        if self.phase in ('CLOUD_PENDING','REVALIDATE','PROBE_STEP'):
            mask=np.asarray(obs['mask'])>0
            if mask.shape!=self._event_mask.shape:
                return self.veto('scene changed',now)
            union=np.count_nonzero(mask|self._event_mask)
            diff=1.-np.count_nonzero(mask&self._event_mask)/max(1,union)
            if diff>self.config.mask_change_threshold:return self.veto('scene changed',now)
        if self.phase=='CLOUD_PENDING':
            if now>self._retry_at:
                self._retry_at=now+self.config.retry_seconds;self._phase('HOLD','cloud timeout',now)
            return stopped(self.reason)
        if self.phase in ('REVALIDATE','PROBE_STEP'):
            selected=next((c for c in evidence['candidates'] if c['candidate_id']==self._reply['candidate_id']
                           and c['assessment']==self._reply['assessment']),None)
            if selected is None or now-self._reply_at>self.config.response_max_age_seconds:
                return self.veto('selected candidate no longer valid',now)
            if self.phase=='REVALIDATE' and (not new or capture<=self._event_capture):return stopped(self.reason)
            decision=self.controller.update(now=now,semantic_sequence=seq,captured_at=capture,
                desired_action=selected['action'],local_valid=True,continuous=False,**evidence['imu'])
            if decision.action=='stop':
                if self.phase=='PROBE_STEP':
                    self._observe_after=now+self.config.settle_seconds;self._reply=None
                    self._retry_at=now+self.config.retry_seconds
                    self._phase('OBSERVE',decision.reason,now)
                return stopped(decision.reason)
            self._phase('PROBE_STEP','cloud selected current local candidate',now)
            return DifferentialDriveCommand(decision.action,1. if decision.action=='pivot-right' else 0.,
                    .3,-.3 if decision.action=='pivot-right' else .3,.5,self.reason,0.)
        if self.phase=='OBSERVE':
            if not new or capture<=self._observe_after:return stopped('await post-probe observation')
            self._phase('HOLD','probe finished; reassess local route',now)
        if self.attempts and new:
            # Normal local step/settle zeros are not failed route evidence.
            # Read this independently from the proposed motion command.
            if evidence.get('local_route_valid',command.action!='stop') is True:self._resume_count+=1
            else:self._resume_count=0
        if command.action!='stop':
            if self.attempts==0:
                self._stall_at=None;self._stall_count=0;self._phase('LOCAL_DRIVING','current local route',now);return command
            if self._resume_count>=self.config.resume_observations:
                self.attempts=0;self._event=None;self.request=None;self._stall_at=None;self._stall_count=0
                self._phase('LOCAL_DRIVING','stable local recovery',now);return command
            return stopped('confirm stable local recovery')
        if not self.attempts:self._resume_count=0
        if evidence.get('stop_code')!='perception_stall':
            self._stall_at=None;self._stall_count=0;return command
        if new:
            if self._stall_at is None:self._stall_at=now
            self._stall_count+=1
        if self.attempts>=self.config.max_attempts:
            self._phase('HOLD','event attempt budget exhausted',now);return stopped(self.reason)
        if self._stall_at is not None and now-self._stall_at>=self.config.stall_seconds and self._stall_count>=self.config.trigger_observations:
            if now>=self._retry_at and not worker_busy:self._begin(evidence,now)
        elif self.phase=='LOCAL_DRIVING':self._phase('STALL_CONFIRM','persistent perception stop confirmation',now)
        return stopped(self.reason)

    def get_state(self):
        return dict(enabled=self.config.enabled,phase=self.phase,reason=self.reason,event_id=self._event,
            selected_candidate_id=None if self._reply is None else self._reply['candidate_id'],
            attempts=self.attempts,frame_policy=self.config.frame_policy,motion_allowed=self.phase=='LOCAL_DRIVING',
            controller=self.controller.get_state(),candidate_version='local-recovery-v1')
