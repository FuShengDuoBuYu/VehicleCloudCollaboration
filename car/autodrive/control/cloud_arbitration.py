"""Cloud advice is a stop/recovery veto, never a source of motor commands."""
from dataclasses import dataclass
import math
import uuid
import numpy as np
from cloud_client.schema import validate_scene

@dataclass(frozen=True)
class CloudArbitrationConfig:
    enabled: bool = False
    trigger_observations: int = 3
    resume_observations: int = 5
    observation_max_age_seconds: float = .3
    request_timeout_seconds: float = 8.
    response_max_age_seconds: float = 10.
    retry_seconds: float = 2.
    mask_change_threshold: float = .4

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('cloud enabled must be boolean')
        for name in ('trigger_observations','resume_observations'):
            value=getattr(self,name)
            if type(value) is not int or value<1:raise ValueError(name+' must be a positive integer')
        for name in ('observation_max_age_seconds','request_timeout_seconds','response_max_age_seconds','retry_seconds'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<=0:raise ValueError(name+' must be finite and positive')
        if isinstance(self.mask_change_threshold,bool) or not 0<self.mask_change_threshold<1:
            raise ValueError('mask_change_threshold must be inside (0,1)')

class SceneArbiter:
    """Single-control-thread state machine; network workers cannot release motion.

    Mask changes are an uncalibrated event proxy. A local stop always wins;
    advice can only release this additional veto after distinct safe frames.
    Route candidates remain stop-and-record until rerouting is implemented.
    """
    def __init__(self,config):
        self.config=config
        self.phase='driving'
        self.reason='cloud disabled' if not config.enabled else 'local driving'
        self.event=None
        self._last_sequence=None
        self._previous_mask=None
        self._event_mask=None
        self._bad=0
        self._stable=0
        self._retry_at=0.
        self._issued=0
        self._ignored=0

    @property
    def allowed(self):return not self.config.enabled or self.phase=='driving'

    @staticmethod
    def difference(first,second):
        if first.shape!=second.shape:return 1.
        union=np.count_nonzero(first|second)
        return 0. if union==0 else 1.-np.count_nonzero(first&second)/union

    def _request(self,observation,mask,reason,now):
        self._issued+=1
        self.event={'event_id':uuid.uuid4().hex,'sequence':int(observation['sequence']),
                    'captured_at':float(observation['captured_at']),'requested_at':now,
                    'reason':str(reason),'ordinal':self._issued}
        self._event_mask=mask.copy()
        self.phase='pending';self.reason='waiting for cloud scene advice';self._stable=0
        return dict(self.event)

    def _hold(self,reason,now):
        self.phase='hold';self.reason=reason;self._stable=0
        self._retry_at=now+self.config.retry_seconds

    def observe(self,observation,local_safe,reason,now):
        if not self.config.enabled:return None
        if not math.isfinite(now):raise ValueError('nonfinite arbiter time')
        if self.event is not None:
            if self.phase=='pending' and now-self.event['requested_at']>=self.config.request_timeout_seconds:
                self._hold('cloud request timed out',now)
            elif self.phase=='validating' and now-self.event['captured_at']>self.config.response_max_age_seconds:
                self._hold('cloud approval expired',now)
        if not local_safe:self._stable=0
        if observation is None:return None
        age=now-float(observation['captured_at'])
        if not math.isfinite(age) or age<0 or age>self.config.observation_max_age_seconds:
            self._stable=0;return None
        sequence=observation['sequence']
        if sequence==self._last_sequence:return None
        if self._last_sequence is not None and sequence<self._last_sequence:
            self._stable=0;return None
        raw=np.asarray(observation['mask'])
        if raw.ndim!=2 or not np.isfinite(raw).all():
            self._stable=0;return None
        mask=raw>0
        self._last_sequence=sequence
        changed=(self._previous_mask is not None and self.difference(mask,self._previous_mask)>self.config.mask_change_threshold)
        self._previous_mask=mask.copy()
        self._bad=0 if local_safe else self._bad+1
        if self.phase=='driving':
            if changed or self._bad>=self.config.trigger_observations:
                return self._request(observation,mask,'fresh mask changed' if changed else reason,now)
        elif self._event_mask is not None and self.difference(mask,self._event_mask)>self.config.mask_change_threshold:
            return self._request(observation,mask,'scene changed since cloud request',now)
        elif self.phase=='hold' and now>=self._retry_at:
            return self._request(observation,mask,'retry current stopped scene',now)
        elif self.phase=='validating':
            self._stable=self._stable+1 if local_safe else 0
            if self._stable>=self.config.resume_observations:
                self.phase='driving';self.reason='cloud candidate confirmed by fresh local observations'
                self.event=None;self._event_mask=None;self._bad=0
        return None

    def receive(self,event_id,scene,now,error=None):
        if self.phase!='pending' or self.event is None or event_id!=self.event['event_id']:
            self._ignored+=1;return False
        if (now-self.event['requested_at']>=self.config.request_timeout_seconds
                or now-self.event['captured_at']>self.config.response_max_age_seconds):
            self._hold('late cloud response rejected',now);return False
        if error:
            self._hold('cloud request failed: '+str(error),now);return False
        try:validate_scene(scene)
        except (ValueError,TypeError,KeyError):
            self._hold('invalid cloud scene schema',now);return False
        if scene['recommendation']!='resume_candidate':
            self._hold('cloud advice: '+scene['recommendation'],now);return False
        self.phase='validating';self.reason='cloud candidate awaiting fresh local confirmation';self._stable=0
        return True

    def get_state(self):
        return {'enabled':self.config.enabled,'phase':self.phase,'motion_allowed':self.allowed,
                'reason':self.reason,'event_id':None if self.event is None else self.event['event_id'],
                'stable_observations':self._stable,'request_count':self._issued,'ignored_responses':self._ignored}
