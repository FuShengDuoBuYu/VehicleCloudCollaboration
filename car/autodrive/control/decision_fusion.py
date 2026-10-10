"""Unified local/cloud decision boundary, independent of transport and actuators.

The default policy wraps the verified rule-based recovery state machine. A
replacement policy implements FusionPolicy, not the driver or provider API.
The adapter checks its output; final freshness, yaw control and hardware
interlocks remain mandatory. This is decision-level fusion, not learned
feature fusion or confidence-weighted averaging of motor commands.
"""
from dataclasses import asdict, dataclass, replace
from collections import deque
from typing import Optional, Protocol
from cloud_client.recovery_contract import validate_recovery
from .cloud_recovery import RecoveryEngine, stopped
from .lane_centering import DifferentialDriveCommand
from .visual_feedback import finite


@dataclass(frozen=True)
class FusionInput:
    local_command: DifferentialDriveCommand
    evidence: dict
    now: float
    local_deadline: Optional[float]
    worker_busy: bool = False


@dataclass(frozen=True)
class CloudAdvice:
    event_id: str
    scene: Optional[dict]
    received_at: float
    error: Optional[str] = None


@dataclass(frozen=True)
class FusionDecision:
    command: DifferentialDriveCommand
    source: str                  # local | cloud_recovery | hold
    reason: str
    deadline: Optional[float]
    event_id: Optional[str]
    candidate_id: Optional[str]
    sequence: Optional[int]
    policy_id: str


class FusionPolicy(Protocol):
    policy_id: str
    @property
    def pending_request(self): ...
    def receive(self, advice: CloudAdvice): ...
    def evaluate(self, inputs: FusionInput) -> FusionDecision: ...
    def veto(self, reason: str, now: float): ...
    def get_state(self) -> dict: ...
    def drain_transitions(self) -> list: ...


class RuleBasedRecoveryPolicy:
    policy_id='rule-recovery-v1'

    def __init__(self,config,run_id):
        self.engine=RecoveryEngine(config,run_id)

    @property
    def pending_request(self): return self.engine.request

    def receive(self,advice):
        self.engine.receive(advice.event_id,advice.scene,advice.received_at,error=advice.error)

    def evaluate(self,inputs):
        command=self.engine.tick(inputs.local_command,inputs.evidence,inputs.now,inputs.worker_busy)
        state=self.engine.get_state()
        source=('hold' if command.action=='stop' else
                'cloud_recovery' if state['phase']=='PROBE_STEP' else 'local')
        deadline=(self.engine.motion_deadline if source=='cloud_recovery' else
                  inputs.local_deadline if source=='local' else None)
        return FusionDecision(command,source,command.reason,deadline,state['event_id'],
            state['selected_candidate_id'] if source=='cloud_recovery' else None,
            (inputs.evidence.get('observation') or {}).get('sequence'),self.policy_id)

    def veto(self,reason,now):self.engine.veto(reason,now)
    def get_state(self):return self.engine.get_state()
    def drain_transitions(self):
        events=list(self.engine.transitions);self.engine.transitions.clear();return events


class DecisionFusionAdapter:
    """One decision format for local cruise, cloud-assisted steps, and all holds.

    Policies are trusted code requiring admission tests. Checks here prevent
    malformed outputs, unbacked cloud selection and expired/unbounded permits;
    they do not replace geometric/yaw validation or the final driver's guards.
    """
    def __init__(self,policy: FusionPolicy):
        self.policy=policy;self._advice=None;self._now=None;self._sequence=None
        self._active_cloud_event=None;self._cloud_deadline=None;self._spent_events=deque(maxlen=128)
        self.last_decision=FusionDecision(stopped('fusion not evaluated'),'hold','fusion not evaluated',
                                         None,None,None,None,policy.policy_id)

    @property
    def pending_request(self):return self.policy.pending_request
    @property
    def motion_deadline(self):return self.last_decision.deadline

    def receive(self,advice: CloudAdvice):
        # Binding is local. Provider text never defines event IDs or deadlines.
        request=self.pending_request
        if not request or advice.event_id!=request['context']['event_id']:return
        try:
            if advice.error or not finite(advice.received_at):raise ValueError('failed advice')
            scene=validate_recovery(advice.scene)
            self._advice=CloudAdvice(advice.event_id,dict(scene),advice.received_at)
        except (ValueError,TypeError):
            self._advice=None
            advice=CloudAdvice(advice.event_id,None,advice.received_at,'invalid or failed cloud advice')
        self.policy.receive(advice)

    def veto(self,reason,now):
        safe_now=now if finite(now) else (self._now or 0.)
        # A faulty plugin may not prevent an adapter-level stop.
        try:self.policy.veto(reason,safe_now)
        except Exception:pass
        self._advice=None
        self._active_cloud_event=None;self._cloud_deadline=None
        self.last_decision=FusionDecision(stopped(reason),'hold',reason,None,
            self.last_decision.event_id,None,self._sequence,self.policy.policy_id)
        return self.last_decision

    def decide(self,inputs: FusionInput) -> FusionDecision:
        observation=inputs.evidence.get('observation') or {}
        sequence=observation.get('sequence');captured=observation.get('captured_at')
        if (not finite(inputs.now) or self._now is not None and inputs.now<self._now
                or type(sequence) is not int or sequence<0
                or self._sequence is not None and sequence<self._sequence
                or not finite(captured) or not 0<=inputs.now-captured<=.6
                or inputs.evidence.get('hard_safe') is not True):
            return self.veto('fusion current evidence veto',inputs.now)
        self._now=inputs.now;self._sequence=sequence
        try:
            decision=self.policy.evaluate(inputs)
            if not isinstance(decision,FusionDecision):raise ValueError('missing fusion decision')
            command=decision.command
            if (not isinstance(command,DifferentialDriveCommand) or decision.policy_id!=self.policy.policy_id
                    or decision.sequence!=sequence or decision.source not in ('local','cloud_recovery','hold')
                    or not all(finite(x) for x in (command.left_speed,command.right_speed,command.steering,command.confidence))):
                raise ValueError('invalid decision contract')
            if command.action=='stop':
                decision=replace(decision,command=stopped(decision.reason),source='hold',deadline=None,candidate_id=None)
                self._active_cloud_event=None;self._cloud_deadline=None
            else:
                if not finite(decision.deadline) or decision.deadline<=inputs.now:raise ValueError('expired motion permit')
                if decision.source=='local':
                    if command!=inputs.local_command or not finite(inputs.local_deadline):raise ValueError('invented local command')
                    deadline=min(decision.deadline,inputs.local_deadline,inputs.now+.25,captured+.6)
                elif decision.source=='cloud_recovery':
                    advice=self._advice
                    request=self.pending_request
                    candidate=next((c for c in inputs.evidence['candidates']
                        if c['candidate_id']==decision.candidate_id),None)
                    if (advice is None or advice.event_id!=decision.event_id
                            or request is None or request['context']['event_id']!=decision.event_id
                            or not 0<=inputs.now-advice.received_at<=10.
                            or captured<=request['context']['frames'][-1]['captured_at']
                            or advice.scene['recommendation']!='try_candidate'
                            or advice.scene['candidate_id']!=decision.candidate_id
                            or candidate is None or candidate['action']!=command.action
                            or candidate['assessment']!=advice.scene['assessment']
                            or command.action not in ('forward','pivot-right')
                            or not 0<command.left_speed<=.3
                            or command.action=='forward' and (command.right_speed!=command.left_speed or command.steering!=0)
                            or command.action=='pivot-right' and not -.3<=command.right_speed<0):
                        raise ValueError('unbacked cloud command')
                    deadline=min(decision.deadline,inputs.now+.15,captured+.6)
                    if self._active_cloud_event==decision.event_id:
                        deadline=min(deadline,self._cloud_deadline)
                    elif decision.event_id in self._spent_events:
                        raise ValueError('cloud permit already consumed')
                    else:
                        self._active_cloud_event=decision.event_id;self._cloud_deadline=deadline
                        self._spent_events.append(decision.event_id)
                else:raise ValueError('hold cannot move')
                if deadline<=inputs.now:raise ValueError('expired bounded permit')
                decision=replace(decision,deadline=deadline)
        except Exception as error:
            return self.veto('fusion policy rejected: '+type(error).__name__,inputs.now)
        self.last_decision=decision
        return decision

    def get_state(self):
        return dict(self.policy.get_state(),fusion=asdict(self.last_decision))

    def drain_transitions(self):return self.policy.drain_transitions()
