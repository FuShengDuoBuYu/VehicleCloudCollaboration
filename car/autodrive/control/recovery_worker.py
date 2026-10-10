"""One in-flight paid request; worker has no driver or actuator reference."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import cv2
from .decision_fusion import CloudAdvice, FusionInput, DecisionFusionAdapter, RuleBasedRecoveryPolicy


class RecoveryCoordinator:
    def __init__(self,config,client,output_dir,*,fusion_policy=None):
        self.output_dir=Path(output_dir)
        self.config=config
        self.fusion=DecisionFusionAdapter(fusion_policy if fusion_policy is not None else
                                         RuleBasedRecoveryPolicy(config,self.output_dir.parent.name))
        self.client=client;self._jobs=queue.Queue(maxsize=1);self._results=queue.Queue(maxsize=1)
        self._closed=threading.Event();self._busy=False;self._submitted=None;self._archive_error=False
        self._thread=threading.Thread(target=self._run,name='cloud-recovery',daemon=True)
        if config.enabled:
            if client is None:raise ValueError('recovery requires cloud client')
            self.output_dir.mkdir(parents=True,exist_ok=True);self._thread.start()

    @property
    def motion_deadline(self):return self.fusion.motion_deadline

    def filter(self,command,evidence,now,local_deadline=None):
        if self._closed.is_set() or self._archive_error:return self.veto('recovery unavailable/evidence write failed',now)
        while True:
            try:event_id,scene,error,completed=self._results.get_nowait()
            except queue.Empty:break
            self._busy=False
            if now-completed>self.config.response_max_age_seconds:error='stale worker result'
            self.fusion.receive(CloudAdvice(event_id,scene,now,error))
        decision=self.fusion.decide(FusionInput(command,evidence,now,local_deadline,self._busy))
        result=decision.command
        request=self.fusion.pending_request
        state=self.fusion.get_state()
        if state['phase']=='CLOUD_PENDING' and request is not None and request['context']['event_id']!=self._submitted and not self._busy:
            self._busy=True;self._submitted=request['context']['event_id'];self._jobs.put_nowait(request)
        # Every decision has its CURRENT sequence and candidates in a local journal.
        try:
            observation=evidence.get('observation') or {}
            self._append('decisions.jsonl',dict(now=now,sequence=observation.get('sequence'),
                captured_at=observation.get('captured_at'),hard_safe=evidence['hard_safe'],
                local_route_valid=evidence.get('local_route_valid'),
                stop_code=evidence['stop_code'],candidates=evidence['candidates'],
                phase=state['phase'],event_id=state['event_id'],fusion=asdict(decision),
                action=result.action,reason=result.reason,motion_deadline=self.motion_deadline,
                transitions=self.fusion.drain_transitions()))
        except Exception:
            self._archive_error=True;return self.veto('recovery decision archive failed',now)
        return result

    def veto(self,reason,now):return self.fusion.veto(reason,now).command

    def record_application(self,command,wheel_state,now):
        try:self._append('applications.jsonl',dict(now=now,event_id=self.fusion.get_state()['event_id'],
                    fusion=asdict(self.fusion.last_decision),
                    action=command.action,reason=command.reason,wheels=wheel_state))
        except Exception:
            self._archive_error=True
            return False
        return True

    def _append(self,name,value):
        with (self.output_dir/name).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

    @staticmethod
    def _write(path,value):
        path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')

    def _run(self):
        while not self._closed.is_set():
            try:request=self._jobs.get(timeout=.1)
            except queue.Empty:continue
            if self._closed.is_set():break
            context=request['context'];event_id=context['event_id'];folder=self.output_dir/event_id
            scene=None;error=None;response=dict(event_id=event_id,worker_started=time.monotonic())
            previous=getattr(self.client,'last_request_metadata',None);entered=False
            try:
                folder.mkdir(exist_ok=False)
                paths=[];manifest=[]
                for index,frame in enumerate(request['frames']):
                    path=folder/('input_%02d.png'%index)
                    if not cv2.imwrite(str(path),frame):raise OSError('input evidence write failed')
                    paths.append(path);manifest.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                self._write(folder/'request.json',dict(context=context,inputs=manifest))
                entered=True;result=self.client.request_scene(paths,context=context)
                # CloudClient redacts all raw response/events and metadata before returning.
                response['result']=asdict(result);scene=result.scene
                response['metadata']=getattr(self.client,'last_request_metadata',{})
            except Exception as exc:
                error=type(exc).__name__;response['error_type']=error
                metadata=getattr(self.client,'last_request_metadata',{})
                if entered and metadata is not previous:response['metadata']=metadata
            completed=time.monotonic();response['worker_finished']=completed
            try:self._write(folder/'response.json',response)
            except Exception:error='EvidenceWriteError'
            if not self._closed.is_set():self._results.put_nowait((event_id,scene,error,completed))

    def get_state(self):
        return dict(self.fusion.get_state(),pending_jobs=self._jobs.qsize(),worker_busy=self._busy,
            worker_alive=self._thread.is_alive(),closed=self._closed.is_set(),archive_error=self._archive_error)

    def close(self):
        self._closed.set()
        if self._thread.is_alive():self._thread.join(timeout=.2)
