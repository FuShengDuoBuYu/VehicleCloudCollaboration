"""One in-flight paid request; worker has no driver or actuator reference."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
import cv2
from .decision_fusion import CloudAdvice, FusionInput, DecisionFusionAdapter, RuleBasedRecoveryPolicy


class AsyncRecoveryJournal:
    """Bounded, ordered evidence queue. Never drop/replace a control record."""
    def __init__(self, output_dir, queue_records=256):
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self._queue=queue.Queue(maxsize=queue_records)
        self._closed=threading.Event()
        self._lock=threading.Lock()
        self._error=None;self._enqueued=0;self._written=0
        self._handles={}
        self._thread=threading.Thread(target=self._run,name='cloud-evidence',daemon=True)
        self._thread.start()

    def submit(self,name,value):
        with self._lock:
            if self._closed.is_set() or self._error is not None:return False
            try:self._queue.put_nowait((name,copy.deepcopy(value)))
            except queue.Full:
                self._error='QueueFull'
                return False
            self._enqueued+=1
        return True

    def _write_record(self,name,value):
        if name not in ('decisions.jsonl','applications.jsonl'):
            raise ValueError('invalid recovery journal')
        if name not in self._handles:
            self._handles[name]=(self.output_dir/name).open('a',encoding='utf-8')
        handle=self._handles[name]
        handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        handle.flush()

    def _run(self):
        try:
            while not self._closed.is_set() or not self._queue.empty():
                try:name,value=self._queue.get(timeout=.05)
                except queue.Empty:continue
                try:
                    self._write_record(name,value)
                    with self._lock:self._written+=1
                finally:self._queue.task_done()
        except Exception as exc:
            with self._lock:self._error=type(exc).__name__
        finally:
            for handle in self._handles.values():
                try:handle.close()
                except Exception as exc:
                    with self._lock:self._error=type(exc).__name__

    def get_state(self):
        with self._lock:
            return dict(enqueued=self._enqueued,written=self._written,error=self._error,
                queue_depth=self._queue.qsize(),queue_capacity=self._queue.maxsize,
                closed=self._closed.is_set() and not self._thread.is_alive())

    def close(self):
        self._closed.set()
        self._thread.join(timeout=5.)
        if self._thread.is_alive():
            with self._lock:self._error='DrainTimeout'
            raise RuntimeError('cloud evidence drain timed out')


class RecoveryCoordinator:
    def __init__(self,config,client,output_dir,*,fusion_policy=None):
        self.output_dir=Path(output_dir)
        self.config=config
        self.fusion=DecisionFusionAdapter(fusion_policy if fusion_policy is not None else
                                         RuleBasedRecoveryPolicy(config,self.output_dir.parent.name))
        self.client=client;self._jobs=queue.Queue(maxsize=1);self._results=queue.Queue(maxsize=1)
        self._closed=threading.Event();self._busy=False;self._submitted=None;self._archive_error=False
        self._decision_index=-1
        self.journal=None
        self._thread=threading.Thread(target=self._run,name='cloud-recovery',daemon=True)
        if config.enabled:
            if client is None:raise ValueError('recovery requires cloud client')
            self.output_dir.mkdir(parents=True,exist_ok=True)
            self.journal=AsyncRecoveryJournal(self.output_dir)
            self._thread.start()

    @property
    def motion_deadline(self):return self.fusion.motion_deadline

    def filter(self,command,evidence,now,local_deadline=None):
        if self.journal is not None and self.journal.get_state()['error'] is not None:self._archive_error=True
        if self._closed.is_set() or self._archive_error:return self.veto('recovery unavailable/evidence write failed',now)
        self._decision_index+=1
        received=[]
        while True:
            try:event_id,scene,error,completed=self._results.get_nowait()
            except queue.Empty:break
            self._busy=False
            if now-completed>self.config.response_max_age_seconds:error='stale worker result'
            advice=CloudAdvice(event_id,scene,now,error)
            received.append(asdict(advice))
            self.fusion.receive(advice)
        decision=self.fusion.decide(FusionInput(command,evidence,now,local_deadline,self._busy))
        result=decision.command
        request=self.fusion.pending_request
        state=self.fusion.get_state()
        if state['phase']=='CLOUD_PENDING' and request is not None and request['context']['event_id']!=self._submitted and not self._busy:
            self._busy=True;self._submitted=request['context']['event_id'];self._jobs.put_nowait(request)
        # Every decision has its CURRENT sequence and candidates in a local journal.
        try:
            observation=evidence.get('observation') or {}
            self._append('decisions.jsonl',dict(now=now,decision_index=self._decision_index,
                received_advice=received,local_command=asdict(command),local_deadline=local_deadline,
                worker_busy=self._busy,imu=evidence.get('imu'),
                sequence=observation.get('sequence'),
                captured_at=observation.get('captured_at'),hard_safe=evidence['hard_safe'],
                observation_ready=evidence.get('observation_ready'),
                motion_veto_reason=evidence.get('motion_veto_reason'),
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
                    decision_index=self._decision_index,
                    fusion=asdict(self.fusion.last_decision),
                    action=command.action,reason=command.reason,wheels=wheel_state))
        except Exception:
            self._archive_error=True
            return False
        return True

    def _append(self,name,value):
        if self.journal is None or not self.journal.submit(name,value):
            raise OSError('recovery journal unavailable or full')

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
            worker_alive=self._thread.is_alive(),closed=self._closed.is_set(),
            archive_error=self._archive_error or (self.journal is not None and self.journal.get_state()['error'] is not None),
            journal=None if self.journal is None else self.journal.get_state())

    def close(self):
        self._closed.set()
        if self._thread.is_alive():self._thread.join(timeout=.2)
        if self.journal is not None:self.journal.close()
