"""One in-flight request + one replaceable latest pending event; no actuation."""
from collections import Counter, deque
import copy
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import threading
import uuid

from .client import CloudClient
from .frames import ImageFrame


@dataclass
class SceneOutcome:
    job_id: str
    status: str
    context: dict
    result: object = None
    error: str = ''


class LatestSceneWorker:
    def __init__(self,client=None,evidence_dir=None):
        self.client=client if client is not None else CloudClient()
        self.evidence_dir=Path(evidence_dir) if evidence_dir is not None else None
        if self.evidence_dir is not None: self.evidence_dir.mkdir(parents=True,exist_ok=True)
        self._condition=threading.Condition();self._pending=None;self._latest=None;self._outcome=None;self._closed=False
        self._counts=Counter();self._records=deque(maxlen=256)
        self._thread=threading.Thread(target=self._run,name='cloud-scene-worker',daemon=True);self._thread.start()

    def _record(self,job,status):
        id_,frame,context=job
        if len(self._records)==self._records.maxlen: self._counts['record_overflow']+=1
        # Only local IDs are logged here; the client preserves redacted raw request/response.
        self._records.append(self._safe_context({'job_id':id_,'status':status,
            **{k:context.get(k) for k in ['event_id','frame_id','candidate_version']}}))

    def submit(self,frame,context=None):
        if not isinstance(frame,ImageFrame): raise ValueError('submit an immutable ImageFrame snapshot, not a mutable path/buffer')
        if context is not None and not isinstance(context,dict): raise ValueError('context must be an object')
        id_=str(uuid.uuid4());context=copy.deepcopy(context or {})
        context.setdefault('event_id',id_);context.setdefault('frame_id',str(uuid.uuid4()))
        job=(id_,frame,context)
        with self._condition:
            if self._closed: raise RuntimeError('cloud worker is closed')
            if self._pending is not None:
                self._counts['superseded_pending']+=1;self._record(self._pending,'superseded_pending')
            if self._outcome is not None:
                self._counts['discarded_result']+=1
                self._records.append({'job_id':self._outcome.job_id,'status':'discarded_unread_result'})
                self._outcome=None
            self._pending=job;self._latest=id_;self._counts['submitted']+=1;self._condition.notify()
        return id_

    def _safe_context(self,context):
        config=getattr(self.client,'config',None);key=getattr(config,'api_key','').strip()
        return CloudClient._redact(copy.deepcopy(context),key) if key else copy.deepcopy(context)

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed: self._condition.wait()
                if self._closed: return
                job=self._pending;self._pending=None;self._counts['started']+=1;self._record(job,'started')
            id_,frame,context=job
            try:
                result=self.client.request_scene(frame,context)
                outcome=SceneOutcome(id_,'completed',self._safe_context(context),result=result)
            except Exception as error:
                outcome=SceneOutcome(id_,'failed',self._safe_context(context),error='cloud request failed: '+type(error).__name__)
            if self.evidence_dir is not None:
                try:
                    record={'job_id':id_,'response_status':outcome.status,'context':outcome.context,
                        'result':asdict(outcome.result) if outcome.result is not None else None,
                        'error':outcome.error,'request':getattr(self.client,'last_request_metadata',{})}
                    with (self.evidence_dir/(id_+'.json')).open('x',encoding='utf-8') as file:
                        json.dump(self._safe_context(record),file,ensure_ascii=False,indent=2,allow_nan=False)
                except Exception as error:
                    # Retain no usable outcome if requested evidence could not be saved.
                    outcome=SceneOutcome(id_,'failed',outcome.context,error='evidence save failed: '+type(error).__name__)
            with self._condition:
                if self._closed or self._latest!=id_:
                    self._counts['discarded_result']+=1;self._record(job,'discarded_obsolete_result')
                else:
                    self._outcome=outcome;self._counts[outcome.status]+=1;self._record(job,outcome.status)

    def poll(self,*,event_id=None,frame_id=None,candidate_version=None):
        with self._condition:
            outcome=self._outcome;self._outcome=None
            if outcome is not None and any(expected is not None and outcome.context.get(k)!=expected
                    for k,expected in [('event_id',event_id),('frame_id',frame_id),('candidate_version',candidate_version)]):
                self._counts['discarded_result']+=1
                self._records.append({'job_id':outcome.job_id,'status':'discarded_context_mismatch'})
                return None
            return outcome

    @property
    def counters(self):
        with self._condition: return dict(self._counts)

    def drain_records(self):
        with self._condition:
            values=list(self._records);self._records.clear();return values

    def close(self,wait=False):
        with self._condition:
            self._closed=True;self._latest=None;self._outcome=None
            if self._pending is not None:
                self._record(self._pending,'cancelled_pending');self._counts['cancelled_pending']+=1;self._pending=None
            self._condition.notify_all()
        # Ongoing transport finishes under its configured timeout. Fast loop needn't wait.
        if wait: self._thread.join()
