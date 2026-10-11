"""Recorded cloud transport for offline fusion; no network/client/hardware imports.

Replies may be reused only for the exact recorded request frames, mask and
candidate hash. Different proposed paths remain unmatched and stopped.
"""
import json
import hashlib
import math
from pathlib import Path
import re
import cv2
import numpy as np

from autodrive.control.decision_fusion import (
    CloudAdvice, FusionInput, DecisionFusionAdapter, RuleBasedRecoveryPolicy)


def read_records(path):
    path=Path(path)
    if not path.is_file():raise ValueError('missing cloud journal: '+path.name)
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if not isinstance(row,dict):raise ValueError('invalid cloud journal record')
        rows.append(row)
    return rows


def request_signature(context):
    return (context.get('candidate_version'),context.get('candidate_sha256'),
            bool(context.get('diagnostic_only',False)),
            context.get('road_mask_sha256'),context.get('frame_policy'),
            tuple((frame['sequence'],frame['captured_at']) for frame in context['frames']))


class RecordedRecoveryCoordinator:
    def __init__(self,config,run):
        self.config=config;self.run=Path(run).resolve()
        self.fusion=DecisionFusionAdapter(RuleBasedRecoveryPolicy(config,self.run.name))
        records=read_records(self.run/'cloud/decisions.jsonl')
        self.decisions=records
        self._events=[];self._current=None;self._submitted=None
        self._matched=0;self._unmatched=0;self._delivered=0
        self._used=set();self._now=None
        exact=any('received_advice' in row for row in records)
        self.timing_source='recorded control consumption' if exact else 'legacy worker completion approximation'
        advice={}
        for row in records:
            for item in row.get('received_advice',[]):
                stamp=float(item['received_at'])
                if not math.isfinite(stamp):raise ValueError('invalid recorded reply clock')
                advice[item['event_id']]=dict(item)
        for folder in sorted((self.run/'cloud').iterdir()):
            if not folder.is_dir():continue
            if not re.fullmatch(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',folder.name):continue
            if self.run not in folder.resolve().parents:raise ValueError('cloud event escaped archive')
            request_path=folder/'request.json'
            if not request_path.is_file():continue
            request=json.loads(request_path.read_text())
            context=request['context'];event=context['event_id']
            if event!=folder.name:raise ValueError('cloud event identity mismatch')
            images=[]
            manifest=request.get('inputs',[])
            if len(manifest)!=len(context['frames']):raise ValueError('missing recorded cloud input images')
            for item in manifest:
                name=item['file']
                if not re.fullmatch(r'input_[0-9]{2}\.png',name):raise ValueError('invalid recorded cloud image path')
                path=folder/name
                if folder.resolve() not in path.resolve().parents:raise ValueError('cloud image escaped archive')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('recorded cloud image hash mismatch')
                image=cv2.imread(str(path))
                if image is None:raise ValueError('invalid recorded cloud input image')
                images.append(path)
            reply=advice.get(event)
            response_path=folder/'response.json'
            response=json.loads(response_path.read_text()) if response_path.is_file() else {}
            if response and response.get('event_id')!=event:raise ValueError('cloud response identity mismatch')
            if reply is None and not exact and response:
                stamp=float(response['worker_finished'])
                if not math.isfinite(stamp):raise ValueError('invalid worker clock')
                reply=dict(received_at=stamp,scene=(response.get('result') or {}).get('scene'),
                           error=response.get('error_type'))
            if reply is not None and response and reply.get('scene')!=(response.get('result') or {}).get('scene'):
                raise ValueError('recorded cloud reply/journal mismatch')
            self._events.append(dict(event_id=event,signature=request_signature(context),reply=reply,images=images))

    @property
    def motion_deadline(self):return self.fusion.motion_deadline

    @staticmethod
    def _images_match(paths,frames):
        return len(paths)==len(frames) and all(np.array_equal(cv2.imread(str(path)),frame)
                                             for path,frame in zip(paths,frames))

    def filter(self,command,evidence,now,local_deadline=None):
        if not math.isfinite(now) or self._now is not None and now<self._now:
            raise ValueError('recorded control clock reversed')
        self._now=now
        busy=False
        if self._current is not None:
            reply=self._current['reply']
            if reply is not None and now>=reply['received_at']:
                self.fusion.receive(CloudAdvice(self._submitted,reply.get('scene'),
                    reply['received_at'],reply.get('error')))
                self._current=None;self._delivered+=1
            else:busy=True
        decision=self.fusion.decide(FusionInput(command,evidence,now,local_deadline,busy))
        request=self.fusion.pending_request
        if (self.fusion.get_state()['phase']=='CLOUD_PENDING' and request is not None
                and request['context']['event_id']!=self._submitted and not busy):
            self._submitted=request['context']['event_id']
            signature=request_signature(request['context'])
            match=next((e for e in self._events if e['event_id'] not in self._used
                        and e['signature']==signature
                        and self._images_match(e['images'],request['frames'])),None)
            if match is None:self._unmatched+=1
            else:
                self._used.add(match['event_id']);self._matched+=1
                self._current=match
        return decision.command

    def veto(self,reason,now):return self.fusion.veto(reason,now).command

    def get_state(self):
        return dict(self.fusion.get_state(),recorded_replay=dict(
            matched_events=self._matched,unmatched_events=self._unmatched,
            delivered_replies=self._delivered,available_events=len(self._events),
            timing_source=self.timing_source,network_requests=0))
