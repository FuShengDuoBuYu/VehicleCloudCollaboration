"""Bounded asynchronous raw sensor archives, separate from the live display."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import numpy as np


def raw_sensor_recording_ready(local,ros,run_id,now,publisher):
    """A previously successful read never authorizes indefinitely stale recording."""
    if not all(isinstance(state,dict) for state in (local,ros,publisher)):
        return False
    def fresh(stamp,limit):
        return (type(now) in (int,float) and math.isfinite(now)
                and type(stamp) in (int,float) and math.isfinite(stamp)
                and 0<=now-stamp<limit)
    local_times=local.get('last_sample_monotonic') or {}
    ros_times=ros.get('last_sample_monotonic') or {}
    return bool(local.get('closed') is False and local.get('error') is None
        and local.get('dropped')==0 and local.get('written',{}).get('uart_rx',0)>0
        and fresh(local_times.get('uart_rx'),.5)
        and publisher.get('running') is True and publisher.get('error') is None
        and ros.get('session_id')==run_id and ros.get('active') is True
        and ros.get('closed') is False and ros.get('error') is None and ros.get('dropped')==0
        and fresh(ros.get('published_monotonic'),1.)
        and all(ros.get('written',{}).get(name,0)>0 and fresh(ros_times.get(name),.5)
                for name in ('lidar','depth')))


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.%s.tmp'%os.getpid())
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(str(temporary), str(path))


class RunSensorWriter:
    def __init__(self, directory, queue_size=64):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue=queue.Queue(maxsize=queue_size)
        self._lock=threading.Lock()
        self._written=Counter();self._last_written={};self._dropped=0;self._error=None;self._closed=False
        self._thread=threading.Thread(target=self._run, name='run-sensors', daemon=True)
        self._thread.start()

    def submit(self, sensor, metadata, array=None):
        if sensor not in ('depth','lidar','uart_rx','uart_tx','telemetry'):
            raise ValueError('unknown sensor archive stream')
        with self._lock:
            if self._closed or self._error is not None:return False
            try:
                self.queue.put_nowait((sensor, dict(metadata), None if array is None else array.copy()))
                return True
            except queue.Full:
                self._dropped+=1
                return False

    def _run(self):
        index=0
        try:
            with (self.directory/'events.jsonl').open('x') as stream:
                while True:
                    entry=self.queue.get()
                    try:
                        if entry is None:break
                        sensor,metadata,array=entry
                        row=dict(metadata,sensor=sensor,index=index)
                        if array is not None:
                            filename='%08d-%s.npy'%(index,sensor)
                            np.save(self.directory/filename,array,allow_pickle=False)
                            row['array_path']=filename
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        stream.flush()
                        with self._lock:
                            self._written[sensor]+=1
                            self._last_written[sensor]=metadata.get('received_monotonic',metadata.get('sent_monotonic'))
                        index+=1
                    except Exception as exc:
                        with self._lock:
                            self._error=type(exc).__name__+': '+str(exc)
                            self._dropped+=1
                    finally:self.queue.task_done()
        except Exception as exc:
            with self._lock:self._error=type(exc).__name__+': '+str(exc)

    def get_state(self):
        with self._lock:
            return dict(directory=str(self.directory),written=dict(self._written),
                        last_sample_monotonic=dict(self._last_written),
                        dropped=self._dropped,error=self._error,closed=self._closed)

    def close(self):
        with self._lock:
            if self._closed:return
            self._closed=True
        while self._thread.is_alive():
            try:self.queue.put(None,timeout=.1);break
            except queue.Full:continue
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            with self._lock:self._error='sensor archive drain timeout'
        atomic_json(self.directory/'status.json',self.get_state())


class RosRunRecording:
    """The existing ROS owner records full scans/depth after a run request.

    Acquisition timestamps and receipt timestamps remain distinct. Display
    decimation is never used as the recorded raw scan or depth source.
    """
    def __init__(self, sensor_directory, boot):
        self.root=Path(sensor_directory);self.boot=boot
        self.writer=None;self.session=None;self.seen=set();self._publish_at=0.

    def poll(self):
        import re
        now=time.monotonic()
        try:
            request=json.loads((self.root/'recording_request.json').read_text())
            name=request.get('session_id')
            active=(request.get('active') is True and request.get('boot_id')==self.boot
                and isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9_]{1,100}',name)
                and type(request.get('expires_monotonic')) in (int,float)
                and now<=request['expires_monotonic']<=now+5.)
        except (OSError,ValueError,TypeError):active=False;name=None
        if self.writer is not None and (not active or name!=self.session):
            self.close()
        if active and self.writer is None and name not in self.seen:
            self.seen.add(name);self.session=name
            try:self.writer=RunSensorWriter(self.root/'recordings'/name)
            except OSError:
                atomic_json(self.root/'recording_state.json',dict(session_id=name,
                    error='cannot create unique sensor run',active=False))
        if self.writer is not None and now>=self._publish_at:
            atomic_json(self.root/'recording_state.json',dict(self.writer.get_state(),
                session_id=self.session,active=True,published_monotonic=now,boot_id=self.boot))
            self._publish_at=now+.1

    def submit(self,sensor,metadata,array):
        if self.writer is not None:self.writer.submit(sensor,metadata,array)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            atomic_json(self.root/'recording_state.json',dict(self.writer.get_state(),
                session_id=self.session,active=False,published_monotonic=time.monotonic(),boot_id=self.boot))
            self.writer=None
