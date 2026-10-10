"""Read completed run evidence; this never constructs a model or a chassis."""
import csv
import json
import math
from pathlib import Path
import re
import cv2
import numpy as np


class RunReplayStore:
    def __init__(self,directory,sensor_directory):
        self.directory=Path(directory).resolve();self.sensor_directory=Path(sensor_directory).resolve()

    def _run(self,session):
        if not isinstance(session,str) or not re.fullmatch('[0-9a-f]{32}',session):
            raise ValueError('invalid session id')
        folder=(self.directory/session).resolve()
        if folder.parent!=self.directory:raise ValueError('session path escaped')
        state=json.loads((folder/'session.json').read_text())
        if state.get('running') is not False:raise ValueError('wait until driving and recording finish')
        archive=state.get('run_archive')
        if not isinstance(archive,str):raise ValueError('no completed runtime archive')
        run=Path(archive).resolve()
        if run.parent!=(folder/'runtime/runs').resolve():raise ValueError('archive path escaped this session')
        return run,state

    @staticmethod
    def _rows(run):
        with (run/'onboard_log.csv').open() as stream:rows=list(csv.DictReader(stream))
        if not rows or len(rows)>12000 or [int(row['sample']) for row in rows]!=list(range(len(rows))):
            raise ValueError('control log is empty or incomplete')
        for row in rows:
            for key,value in row.items():
                if value=='':row[key]=None;continue
                if value in ('True','False'):row[key]=value=='True';continue
                try:
                    number=float(value);row[key]=number if math.isfinite(number) else None
                except (ValueError,TypeError):pass
        return rows

    def data(self,session):
        run,state=self._run(session)
        rows=self._rows(run)
        telemetry=[]
        try:
            with (run/'sensors/events.jsonl').open() as stream:
                telemetry=[event for event in map(json.loads,stream) if event.get('sensor')=='telemetry']
        except FileNotFoundError:pass
        position=0;current=None
        for row in rows:
            when=row.get('control_monotonic')
            if type(when) not in (int,float):continue
            while position<len(telemetry) and telemetry[position]['received_monotonic']<=when:
                current=telemetry[position];position+=1
            if current is not None and when-current['received_monotonic']<.5:
                row['telemetry']=current['samples']
        return dict(session_id=session,run_id=run.name,frames=rows,
            termination=state.get('termination'),sensor_recording=state.get('sensor_recording'),
            limitation='历史帧与当时日志回放；更改动作不会改变历史车辆轨迹。')

    def sessions(self):
        paths=sorted(self.directory.glob('*/session.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        result=[]
        for path in paths[:20]:
            if not re.fullmatch('[0-9a-f]{32}',path.parent.name):continue
            try:
                state=json.loads(path.read_text())
                if state.get('running') is False and state.get('run_archive'):
                    result.append(dict(session_id=path.parent.name,completed_at=state.get('completed_at'),
                        state=state.get('state'),run_id=Path(state['run_archive']).name))
            except (OSError,ValueError,TypeError):continue
        return result

    def _sensor(self,session,sample,sensor):
        run,_=self._run(session);rows=self._rows(run)
        if not 0<=sample<len(rows):raise ValueError('sample out of range')
        when=rows[sample].get('control_monotonic')
        if type(when) not in (int,float):raise ValueError('no sensor synchronization timestamp')
        root=(self.sensor_directory/'recordings'/run.name).resolve()
        if root.parent!=(self.sensor_directory/'recordings').resolve():raise ValueError('sensor path escaped')
        best=None
        with (root/'events.jsonl').open() as stream:
            for line in stream:
                event=json.loads(line);stamp=event.get('received_monotonic')
                if event.get('sensor')==sensor and type(stamp) in (int,float) and 0<=when-stamp<=.5:
                    if best is None or stamp>best['received_monotonic']:best=event
        if best is None:raise ValueError('no synchronized '+sensor+' sample')
        path=(root/best['array_path']).resolve()
        if path.parent!=root or path.suffix!='.npy':raise ValueError('sensor data path escaped')
        return np.load(path,allow_pickle=False),best,when-best['received_monotonic']

    def scan(self,session,sample):
        array,event,age=self._sensor(session,sample,'lidar')
        ranges=array[0];angles=event['angle_min']+np.arange(len(ranges))*event['angle_increment']
        valid=np.isfinite(ranges)&(ranges>0)&(ranges>=event['range_min'])&(ranges<=event['range_max'])
        return dict(age_s=age,points=len(ranges),xy_m=np.stack((ranges[valid]*np.cos(angles[valid]),
            ranges[valid]*np.sin(angles[valid])),axis=1).tolist())

    def image(self,session,sample,view):
        if view=='depth':
            array,event,_=self._sensor(session,sample,'depth')
            dtype='>u2' if event['is_bigendian'] else '<u2'
            depth=np.frombuffer(array.tobytes(),dtype=dtype).reshape(event['height'],event['step']//2)[:,:event['width']]
            frame=cv2.applyColorMap(np.uint8(np.clip(depth/4000*255,0,255)),cv2.COLORMAP_TURBO)
            frame[depth==0]=0
        else:
            if view not in ('raw','annotated','birdeye'):raise ValueError('unknown replay view')
            run,_=self._run(session)
            capture=cv2.VideoCapture(str(run/(view+'.mp4')))
            try:
                if not 0<=sample<int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):raise ValueError('sample out of range')
                capture.set(cv2.CAP_PROP_POS_FRAMES,sample);ok,frame=capture.read()
                if not ok:raise ValueError('replay frame unavailable')
            finally:capture.release()
        ok,jpg=cv2.imencode('.jpg',frame)
        if not ok:raise ValueError('replay JPEG failed')
        return jpg.tobytes()
