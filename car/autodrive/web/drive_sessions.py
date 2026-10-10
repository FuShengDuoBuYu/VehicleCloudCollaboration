"""Local session supervisor; HTTP supplies start/stop intent, never PWM/config.

Run separately from the restricted dashboard service. The fixed existing
experiment launcher owns camera/UART leases and restores monitors on exit.
Starting this supervisor alone does not open any hardware.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import subprocess
import threading
import time
import uuid

from autodrive.runtime.sensor_recording import atomic_json

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_DIRECTORY=ROOT/'outputs/autodrive_sessions'
DEFAULT_SOCKET=ROOT/'outputs/vehicle_dashboard/control.sock'


class DriveSessionManager:
    def __init__(self,directory=DEFAULT_DIRECTORY,command_factory=None,motors_enabled=False):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.motors_enabled=motors_enabled is True
        self.command_factory=command_factory or self._command
        self._lock=threading.RLock();self._closed=threading.Event()
        self._child=None;self._console=None;self._request=None
        self._state=dict(available=True,state='idle',running=False,session_id=None,
            directory=None,mode='hardware' if self.motors_enabled else 'dry-run',stopped_verified=False,
            runtime_limit_seconds=0)
        self._thread=threading.Thread(target=self._watch,name='panel-session-watch',daemon=True)
        self._thread.start()

    def _command(self,folder):
        result=['bash',str(ROOT/'run_vehicle_experiment.sh'),
            '--vehicle-profile','rosmaster_jetson_yolopv2_visual_feedback',
            '--output-dir',str(folder/'runtime'),'--panel-session-dir',str(folder/'control'),
            '--max-runtime-seconds','0']
        if self.motors_enabled:result.extend(['--field-trial','--enable-motors',
            '--confirm-motor-motion','I_UNDERSTAND_MOTORS_WILL_MOVE'])
        return result

    def get_state(self):
        with self._lock:return dict(self._state)

    def start(self,request_id):
        if not isinstance(request_id,str) or not 1<=len(request_id)<=100:
            raise ValueError('request_id required')
        with self._lock:
            if self._closed.is_set():raise ValueError('supervisor is closing')
            if request_id==self._request:return dict(self._state)
            if self._child is not None:raise ValueError('another session is active or closing')
            name=uuid.uuid4().hex;folder=self.directory/name;folder.mkdir(exist_ok=False)
            control=folder/'control';control.mkdir()
            atomic_json(control/'lease.json',dict(expires_monotonic=time.monotonic()+3.))
            self._request=request_id
            self._state=dict(available=True,state='starting',running=True,session_id=name,directory=str(folder),
                mode='hardware' if self.motors_enabled else 'dry-run',
                started_at=time.time(),stopped_verified=False,reason='operator clicked start',
                runtime_limit_seconds=0)
            atomic_json(folder/'session.json',self._state)
            command=self.command_factory(folder)
            atomic_json(folder/'command.json',dict(argv=command,mode=self._state['mode']))
            self._console=(folder/'console.log').open('x')
            try:self._child=subprocess.Popen(command,stdout=self._console,stderr=subprocess.STDOUT,
                start_new_session=True,cwd=str(ROOT))
            except Exception:
                self._console.close();self._console=None
                self._state.update(state='error',running=False,reason='runtime launch failed')
                atomic_json(folder/'session.json',self._state)
                raise
            return dict(self._state)

    def heartbeat(self,session_id):
        with self._lock:
            if session_id!=self._state['session_id'] or self._child is None:
                raise ValueError('heartbeat is not for the current session')
            if self._state['state']=='stopping':raise ValueError('session is stopping')
            atomic_json(Path(self._state['directory'])/'control/lease.json',
                        dict(expires_monotonic=time.monotonic()+3.))
            return dict(self._state)

    def stop(self,session_id=None,reason='operator clicked end'):
        with self._lock:
            if session_id is not None and session_id!=self._state['session_id']:
                raise ValueError('stop is not for the current session')
            if self._child is None:return dict(self._state)
            if self._state['state']!='stopping':
                atomic_json(Path(self._state['directory'])/'control/stop.json',
                            dict(reason=reason,requested_monotonic=time.monotonic()))
                self._state.update(state='stopping',reason=reason)
                try:self._child.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
            return dict(self._state)

    def _watch(self):
        while not self._closed.wait(.05):
            with self._lock:
                if self._child is None:continue
                folder=Path(self._state['directory'])
                code=self._child.poll()
                if code is not None:
                    self._console.close();self._console=None;self._child=None
                    try:runtime=json.loads((folder/'runtime/status.json').read_text())
                    except (OSError,ValueError):runtime={}
                    termination=runtime.get('termination') or {}
                    verified=termination.get('stopped') is True
                    self._state.update(state='finished' if code==0 and verified else 'error',
                        running=False,exit_code=code,stopped_verified=verified,
                        completed_at=time.time(),run_archive=runtime.get('run_archive'),
                        termination=termination,sensor_recording=runtime.get('sensor_recording'))
                    atomic_json(folder/'session.json',self._state)
                    continue
                try:
                    expiry=json.loads((folder/'control/lease.json').read_text())['expires_monotonic']
                    current=type(expiry) in (int,float) and time.monotonic()<=expiry<=time.monotonic()+4.
                except (OSError,ValueError,KeyError,TypeError):current=False
                if not current:self.stop(reason='panel control lease expired')
                elif self._state['state']=='starting' and (folder/'runtime/status.json').exists():
                    try:runtime=json.loads((folder/'runtime/status.json').read_text())
                    except (OSError,ValueError):continue
                    row=runtime.get('last_result') or {}
                    # A newly created status file can still describe model
                    # warmup with a several-second-old first CUDA result.
                    age=row.get('semantic_result_age_s')
                    maximum=(runtime.get('diagnostics') or {}).get('yolopv2_fusion',{}).get('max_result_age_seconds',.3)
                    if type(age) in (int,float) and 0<=age<=maximum:
                        self._state['state']='running'
                        atomic_json(folder/'session.json',self._state)

    def close(self):
        self.stop(reason='panel supervisor closing')
        deadline=time.monotonic()+15
        while self.get_state()['running'] and time.monotonic()<deadline:time.sleep(.05)
        self._closed.set();self._thread.join(timeout=1)


class SessionClient:
    def __init__(self,path=DEFAULT_SOCKET):self.path=str(path)
    def request(self,action,**values):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.settimeout(.8);connection.connect(self.path)
            connection.sendall((json.dumps(dict(action=action,**values))+'\n').encode())
            with connection.makefile('rb') as stream:response=json.loads(stream.readline(16385))
        if 'error' in response:raise ValueError(response['error'])
        return response
    def get_state(self):
        try:return self.request('state')
        except (OSError,ValueError):return dict(available=False,state='unavailable',running=False,
            reason='车端启停程序尚未就绪',mode=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket',default=str(DEFAULT_SOCKET))
    parser.add_argument('--output-dir',default=str(DEFAULT_DIRECTORY))
    parser.add_argument('--authorize-motion',default='',help='Explicitly permit future panel Start clicks')
    args=parser.parse_args()
    if args.authorize_motion not in ('','I_UNDERSTAND_MOTORS_WILL_MOVE'):
        parser.error('invalid motion authorization')
    path=Path(args.socket);path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix('.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():path.unlink()
    manager=DriveSessionManager(args.output_dir,motors_enabled=bool(args.authorize_motion))
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(1)
            try:
                request=json.loads(self.rfile.readline(16385))
                action=request.get('action')
                if action=='state':result=manager.get_state()
                elif action=='start':result=manager.start(request.get('request_id'))
                elif action=='stop':result=manager.stop(request.get('session_id'))
                elif action=='heartbeat':result=manager.heartbeat(request.get('session_id'))
                else:raise ValueError('unknown session intent')
            except (ValueError,TypeError,AttributeError,OSError) as exc:result=dict(error=str(exc))
            self.wfile.write((json.dumps(result,ensure_ascii=False)+'\n').encode())
    server=socketserver.ThreadingUnixStreamServer(str(path),Handler);server.daemon_threads=True
    os.chmod(path,0o600)
    def stop(*_):
        manager.stop(reason='supervisor interrupted')
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    print('Panel session supervisor ready; mode='+manager.get_state()['mode'],flush=True)
    try:server.serve_forever(poll_interval=.1)
    finally:
        manager.close();server.server_close();path.unlink(missing_ok=True);lock.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
