"""Read-only vehicle dashboard. No device opens, process starts or cloud requests."""
import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import threading
import time
from urllib.parse import urlsplit, parse_qs

LIMIT = 1024 * 1024
# User-provided voltage endpoints (2026-10-09), not a measured SOC curve.
BATTERY_EMPTY_V = 10.0
BATTERY_FULL_V = 12.0
SENSOR_NAMES = ('battery', 'camera', 'lidar', 'depth', 'imu', 'attitude', 'encoder')
SOURCES = {'serial': ('battery', 'imu', 'attitude', 'encoder'), 'ros': ('lidar', 'depth'), 'camera': ('camera',)}
VALUE_FIELDS = {
    'battery': ('voltage_v', 'percentage'), 'camera': ('width', 'height', 'fps', 'source'),
    'lidar': ('fps', 'valid_points', 'points', 'nearest_m', 'xy_m'),
    'depth': ('fps', 'width', 'height', 'valid_fraction', 'nearest_m', 'median_m'),
    'imu': ('gyro_rad_s', 'accel_m_s2', 'mag_sdk_units'),
    'attitude': ('radians',), 'encoder': ('native_ticks',),
}
CLOUD_FIELDS = ('status', 'event_id', 'request_id', 'trigger_reason', 'triggered_at', 'started_at',
                'completed_at', 'latency_ms', 'provider', 'model', 'arbitration', 'error_code')
SCENE_FIELDS = ('scene_summary', 'road_state', 'risk_level', 'recommendation', 'route_hint',
                'uncertainties', 'reason', 'objects', 'signs')


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value[:1024]]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def publish_snapshot(path, sensors, **extra):
    """Atomic producer heartbeat; each sensor carries its original sample age."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=1, boot_id=boot_id(), monotonic_s=time.monotonic(),
                   published_at=time.time(), pid=os.getpid(), sensors=sensors)
    payload.update(extra)
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(finite(payload), ensure_ascii=False, allow_nan=False))
    os.replace(str(temporary), str(path))


def _read_json(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError('oversized snapshot')
    value = json.loads(raw, parse_constant=lambda value: None)
    if not isinstance(value, dict):
        raise ValueError('snapshot must be an object')
    return finite(value)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _only(value, fields):
    return {key: deepcopy(value[key]) for key in fields if key in value} if isinstance(value, dict) else {}


def valid_sensor_value(name, value):
    def vector(key, length):
        v=value.get(key)
        return isinstance(v,list) and len(v)==length and all(_number(x) for x in v)
    if name=='battery':return _number(value.get('voltage_v')) and 0<value['voltage_v']<60
    if name=='imu':return vector('gyro_rad_s',3) and vector('accel_m_s2',3)
    if name=='attitude':return vector('radians',3)
    if name=='encoder':return vector('native_ticks',4) and all(type(x) is int for x in value['native_ticks'])
    if name=='camera':return isinstance(value.get('source'),str)
    if name=='lidar':
        return (type(value.get('valid_points')) is int and value['valid_points']>0
            and type(value.get('points')) is int and value['points']>=value['valid_points']
            and _number(value.get('nearest_m')) and value['nearest_m']>0
            and isinstance(value.get('xy_m'),list) and len(value['xy_m'])<=360
            and all(isinstance(v,list) and len(v)==2 and all(_number(x) for x in v) for v in value['xy_m']))
    if name=='depth':
        return (type(value.get('width')) is int and value['width']>0
            and type(value.get('height')) is int and value['height']>0
            and _number(value.get('valid_fraction')) and 0<value['valid_fraction']<=1
            and _number(value.get('nearest_m')) and value['nearest_m']>0
            and _number(value.get('median_m')) and value['median_m']>=value['nearest_m'])
    return False


def sensor_empty(status='not_integrated', reason='尚无实时发布器'):
    return dict(status=status, value=None, age_s=None, reason=reason)


class DashboardState:
    def __init__(self, output_dir, sensor_dir, additional_runtime_dirs=()):
        self.output_dir = Path(output_dir)
        self.runtime_dirs = [self.output_dir] + [Path(p) for p in additional_runtime_dirs]
        self.preferred_runtime = None
        self.sensor_dir = Path(sensor_dir)
        self._boot = boot_id()
        self._boot_wall = time.time() - time.monotonic()
        self._system_cache = None
        self._system_at = 0
        self._system_lock = threading.Lock()

    def runtime_directory(self):
        if self.preferred_runtime is not None and (self.preferred_runtime/'status.json').is_file():
            return self.preferred_runtime
        found = []
        for folder in self.runtime_dirs:
            try:
                found.append((folder.joinpath('status.json').stat().st_mtime, folder))
            except OSError:
                pass
        return max(found, key=lambda entry: entry[0])[1] if found else self.output_dir

    def _sensors(self):
        sensors = {name: sensor_empty() for name in SENSOR_NAMES}
        for source, names in SOURCES.items():
            path = self.sensor_dir / (source + '.json')
            try:
                document = _read_json(path)
                stamp = document.get('monotonic_s')
                if not _number(stamp) or stamp > time.monotonic() + .1:
                    raise ValueError('invalid monotonic timestamp')
                elapsed = max(0, time.monotonic() - stamp)
                same_boot = document.get('boot_id') == self._boot
                data = document.get('sensors')
                if not isinstance(data, dict):
                    raise ValueError('invalid sensors object')
                for name in names:
                    item = data.get(name)
                    if not isinstance(item, dict):
                        continue
                    status = item.get('status', 'error')
                    if status not in ('online', 'stale', 'error', 'busy', 'disabled', 'stopped', 'waiting', 'not_integrated'):
                        status = 'error'
                    age = item.get('age_s')
                    age = age + elapsed if _number(age) and age >= 0 else None
                    value = _only(item.get('value'), VALUE_FIELDS[name]) or None
                    reason = str(item.get('reason', ''))[:200]
                    if not same_boot or elapsed > 3 or (age is not None and age > 3):
                        status, reason = 'stale', '发布器或采样已过期'
                    elif status == 'online' and (age is None or value is None):
                        status, reason = 'error', '缺少有效采样值或时间'
                    if name == 'battery' and value is not None:
                        value.update(percentage=None, percentage_estimated=True,
                            empty_voltage_v=BATTERY_EMPTY_V, full_voltage_v=BATTERY_FULL_V)
                        if not _number(value.get('voltage_v')) or not 0 < value['voltage_v'] < 60:
                            status, reason = 'error', '电压读数无效'
                        elif status == 'online':
                            value['percentage'] = round(max(0.0, min(100.0,
                                (value['voltage_v']-BATTERY_EMPTY_V)
                                / (BATTERY_FULL_V-BATTERY_EMPTY_V)*100)), 1)
                    if status == 'online' and not valid_sensor_value(name, value or {}):
                        status, reason = 'error', '传感器采样值无效'
                    sensors[name] = dict(status=status, value=value, age_s=age, reason=reason)
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, OverflowError):
                for name in names:
                    sensors[name] = sensor_empty('error', '无法读取有效传感器快照')
        for name, device in (('camera','/dev/video0'), ('lidar','/dev/rplidar'), ('battery','/dev/myserial')):
            sensors[name]['device_present'] = Path(device).exists()
        return sensors

    def _system(self):
        with self._system_lock:
            if self._system_cache is not None and time.monotonic()-self._system_at < 2:
                return deepcopy(self._system_cache)
            memory = {}
            try:
                for line in Path('/proc/meminfo').read_text().splitlines():
                    key, value = line.split(':', 1)
                    if key in ('MemTotal', 'MemAvailable'):
                        memory[key] = int(value.strip().split()[0]) * 1024
            except (OSError, ValueError):
                pass
            temperatures = {}
            for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
                try:
                    name = (zone/'type').read_text().strip()
                    value = float((zone/'temp').read_text())/1000
                    if -20 <= value <= 150:
                        temperatures[name] = value
                except (OSError, ValueError):
                    pass
            disk = shutil.disk_usage(self.output_dir if self.output_dir.exists() else Path.cwd())
            self._system_cache = dict(host=os.uname().nodename, uptime_s=time.monotonic(),
                memory_total_bytes=memory.get('MemTotal'), memory_available_bytes=memory.get('MemAvailable'),
                disk_free_bytes=disk.free, disk_total_bytes=disk.total, temperatures_c=temperatures,
                load_average=list(os.getloadavg()))
            self._system_at = time.monotonic()
            return deepcopy(self._system_cache)

    def get_state(self):
        folder = self.runtime_directory()
        runtime = {}
        age = None
        status = 'not_started'
        try:
            path = folder/'status.json'
            modified = path.stat().st_mtime
            age = time.time()-modified
            runtime = _read_json(path)
            status = 'live'
            if modified < self._boot_wall or age > 2.5:
                status = 'stale'
            if age < -.1:
                status = 'error'
            if isinstance(runtime.get('termination'), dict) and runtime['termination'].get('stopped'):
                status = 'stopped'
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            status = 'error'
        row = runtime.get('last_result') or {}
        if not isinstance(row, dict):
            row = {}
            status = 'error'
        sensors = self._sensors()
        frames = {}
        for name, filename in (('annotated','latest.jpg'), ('birdeye','latest_birdeye.jpg')):
            try:
                frame_age = time.time()-(folder/filename).stat().st_mtime
                frames[name] = dict(available=True, age_seconds=frame_age,
                    current_run=status=='live' and 0 <= frame_age < 2.5,
                    live=status=='live' and 0 <= frame_age < 2.5)
            except OSError:
                frames[name] = dict(available=False, age_seconds=None, current_run=False, live=False)
        if runtime:
            frame_age = row.get('frame_age_s')
            camera_status = 'online' if status == 'live' else status
            if camera_status == 'online' and (not _number(frame_age) or frame_age > .5):
                camera_status = 'stale'
            sensors['camera'] = dict(status=camera_status, age_s=(max(0,age)+frame_age if _number(frame_age) else None),
                value={'source':runtime.get('source'), 'fps': None}, reason='相机状态来自运行时；图像发布频率不等于采集帧率',
                device_present=Path('/dev/video0').exists())
        cloud_source = runtime.get('cloud')
        arbitration = runtime.get('cloud_arbitration')
        if not isinstance(cloud_source, dict) and isinstance(arbitration, dict):
            phase = arbitration.get('phase')
            cloud_source = dict(status=('disabled' if not arbitration.get('enabled') else
                {'pending':'pending','validating':'completed','hold':'hold','driving':'idle'}.get(phase,'unknown')),
                event_id=arbitration.get('event_id'), arbitration=_only(arbitration,
                    ('phase','reason','motion_allowed','stable_observations','request_count','ignored_responses')))
            event_id = arbitration.get('event_id')
            archive = runtime.get('run_archive')
            if isinstance(event_id,str) and re.fullmatch(r'[a-f0-9]{32}',event_id) and isinstance(archive,str):
                # Read only the named event under this configured runtime's runs.
                run = Path(archive).resolve()
                runs_root = (folder/'runs').resolve()
                if run.parent == runs_root:
                    event_folder = run/'cloud'/event_id
                    try:
                        request = _read_json(event_folder/'request.json')
                        if request.get('event_id') == event_id:
                            cloud_source['trigger_reason'] = request.get('reason')
                        response = _read_json(event_folder/'response.json')
                        if response.get('event_id') == event_id:
                            cloud_source['result'] = response.get('scene')
                            metadata = response.get('metadata') or {}
                            if isinstance(metadata,dict):
                                cloud_source.update(request_id=metadata.get('request_id'),
                                    latency_ms=metadata.get('elapsed_ms'),provider=metadata.get('provider'),
                                    model=metadata.get('requested_model'))
                            cloud_source['error_code'] = response.get('error_type')
                    except (OSError,ValueError,TypeError):
                        pass  # A pending event may not yet have an atomic response.
        if isinstance(cloud_source, dict):
            cloud = _only(cloud_source, CLOUD_FIELDS)
            cloud['result'] = _only(cloud_source.get('result'), SCENE_FIELDS)
            # Drop arbitrary nested provider contents; only documented scene fields leave the service.
            cloud['result']['objects'] = [_only(v, ('category','position','blocks_corridor','evidence')) for v in cloud['result'].get('objects',[])[:100] if isinstance(v,dict)] if isinstance(cloud['result'].get('objects',[]),list) else []
            cloud['result']['signs'] = [_only(v, ('text','meaning','applies_to_ego','evidence')) for v in cloud['result'].get('signs',[])[:100] if isinstance(v,dict)] if isinstance(cloud['result'].get('signs',[]),list) else []
            cloud['status'] = cloud.get('status', 'unknown')
        else:
            cloud = {'status':'not_integrated', 'result':None}
        cloud.update(confidence=None, source_status=status, age_s=age)
        navigation = _only(runtime.get('navigation'), ('status','task','destination','route_constraint','local_target','frame_id'))
        if not navigation:
            navigation = dict(status='not_integrated', task=None, destination=None, local_target=None,
                reason='尚无全局定位/目标导航发布；局部 LCC 前视点不是目的地')
        navigation['source_status'] = status
        safe_runtime = _only(runtime, ('vehicle','mode','source','last_result','wheel_driver','watchdog',
            'motion_gate','termination','diagnostics','run_archive','corner_continuation',
            'stationary_corner','sensor_recording'))
        # Runtime diagnostics have a fixed local producer; never pass arbitrary cloud/config/log envelopes.
        safe_runtime.pop('cloud',None)
        return finite(dict(schema_version=1, read_only=True, server_time=time.time(),
            system=self._system(), sensors=sensors, cloud=cloud, navigation=navigation,
            autonomy=dict(status=status, age_s=age, source_directory=str(folder),
                mode=runtime.get('mode'), run_id=Path(runtime['run_archive']).name if isinstance(runtime.get('run_archive'),str) else None),
            process=dict(state=status, running=status=='live', motors_enabled=runtime.get('mode')=='hardware',
                observed_only=True, message='只读监控：运行启停由现场车端程序负责', logs=[]),
            runtime=safe_runtime, runtime_status=dict(available=bool(runtime),age_seconds=age,current_run=status=='live'),
            frames=frames))


class DashboardServer:
    def __init__(self, state, host='0.0.0.0', port=8080, control_client=None):
        self.state, self.host, self.port = state, host, port
        self.control_client=control_client

    def make_http_server(self):
        state = self.state
        control_client=self.control_client
        assets = Path(__file__).parent
        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def do_GET(self):
                route = urlsplit(self.path).path
                if route in ('/api/drive/replay-data','/api/drive/frame.jpg','/api/drive/scan'):
                    try:
                        from .run_replay import RunReplayStore
                        from .drive_sessions import DEFAULT_DIRECTORY
                        if control_client is not None and control_client.get_state().get('running'):
                            raise ValueError('driving is active; replay after stopping')
                        query=parse_qs(urlsplit(self.path).query)
                        session=query.get('session_id',[''])[0]
                        store=RunReplayStore(DEFAULT_DIRECTORY,state.sensor_dir)
                        if route=='/api/drive/replay-data':
                            value=store.data(session)
                        else:
                            sample=int(query.get('sample',['-1'])[0])
                            if route=='/api/drive/frame.jpg':
                                self.respond(200,store.image(session,sample,query.get('view',['raw'])[0]),'image/jpeg');return
                            value=store.scan(session,sample)
                        self.respond(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json');return
                    except (OSError,ValueError,TypeError,KeyError,IndexError):
                        self.respond(422,b'{"error":"completed synchronized replay sample unavailable"}','application/json');return
                if route in ('/api/state','/api/health'):
                    controls=(control_client.get_state() if control_client is not None else
                              dict(available=False,state='unavailable',running=False))
                    directory=controls.get('directory')
                    state.preferred_runtime = None
                    if isinstance(directory,str):
                        from .drive_sessions import DEFAULT_DIRECTORY
                        folder=Path(directory).resolve()
                        if DEFAULT_DIRECTORY in folder.parents and folder.name.isalnum():
                            runtime=folder/'runtime'
                            if runtime not in state.runtime_dirs:state.runtime_dirs.append(runtime)
                            if controls.get('running'):state.preferred_runtime=runtime
                    value = state.get_state()
                    value['drive_control']=controls
                    value['read_only']=not controls.get('available',False)
                    from .run_replay import RunReplayStore
                    from .drive_sessions import DEFAULT_DIRECTORY
                    value['drive_runs']=RunReplayStore(DEFAULT_DIRECTORY,state.sensor_dir).sessions()
                    if route == '/api/health':
                        value = dict(service_alive=True, schema_version=1, read_only=value['read_only'],
                            all_sensors_online=all(v['status']=='online' for v in value['sensors'].values()),
                            sensors={k:v['status'] for k,v in value['sensors'].items()}, autonomy=value['autonomy']['status'])
                    self.respond(200,json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8')
                    return
                mapping = {'/':(assets/'dashboard.html','text/html; charset=utf-8'),
                    '/replay':(assets/'replay.html','text/html; charset=utf-8'),
                    '/replay.js':(assets/'replay.js','text/javascript; charset=utf-8'),
                    '/dashboard.js':(assets/'dashboard.js','text/javascript; charset=utf-8'),
                    '/dashboard.css':(assets/'dashboard.css','text/css; charset=utf-8'),
                    '/api/latest-frame.jpg':(state.runtime_directory()/'latest.jpg','image/jpeg'),
                    '/api/latest-birdeye.jpg':(state.runtime_directory()/'latest_birdeye.jpg','image/jpeg'),
                    '/api/depth.jpg':(state.sensor_dir/'depth.jpg','image/jpeg')}
                if route not in mapping:
                    self.respond(404,b'Not found','text/plain');return
                path, mime=mapping[route]
                try:
                    data=path.read_bytes()
                except OSError:
                    self.respond(404,b'No current image','text/plain');return
                self.respond(200,data,mime)

            def do_POST(self):
                route=urlsplit(self.path).path
                if route not in ('/api/drive/start','/api/drive/stop','/api/drive/heartbeat') or control_client is None:
                    self.respond(403,b'{"error":"drive control supervisor unavailable"}','application/json');return
                origin=self.headers.get('Origin')
                if (self.headers.get('X-Vehicle-Control')!='panel'
                        or origin is not None and urlsplit(origin).netloc!=self.headers.get('Host')):
                    self.respond(403,b'{"error":"explicit same-origin panel intent required"}','application/json');return
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=4096:raise ValueError('invalid request size')
                    body=json.loads(self.rfile.read(length))
                    if not isinstance(body,dict):raise ValueError('request must be an object')
                    action=route.rsplit('/',1)[1]
                    key='request_id' if action=='start' else 'session_id'
                    if set(body)!={key}:raise ValueError('only the session intent is accepted')
                    if action=='start' and not control_client.get_state().get('available'):
                        self.respond(503,b'{"error":"control supervisor unavailable"}','application/json');return
                    value=control_client.request(action,**body)
                    self.respond(200,json.dumps(value,ensure_ascii=False).encode(),'application/json');return
                except (ValueError,TypeError) as exc:
                    self.respond(409,json.dumps(dict(error=str(exc))).encode(),'application/json');return
                except OSError:
                    self.respond(503,b'{"error":"control supervisor unavailable"}','application/json');return

            def do_PUT(self):
                self.respond(403,b'{"error":"only explicit POST session intent is accepted"}','application/json')

            do_DELETE=do_PUT

            def respond(self, code, data, mime):
                self.send_response(code)
                self.send_header('Content-Type',mime)
                self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('X-Frame-Options','SAMEORIGIN')
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError,TimeoutError):
                    pass

            def log_message(self,*args):
                pass
        server=ThreadingHTTPServer((self.host,self.port),Handler)
        server.daemon_threads=True
        return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='0.0.0.0')
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--runtime-dir',action='append',required=True)
    parser.add_argument('--sensor-dir',required=True)
    parser.add_argument('--session-socket',help='Optional local session supervisor socket')
    args=parser.parse_args()
    state=DashboardState(args.runtime_dir[0],args.sensor_dir,args.runtime_dir[1:])
    from .drive_sessions import SessionClient
    server=DashboardServer(state,args.host,args.port,
        SessionClient(args.session_socket) if args.session_socket else None).make_http_server()
    def stop(*_):
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print('Read-only dashboard listening on port %s' % server.server_port,flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()

if __name__=='__main__': main()
