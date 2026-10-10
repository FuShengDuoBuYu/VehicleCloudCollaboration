"""Optional sensor publishers: no actuator commands; borrow a session when driving.

Standalone serial mode owns the UART exclusively; do not keep it running alongside
an independently launched controller. Dashboard HTTP itself never opens the UART.
"""
import argparse
from collections import deque
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from .dashboard import publish_snapshot


def serial_readings(session):
    samples = session.telemetry()
    readings = {}
    for target, group in (('battery','motion'), ('imu','imu'), ('attitude','attitude'), ('encoder','encoder')):
        sample = samples.get(group, {})
        value = sample.get('value')
        status = 'online' if sample.get('valid') else 'stale' if value else 'waiting'
        if target == 'battery' and value is not None:
            value = {'voltage_v':value.get('battery_v'), 'percentage':None}
        readings[target] = dict(value=value, status=status, age_s=sample.get('age_s'), reason='底盘被动遥测')
    return readings


def lidar_reading(ranges, lower, upper, angle_min, angle_increment, fps):
    valid = [(i,float(v)) for i,v in enumerate(ranges) if math.isfinite(v) and lower <= v <= upper and v > 0]
    stride = max(1, math.ceil(len(valid)/360))
    return dict(fps=fps, points=len(ranges), valid_points=len(valid),
        nearest_m=min((v for _,v in valid), default=None),
        xy_m=[[round(v*math.cos(angle_min+i*angle_increment),3), round(v*math.sin(angle_min+i*angle_increment),3)]
              for i,v in valid[::stride]])


def depth_reading(frame, fps):
    import numpy as np
    valid=frame[np.isfinite(frame) & (frame > 0)]
    return dict(width=int(frame.shape[1]),height=int(frame.shape[0]),fps=fps,
        valid_fraction=float(valid.size/frame.size) if frame.size else 0,
        nearest_m=float(valid.min())/1000 if valid.size else None,
        median_m=float(np.median(valid))/1000 if valid.size else None)


def _rate(times):
    return (len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None


class SharedSerialPublisher:
    """Borrow an existing controller session; never close it or send commands."""
    def __init__(self, session, directory):
        import threading
        self.session=session
        self.path=Path(directory)/'serial.json'
        self._stop=threading.Event()
        self._error=None
        self._thread=threading.Thread(target=self._run,name='dashboard-serial',daemon=True)
        self._thread.start()

    def _run(self):
        next_update=0
        try:
            while not self._stop.wait(.02):
                self.session.poll_available()
                if time.monotonic()>=next_update:
                    publish_snapshot(self.path,serial_readings(self.session))
                    next_update=time.monotonic()+.25
        except Exception as exc:
            self._error=type(exc).__name__
            # Observability failure cannot issue a command or escape into the
            # control loop. The UART transport separately latches I/O faults.
            try:
                publish_snapshot(self.path,{name:dict(status='error',value=None,age_s=None,
                    reason='共享遥测发布失败') for name in ('battery','imu','attitude','encoder')})
            except OSError:
                pass

    def get_state(self):
        return dict(running=self._thread.is_alive() and not self._stop.is_set(),error=self._error)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)


def collect_serial(directory, device, duration):
    control_dir=Path(__file__).resolve().parents[2]/'control'
    sys.path.insert(0,str(control_dir))
    from vehicle_control.platforms.rosmaster_transport import RosmasterSerialSession
    running=[True]
    def stop(*_):running[0]=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    path=Path(directory)/'serial.json'
    try:
        session=RosmasterSerialSession(serial_port=device,timeout=.05)
    except OSError:
        publish_snapshot(path,{n:dict(status='busy',value=None,age_s=None,reason='串口不可用或由控制器占用') for n in ('battery','imu','attitude','encoder')})
        return 2
    start=time.monotonic();next_update=start
    try:
        while running[0] and (duration<=0 or time.monotonic()-start<duration):
            session.poll()
            if time.monotonic()>=next_update:
                publish_snapshot(path,serial_readings(session))
                next_update=time.monotonic()+.25
    finally:
        readings=serial_readings(session)
        status=session.get_status()
        session.close()
        for item in readings.values():item.update(status='stopped',reason='被动采集已结束，保留最后采样')
        publish_snapshot(path,readings)
        print('Serial monitor closed; TX bytes = %s' % status['tx_bytes'],flush=True)
    return 0


def collect_ros(directory, start_drivers, duration):
    import cv2
    import numpy as np
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, LaserScan
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'ros.json';running=[True];processes=[]
    data={n:dict(status='waiting',value=None,age_s=None,reason='等待传感器消息') for n in ('lidar','depth')}
    received={};history={n:deque(maxlen=60) for n in data};last_image=[0]
    def stop(*_):running[0]=False
    rclpy.init();node=rclpy.create_node('vehicle_dashboard_sensors')
    from autodrive.runtime.sensor_recording import RosRunRecording
    from .dashboard import boot_id
    recording=RosRunRecording(directory,boot_id())
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    qos=QoSProfile(depth=1,history=HistoryPolicy.KEEP_LAST,reliability=ReliabilityPolicy.BEST_EFFORT)
    def lidar(msg):
        now=time.monotonic();history['lidar'].append(now)
        recording.submit('lidar',dict(received_monotonic=now,
            source_stamp_s=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,
            frame_id=msg.header.frame_id,angle_min=msg.angle_min,angle_max=msg.angle_max,
            angle_increment=msg.angle_increment,time_increment=msg.time_increment,
            scan_time=msg.scan_time,range_min=msg.range_min,range_max=msg.range_max),
            np.array([msg.ranges,msg.intensities if len(msg.intensities)==len(msg.ranges)
                      else [np.nan]*len(msg.ranges)],dtype=np.float32))
        value=lidar_reading(msg.ranges,msg.range_min,msg.range_max,msg.angle_min,msg.angle_increment,_rate(history['lidar']))
        received['lidar']=now
        data['lidar']=dict(status='online' if value['valid_points'] else 'error',value=value,age_s=0,reason='LaserScan 实测')
    def depth(msg):
        now=time.monotonic()
        if msg.encoding != '16UC1' or len(msg.data)!=msg.step*msg.height or msg.step%2 or msg.step<msg.width*2:
            data['depth']=dict(status='error',value=None,age_s=None,reason='深度帧格式异常');return
        frame=np.frombuffer(bytes(msg.data),dtype='>u2' if msg.is_bigendian else '<u2').reshape(msg.height,msg.step//2)[:,:msg.width]
        recording.submit('depth',dict(received_monotonic=now,
            source_stamp_s=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,
            frame_id=msg.header.frame_id,encoding=msg.encoding,is_bigendian=msg.is_bigendian,
            width=msg.width,height=msg.height,step=msg.step),
            np.frombuffer(bytes(msg.data),dtype=np.uint8))
        history['depth'].append(now);received['depth']=now
        value=depth_reading(frame,_rate(history['depth']))
        data['depth']=dict(status='online' if value['valid_fraction']>0 else 'error',value=value,age_s=0,reason='16UC1 有效深度')
        if now-last_image[0]>.5:
            display=cv2.applyColorMap(np.uint8(np.clip(frame/4000*255,0,255)),cv2.COLORMAP_TURBO)
            display[frame==0]=0
            ok,jpg=cv2.imencode('.jpg',display,[cv2.IMWRITE_JPEG_QUALITY,70])
            if ok:
                temporary=directory/'depth.tmp';temporary.write_bytes(jpg.tobytes());temporary.replace(directory/'depth.jpg')
            last_image[0]=now
    node.create_subscription(LaserScan,'/scan',lidar,qos)
    node.create_subscription(Image,'/camera/depth/image_raw',depth,qos)
    publish_snapshot(path,data)
    try:
        if start_drivers:
            commands=[['ros2','run','sllidar_ros2','sllidar_node','--ros-args','-p','serial_port:=/dev/rplidar','-p','serial_baudrate:=1000000','-p','scan_mode:=DenseBoost','-p','angle_compensate:=true'],
                ['ros2','run','astra_camera','astra_camera_node','--ros-args','-r','__ns:=/camera','-p','camera_name:=camera','-p','enable_color:=false','-p','enable_ir:=false','-p','enable_depth:=true','-p','use_uvc_camera:=false','-p','enable_point_cloud:=false','-p','enable_colored_point_cloud:=false','-p','publish_tf:=false','-p','depth_width:=640','-p','depth_height:=480','-p','depth_fps:=30']]
            for command in commands:
                processes.append(subprocess.Popen(command,start_new_session=True))
        start=time.monotonic();next_update=start
        while running[0] and rclpy.ok() and (duration<=0 or time.monotonic()-start<duration):
            recording.poll()
            rclpy.spin_once(node,timeout_sec=.05)
            now=time.monotonic()
            if now>=next_update:
                for name,item in data.items():
                    if name in received:
                        item['age_s']=now-received[name]
                        if item['age_s']>2:item['status']='stale'
                for name,proc in zip(('lidar','depth'),processes):
                    if proc.poll() is not None:
                        data[name].update(status='error',reason='传感器驱动退出')
                        raise RuntimeError('sensor driver exited: '+name)
                publish_snapshot(path,data);next_update=now+.5
    finally:
        recording.close()
        import os
        for proc in processes:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGINT)
                except ProcessLookupError:pass
        for proc in processes:
            try:proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
        for item in data.values():item.update(status='stopped',reason='传感器发布器已结束')
        publish_snapshot(path,data)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('serial','ros'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--serial-port',default='/dev/myserial')
    parser.add_argument('--duration',type=float,default=0)
    parser.add_argument('--start-drivers',action='store_true',help='Explicitly start lidar scan and depth nodes')
    args=parser.parse_args()
    if not math.isfinite(args.duration) or args.duration<0:parser.error('duration must be finite and nonnegative')
    return collect_serial(args.output_dir,args.serial_port,args.duration) if args.mode=='serial' else collect_ros(args.output_dir,args.start_drivers,args.duration)

if __name__=='__main__':raise SystemExit(main())
