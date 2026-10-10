import math
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.web.sensor_publishers import serial_readings, lidar_reading, depth_reading

class SensorPublisherTest(unittest.TestCase):
    def test_passive_serial_adapter_never_sends_commands(self):
        class Session:
            def telemetry(self):
                return {'motion':{'valid':True,'age_s':.02,'value':{'battery_v':12.1}},
                        'imu':{'valid':False,'stale':True,'age_s':10,'value':{'gyro_rad_s':[0,0,0]}}}
        values=serial_readings(Session())
        self.assertEqual(values['battery']['value']['voltage_v'],12.1)
        self.assertEqual(values['imu']['status'],'stale')
        self.assertEqual(values['encoder']['status'],'waiting')

    def test_lidar_filters_invalid_ranges_and_bounds_public_points(self):
        value=lidar_reading([float('inf'),float('nan'),0,-1,2,3],.1,12,0,.1,10)
        self.assertEqual(value['valid_points'],2)
        self.assertEqual(value['nearest_m'],2)
        self.assertLessEqual(len(lidar_reading([2.]*4000,.1,12,0,.001,10)['xy_m']),360)

    def test_depth_zeros_are_not_measurements(self):
        value=depth_reading(np.array([[0,1000],[2000,0]],dtype=np.uint16),30)
        self.assertEqual(value['valid_fraction'],.5)
        self.assertEqual(value['nearest_m'],1)
        self.assertEqual(value['median_m'],1.5)
        self.assertIsNone(depth_reading(np.zeros((2,2),dtype=np.uint16),30)['nearest_m'])

if __name__=='__main__':unittest.main()

class SharedSerialTest(unittest.TestCase):
    def test_nonblocking_poll_does_not_wait_for_no_data(self):
        from car.test.test_rosmaster_transport import ByteSerial
        from vehicle_control.platforms.rosmaster_transport import RosmasterSerialSession
        fake=ByteSerial()
        def forbidden_read(count):raise AssertionError('empty UART must not be read')
        fake.read=forbidden_read
        session=RosmasterSerialSession(serial_factory=lambda **kw:fake)
        self.addCleanup(session.close)
        self.assertEqual(session.poll_available(),b'')
        self.assertEqual(fake.tx,[])

    def test_nonblocking_poll_skips_control_lock(self):
        import threading
        from car.test.test_rosmaster_transport import ByteSerial
        from vehicle_control.platforms.rosmaster_transport import RosmasterSerialSession
        session=RosmasterSerialSession(serial_factory=lambda **kw:ByteSerial())
        self.addCleanup(session.close)
        held=threading.Event();release=threading.Event()
        def holder():
            with session._lock:held.set();release.wait(2)
        thread=threading.Thread(target=holder);thread.start();held.wait(1)
        try:self.assertEqual(session.poll_available(),b'')
        finally:release.set();thread.join(2)
