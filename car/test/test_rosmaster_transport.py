import os
import pty
import struct
import sys
import time
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))
from vehicle_control.platforms.rosmaster_transport import RosmasterSerialSession, RosmasterCommunicationError


class ByteSerial:
    def __init__(self, **options):
        self.options=options; self.rx=bytearray(); self.tx=[]; self.is_open=True
        self.short=False; self.write_error=False; self.read_error=False
    @property
    def in_waiting(self): return len(self.rx)
    def read(self, count):
        if self.read_error: raise OSError("disconnected")
        data=bytes(self.rx[:count]); del self.rx[:count]; return data
    def write(self, data):
        if self.write_error: raise OSError("write failed")
        self.tx.append(bytes(data)); return len(data)-1 if self.short else len(data)
    def close(self): self.is_open=False


def incoming(kind,payload):
    # Independent fixture derived from local vendor RX length/checksum format.
    body=bytes([len(payload)+3,kind])+payload
    return b"\xff\xfb"+body+bytes([sum(body)&255])


class TransportTests(unittest.TestCase):
    def session(self, **kwargs):
        fake=ByteSerial()
        s=RosmasterSerialSession(serial_factory=lambda **options: fake,delay=0,**kwargs)
        self.addCleanup(s.close); return s,fake

    def test_open_and_close_never_emit_actuator_or_configuration_frames(self):
        s,f=self.session(); self.assertEqual(f.tx,[]); s.close(); self.assertEqual(f.tx,[])

    def test_stop_packet_matches_vendor_wire_protocol(self):
        s,f=self.session(); s.set_motor(0,0,0,0)
        self.assertEqual(f.tx,[bytes.fromhex("fffc07100000000017")])

    def test_signed_motor_and_servo_payloads(self):
        s,f=self.session();s.set_motor(-1,2,-3,4);s.set_pwm_servo(2,90)
        self.assertEqual(f.tx,[bytes.fromhex("fffc0710ff02fd0419"),bytes.fromhex("fffc0503025a64")])

    def test_short_write_is_a_visible_fault_and_latches_nonzero_output_off(self):
        s,f=self.session();f.short=True
        with self.assertRaises(RosmasterCommunicationError): s.set_motor(10,0,0,0)
        f.short=False
        with self.assertRaises(RosmasterCommunicationError):s.set_motor(5,0,0,0)
        s.set_motor(0,0,0,0)
        self.assertFalse(s.get_status()["healthy"])

    def test_write_exception_is_propagated(self):
        s,f=self.session();f.write_error=True
        with self.assertRaises(RosmasterCommunicationError):s.set_pwm_servo(1,20)
        self.assertFalse(s.get_status()["healthy"])

    def test_invalid_input_never_writes(self):
        s,f=self.session()
        for values in [(101,0,0,0),(127,0,0,0),(1.5,0,0,0),(True,0,0,0)]:
            with self.subTest(values=values),self.assertRaises(ValueError):s.set_motor(*values)
        for channel,angle in [(5,0),(1,-1),(1,181),(1,90.5)]:
            with self.subTest(channel=channel,angle=angle),self.assertRaises(ValueError):s.set_pwm_servo(channel,angle)
        self.assertEqual(f.tx,[])

    def test_initial_values_are_invalid_null_then_valid_zero_after_frame(self):
        s,f=self.session()
        self.assertFalse(s.telemetry()["encoder"]["valid"]);self.assertIsNone(s.telemetry()["encoder"]["value"])
        f.rx.extend(bytes.fromhex("fffb130d0000000000000000000000000000000020"));s.poll()
        self.assertTrue(s.telemetry()["encoder"]["valid"]);self.assertEqual(s.telemetry()["encoder"]["value"]["native_ticks"],[0,0,0,0])

    def test_partial_noisy_bad_and_unknown_frames_resynchronize(self):
        s,f=self.session(); packet=incoming(0x0d,struct.pack("<4i",1,-2,3,-4))
        broken=packet[:-1]+bytes([packet[-1]^1])
        f.rx.extend(b"noise"+broken+incoming(0x7f,b"x")+packet[:8]);s.poll()
        self.assertFalse(s.telemetry()["encoder"]["valid"])
        f.rx.extend(packet[8:]);s.poll()
        self.assertEqual(s.telemetry()["encoder"]["value"]["native_ticks"],[1,-2,3,-4])
        self.assertGreater(s.get_status()["checksum_errors"],0)

    def test_wrong_payload_length_cannot_replace_valid_encoder(self):
        s,f=self.session();f.rx.extend(incoming(0x0d,struct.pack("<4i",1,2,3,4))+incoming(0x0d,b"x"));s.poll()
        self.assertEqual(s.telemetry()["encoder"]["value"]["native_ticks"],[1,2,3,4])
        self.assertEqual(s.get_status()["length_errors"],1)

    def test_telemetry_age_is_receipt_time_and_stale_does_not_refresh(self):
        s,f=self.session();f.rx.extend(incoming(0x0d,b"\0"*16));s.poll()
        first=s.telemetry(max_age=1)["encoder"];time.sleep(.015)
        old=s.telemetry(max_age=.001)["encoder"]
        self.assertFalse(old["valid"]);self.assertTrue(old["stale"])
        self.assertEqual(first["received_monotonic"],old["received_monotonic"])
        self.assertGreater(old["age_s"],first["age_s"])

    def test_speed_battery_and_attitude_scaling(self):
        s,f=self.session();f.rx.extend(incoming(0x0a,struct.pack("<hhhB",100,-200,300,121))+incoming(0x0c,struct.pack("<3h",1000,-2000,3000)));s.poll()
        self.assertEqual(s.telemetry()["motion"]["value"],{"velocity":[.1,-.2,.3],"battery_v":12.1})
        self.assertEqual(s.telemetry()["attitude"]["value"]["radians"],[.1,-.2,.3])

    def test_imu_frame_type_determines_scaling(self):
        s,f=self.session();f.rx.extend(incoming(0x0e,struct.pack("<9h",1000,-2000,3000,0,0,9800,1,2,3)));s.poll()
        v=s.telemetry()["imu"]["value"]
        self.assertEqual(v["protocol"],"ICM");self.assertEqual(v["gyro_rad_s"],[1,-2,3]);self.assertEqual(v["accel_m_s2"],[0,0,9.8])
        f.rx.extend(incoming(0x0b,struct.pack("<9h",3755,3755,3755,1672,0,0,1,2,3)));s.poll()
        v=s.telemetry()["imu"]["value"];self.assertEqual(v["protocol"],"MPU");self.assertLess(v["gyro_rad_s"][1],0)

    def test_read_disconnect_invalidates_cached_feedback(self):
        s,f=self.session();f.rx.extend(incoming(0x0d,b"\0"*16));s.poll();f.read_error=True
        with self.assertRaises(RosmasterCommunicationError):s.poll()
        self.assertFalse(s.telemetry()["encoder"]["valid"])

    def test_invalid_timeout_does_not_open_device(self):
        for options in [{"timeout":0},{"timeout":float("inf")},{"write_timeout":None},{"delay":-1}]:
            with self.subTest(options=options),self.assertRaises(ValueError):
                RosmasterSerialSession(serial_factory=lambda **kw:self.fail("must not open"),**options)

    def test_waiting_for_writer_lock_does_not_make_expired_sample_valid(self):
        s,f=self.session();f.rx.extend(incoming(0x0d,b"\0"*16));s.poll()
        started=threading.Event();release=threading.Event()
        def blocking_write(data):
            started.set();release.wait(timeout=1);return len(data)
        f.write=blocking_write
        writer=threading.Thread(target=s.set_motor,args=(0,0,0,0));writer.start()
        self.assertTrue(started.wait(timeout=1))
        timer=threading.Timer(.05,release.set);timer.start()
        try:sample=s.telemetry(max_age=.01)["encoder"]
        finally:release.set();writer.join();timer.cancel()
        self.assertFalse(sample["valid"]);self.assertTrue(sample["stale"])
        self.assertGreater(sample["age_s"],.04)

    def test_current_firmware_identity_frame_includes_reserved_byte(self):
        s,f=self.session();f.rx.extend(bytes.fromhex("fffb051501001b"));s.poll()
        self.assertTrue(s.telemetry()["car_type"]["valid"])
        self.assertEqual(s.telemetry()["car_type"]["value"],{"car_type":1,"reserved":0})

    def test_real_pseudoterminal_has_exclusive_owner_and_bounded_poll(self):
        master,slave=pty.openpty();path=os.ttyname(slave)
        self.addCleanup(os.close,master);self.addCleanup(os.close,slave)
        with RosmasterSerialSession(path,timeout=.02,write_timeout=.02,delay=0) as s:
            with self.assertRaises(RosmasterCommunicationError):RosmasterSerialSession(path,timeout=.02,write_timeout=.02)
            start=time.monotonic();s.poll();self.assertLess(time.monotonic()-start,.15)
            os.write(master,incoming(0x0d,struct.pack("<4i",9,8,7,6)))
            for _ in range(10):
                s.poll()
                if s.telemetry()["encoder"]["valid"]: break
            self.assertEqual(s.telemetry()["encoder"]["value"]["native_ticks"],[9,8,7,6])
        with RosmasterSerialSession(path,timeout=.02,write_timeout=.02,delay=0):pass

if __name__ == "__main__":unittest.main()
