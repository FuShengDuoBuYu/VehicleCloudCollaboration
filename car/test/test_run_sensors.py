"""Archive all sensor types using synthetic samples, with no device access."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    records=importlib.import_module('autodrive.runtime.sensor_recording')
except ModuleNotFoundError as exc:
    if exc.name!='autodrive.runtime.sensor_recording': raise
    records=None


class SensorRecordingTests(unittest.TestCase):
    def test_live_uart_freshness_and_publisher_health_are_required(self):
        self.assertTrue(hasattr(records,'raw_sensor_recording_ready'))
        local=dict(closed=False,written={'uart_rx':1},last_sample_monotonic={'uart_rx':8.},dropped=0,error=None)
        ros=dict(closed=False,session_id='r1',active=True,published_monotonic=10.,
            written={'depth':1,'lidar':1},last_sample_monotonic={'depth':10.,'lidar':10.},dropped=0,error=None)
        publisher=dict(running=True,error=None)
        self.assertFalse(records.raw_sensor_recording_ready(local,ros,'r1',10.,publisher))
        local['last_sample_monotonic']['uart_rx']=9.99
        self.assertTrue(records.raw_sensor_recording_ready(local,ros,'r1',10.,publisher))
        for publisher in (dict(running=False,error=None),dict(running=True,error='OSError')):
            self.assertFalse(records.raw_sensor_recording_ready(local,ros,'r1',10.,publisher))

    def test_shared_publisher_file_error_is_visible_to_control_health(self):
        from autodrive.web.sensor_publishers import SharedSerialPublisher
        from unittest.mock import Mock
        import time
        self.assertTrue(hasattr(SharedSerialPublisher,'get_state'))
        with tempfile.TemporaryDirectory() as temp, \
             patch('autodrive.web.sensor_publishers.publish_snapshot',side_effect=OSError('disk unavailable')):
            publisher=SharedSerialPublisher(Mock(),temp)
            try:
                deadline=time.monotonic()+1.
                while publisher.get_state()['running'] and time.monotonic()<deadline:time.sleep(.01)
                self.assertFalse(publisher.get_state()['running'])
                self.assertIn('OSError',publisher.get_state()['error'])
            finally:publisher.close()

    def test_ros_stop_publishes_inactive_closed_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);recorder=records.RosRunRecording(root,'test-boot')
            with patch.object(records.time,'monotonic',return_value=10.):
                records.atomic_json(root/'recording_request.json',dict(active=True,
                    boot_id='test-boot',session_id='r1',expires_monotonic=12.))
                recorder.poll()
                self.assertTrue(json.loads((root/'recording_state.json').read_text())['active'])
                records.atomic_json(root/'recording_request.json',dict(active=False))
                recorder.poll()
                state=json.loads((root/'recording_state.json').read_text())
                self.assertFalse(state['active'])
                self.assertTrue(state['closed'])
                self.assertEqual(state['session_id'],'r1')

    def test_raw_depth_and_lidar_survive_archive_and_metadata_keeps_timestamps(self):
        self.assertIsNotNone(records, 'run sensor recorder missing')
        with tempfile.TemporaryDirectory() as temp:
            writer=records.RunSensorWriter(Path(temp)/'r1')
            depth=np.array([[0,1000],[2345,65535]],np.uint16)
            ranges=np.array([1.,np.nan,np.inf,2.5],np.float32)
            writer.submit('depth',{'received_monotonic':12.,'source_stamp_s':8.},depth)
            writer.submit('lidar',{'received_monotonic':12.1,'angle_min':-.5},ranges)
            writer.submit('uart_rx',{'received_monotonic':12.2,'data_hex':'fffb0100'})
            writer.close()
            rows=[json.loads(line) for line in (Path(temp)/'r1/events.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),3)
            np.testing.assert_array_equal(np.load(Path(temp)/'r1'/rows[0]['array_path']),depth)
            np.testing.assert_array_equal(np.load(Path(temp)/'r1'/rows[1]['array_path']),ranges)
            self.assertEqual(rows[0]['received_monotonic'],12.)
            self.assertEqual(rows[0]['source_stamp_s'],8.)
            self.assertEqual(writer.get_state()['written']['lidar'],1)

    def test_closed_writer_rejects_samples_and_original_directory_is_not_overwritten(self):
        self.assertIsNotNone(records, 'run sensor recorder missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'r1';writer=records.RunSensorWriter(root);writer.close()
            self.assertFalse(writer.submit('depth',{}))
            with self.assertRaises(FileExistsError): records.RunSensorWriter(root)


if __name__=='__main__':unittest.main()
