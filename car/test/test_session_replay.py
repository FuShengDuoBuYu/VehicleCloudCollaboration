"""Offline session data validation, synthetic journals, no hardware."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.tools import replay_outer_loop as replay
try:
    bundle=importlib.import_module('autodrive.tools.replay_session')
except ModuleNotFoundError:
    bundle=None


class SessionReplayTests(unittest.TestCase):
    def test_full_bundle_replays_with_no_hardware_and_preserves_originals(self):
        self.assertIsNotNone(bundle)
        from autodrive.runtime.sensor_recording import RunSensorWriter
        from car.test.test_yolopv2_primary import MaskModel
        import csv
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir()
            inputs=run/'semantic_inputs';inputs.mkdir()
            model=MaskModel()
            np.savez(inputs/'00000007.npz',sequence=7,captured_at=2.,road=model.road,
                lane=model.lane,frame=np.zeros((480,640,3),np.uint8),detections_json='[]')
            rows=[dict(sample=i,timestamp_s=i*.05,control_monotonic=2.+i*.05,
                action='stop',semantic_sequence=7,semantic_result_age_s=.05+i*.05,
                semantic_fusion_source='yolopv2') for i in range(8)]
            with (run/'onboard_log.csv').open('w') as stream:
                out=csv.DictWriter(stream,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
            (run/'status.json').write_text(json.dumps(dict(termination=dict(stopped=True,completed_at=100.),
                last_result=dict(sample=7),diagnostics=dict(archive=dict(written_frames=8,dropped_frames=0,error=None)))))
            _,config=replay.load_config(replay.DEFAULT_CONFIG,'rosmaster_jetson_yolopv2_outer_trial')
            config.setdefault('runtime',{})['telemetry_dir']=str(root/'ros')
            config['runtime']['route_hint']='center'
            (run/'resolved_runtime_config.yaml').write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError,'outside'):
                bundle.replay_session(run,root/'ros/recordings/run/derived')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                bundle.replay_session(run,root/'rejected')
            self.assertFalse((root/'rejected').exists())
            local=RunSensorWriter(run/'sensors')
            for sensor in ('uart_rx','telemetry'):
                local.submit(sensor,{'received_monotonic':1.95,'data_hex':'ff'})
            local.close()
            ros=RunSensorWriter(root/'ros/recordings/run')
            for sensor in ('lidar','depth'):
                ros.submit(sensor,{'received_monotonic':1.96},np.array([0,1],np.uint8))
            ros.close()
            before=bundle.inventory(run)
            with patch('longtail.detectors.yolopv2_detector.YOLOPv2Detector') as gpu, \
                 patch('autodrive.runtime.onboard.create_chassis') as chassis, \
                 patch.object(replay,'analyze',wraps=replay.analyze) as analyzed:
                result=bundle.replay_session(run,root/'derived')
                gpu.assert_not_called();chassis.assert_not_called()
                self.assertTrue(all(call.args[5]=='center' for call in analyzed.call_args_list))
            self.assertTrue(result['all_raw_streams_complete'])
            self.assertEqual(result['controller_summary']['frames'],8)
            self.assertEqual(bundle.inventory(run),before)
            timeline=[json.loads(line) for line in (root/'derived/sensor_timeline.jsonl').read_text().splitlines()]
            self.assertEqual(set(timeline[0]['sensors']),{'uart_rx','telemetry','lidar','depth'})

    def test_frozen_configuration_never_merges_current_defaults(self):
        self.assertTrue(hasattr(replay,'recorded_replay_config'))
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            config={'perception':{'mode':'yolopv2'},'camera':{},'wheels':{},
                    'safety':{},'perspective':{},'frozen_marker':123}
            with patch.object(replay,'load_config') as today:
                with self.assertRaisesRegex(ValueError,'resolved'):
                    replay.recorded_replay_config(run,'recorded')
                (run/'resolved_runtime_config.yaml').write_text(yaml.safe_dump(config))
                loaded,evidence=replay.recorded_replay_config(run,'recorded')
                self.assertEqual(loaded,config)
                self.assertEqual(evidence['selection'],'recorded')
                self.assertEqual(len(evidence['sha256']),64)
                today.assert_not_called()

    def test_sensor_index_never_uses_future_samples_and_preserves_raw_arrays(self):
        self.assertIsNotNone(bundle)
        from autodrive.runtime.sensor_recording import RunSensorWriter
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'sensors';writer=RunSensorWriter(root)
            raw=np.array([1.,np.nan,np.inf],np.float32)
            writer.submit('lidar',{'received_monotonic':10.,'source_stamp_s':7.},raw)
            writer.submit('lidar',{'received_monotonic':11.},raw)
            writer.close()
            originals={p.name:p.read_bytes() for p in root.iterdir()}
            events,health=bundle.read_sensor_journal(root)
            self.assertTrue(health['complete'])
            matched=bundle.align_sensor_events([{'sample':'0','control_monotonic':'10.5'},
                {'sample':'1','control_monotonic':'9.9'}],events)
            self.assertEqual(matched[0]['sensors']['lidar']['index'],0)
            self.assertEqual(matched[0]['sensors']['lidar']['age_s'],.5)
            self.assertNotIn('lidar',matched[1]['sensors'])
            for name,data in originals.items():self.assertEqual((root/name).read_bytes(),data)

    def test_incomplete_journal_and_escaping_array_are_rejected(self):
        self.assertIsNotNone(bundle)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'events.jsonl').write_text(json.dumps(dict(
                index=0,sensor='lidar',received_monotonic=1.,array_path='../escape.npy'))+'\n')
            (root/'status.json').write_text(json.dumps(dict(closed=True,written={'lidar':1},
                dropped=0,error=None)))
            with self.assertRaisesRegex(ValueError,'escaped'):
                bundle.read_sensor_journal(root)
            (root/'events.jsonl').write_text(json.dumps(dict(index=1,sensor='lidar',received_monotonic=1.))+'\n')
            with self.assertRaisesRegex(ValueError,'sequence'):
                bundle.read_sensor_journal(root)


if __name__=='__main__':unittest.main()
