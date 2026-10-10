"""Recorded-video replay integration, with a synthetic model and no hardware."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.tools.replay_outer_loop import replay
from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
from car.test.test_yolopv2_primary import MaskModel


class OuterLoopReplayTests(unittest.TestCase):
    def test_lossless_replay_preserves_cached_sequence_and_stale_stop_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root/'run'; run.mkdir()
            inputs = run/'semantic_inputs'; inputs.mkdir()
            model = MaskModel()
            np.savez_compressed(inputs/'00000007.npz', sequence=7, captured_at=2.,
                                road=model.road, lane=model.lane,
                                frame=np.zeros((480,640,3),np.uint8), detections_json='[]')
            with (run/'onboard_log.csv').open('w') as stream:
                out = csv.DictWriter(stream,fieldnames=['timestamp_s','sample','action',
                    'semantic_sequence','semantic_result_age_s','semantic_fusion_source'])
                out.writeheader()
                for i in range(8): out.writerow({'timestamp_s':i*.05,'sample':i,'action':'stop',
                    'semantic_sequence':7,'semantic_result_age_s':.05+i*.05,
                    'semantic_fusion_source':'yolopv2'})
            with self.assertRaisesRegex(ValueError,'incomplete'):
                replay(run,root/'without-final-status',mode='recorded')
            (run/'status.json').write_text(json.dumps({'termination':{'stopped':True,'completed_at':100.},
                'last_result':{'sample':7},'diagnostics':{'archive':{'written_frames':8,'dropped_frames':0,'error':None}}}))
            with patch('longtail.detectors.yolopv2_detector.YOLOPv2Detector') as gpu, \
                 patch('autodrive.runtime.onboard.create_chassis') as chassis:
                result = replay(run,root/'derived',mode='recorded')
                gpu.assert_not_called(); chassis.assert_not_called()
            self.assertEqual(result['actions'],{'stop':8})
            commands = json.loads((root/'derived/commands.json').read_text())
            self.assertIn('stale',commands[-1]['reason'])
            self.assertEqual(result['unique_semantic_inputs'],1)
            # Old CSV rounded 0.30004 to .3; its explicit stale veto must win.
            csv_path = run/'onboard_log.csv'
            with csv_path.open() as stream: rows = list(csv.DictReader(stream))
            rows[-1]['semantic_result_age_s'] = '.3'
            rows[-1]['semantic_fusion_source'] = 'YOLOPv2 stale result'
            def save():
                with csv_path.open('w') as stream:
                    out = csv.DictWriter(stream,fieldnames=list(rows[0])); out.writeheader();out.writerows(rows)
            save()
            replay(run,root/'rounded',mode='recorded')
            self.assertIn('stale',json.loads((root/'rounded/commands.json').read_text())[-1]['reason'])
            rows[4]['sample'] = '99'; save()
            with self.assertRaisesRegex(ValueError,'incomplete'):
                replay(run,root/'gap',mode='recorded')

    def test_replay_preserves_originals_emits_virtual_commands_and_checks_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/'run';run.mkdir()
            writer = cv2.VideoWriter(str(run/'raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20.,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in range(10): writer.write(np.zeros((480,640,3),np.uint8))
            writer.release()
            with (run/'onboard_log.csv').open('w') as stream:
                csv_writer = csv.DictWriter(stream,fieldnames=['timestamp_s','sample','action'])
                csv_writer.writeheader()
                for i in range(10): csv_writer.writerow({'timestamp_s':i*.05,'sample':i,'action':'forward'})
            originals = {p.name:p.read_bytes() for p in run.iterdir()}
            def detector(config,**kwargs):
                return YOLOPv2SemanticDetector(config,model=MaskModel(),**kwargs)
            with patch('autodrive.perception.yolopv2_semantic.YOLOPv2SemanticDetector',side_effect=detector), \
                 patch('autodrive.runtime.onboard.create_chassis') as chassis:
                summary = replay(run,root/'derived',expectations=[{'start_frame':6,'end_frame':9,
                    'actions':['forward'],'source':'synthetic straight-road test'}])
                chassis.assert_not_called()
            self.assertEqual(summary['frames'],10)
            self.assertEqual(summary['maximum_pwm'],30)
            self.assertTrue(summary['expected_behavior_verified'])
            for name,data in originals.items(): self.assertEqual((run/name).read_bytes(),data)
            commands=json.loads((root/'derived/commands.json').read_text())
            self.assertEqual(commands[-1]['pwm'],[30,30,30,30])


if __name__=='__main__':unittest.main()
