"""Exercise the real runtime lifecycle with synthetic video/model and no hardware."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.runtime import onboard
from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
from car.test.test_yolopv2_primary import MaskModel


class OnboardCloudTests(unittest.TestCase):
    def test_incompatible_cloud_contract_fails_before_components_or_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'input.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10.,(640,480))
            self.assertTrue(writer.isOpened());writer.write(np.zeros((480,640,3),np.uint8));writer.release()
            config_path,config=onboard.load_config(onboard.DEFAULT_CONFIG,'rosmaster_jetson_yolopv2')
            config['cloud_arbitration']['enabled']=True
            client=SimpleNamespace(config=SimpleNamespace(contract='road-observation-fast-v1'))
            arguments=['onboard','--video',str(source),'--output-dir',str(root/'output')]
            with patch.object(sys,'argv',arguments), patch.object(onboard,'load_config',return_value=(config_path,config)), \
                 patch.object(onboard,'build_components') as components, \
                 patch('cloud_client.client.CloudClient',return_value=client), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'road-scene-v1'):
                    onboard.main()
                components.assert_not_called()

    def test_disabled_cloud_without_archive_runs_and_closes_without_devices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (640,480))
            self.assertTrue(writer.isOpened())
            for _ in range(6):
                writer.write(np.zeros((480,640,3),np.uint8))
            writer.release()
            arguments=['onboard','--video',str(source),'--vehicle-profile','rosmaster_jetson_yolopv2',
                       '--max-samples','3','--no-run-archive','--output-dir',str(root/'output')]
            def detector(config,**kwargs):
                return YOLOPv2SemanticDetector(config,model=MaskModel(),**kwargs)
            with patch.object(sys,'argv',arguments), patch('autodrive.perception.yolopv2_semantic.YOLOPv2SemanticDetector',side_effect=detector), \
                 patch('autodrive.runtime.onboard.create_chassis') as chassis, \
                 patch('cloud_client.client.CloudClient') as cloud, contextlib.redirect_stdout(io.StringIO()):
                onboard.main()
                chassis.assert_not_called();cloud.assert_not_called()
            status=json.loads((root/'output/status.json').read_text())
            self.assertTrue(status['termination']['stopped'])
            self.assertFalse(status['cloud_arbitration']['enabled'])
            self.assertTrue(status['cloud_arbitration']['closed'])
            self.assertEqual(status['diagnostics']['control_log']['written_rows'],3)
            self.assertIsNone(status['run_archive'])

if __name__=='__main__':unittest.main()
