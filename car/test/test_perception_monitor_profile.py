"""Profile selection tests; intercept exec before any camera/model is opened."""
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

CAR = Path(__file__).resolve().parents[1]
for directory in (CAR, CAR / 'control'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from autodrive.web import perception_monitor
from vehicle_control import profile as profile_loader


class MonitorProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profiles = self.root / 'car/control/vehicle_control/profiles'
        self.profiles.mkdir(parents=True)
        self.config = self.root / 'car/autodrive/config/onboard_runtime.yaml'
        self.config.parent.mkdir(parents=True)
        self.config.write_text(yaml.safe_dump({'version': 1, 'vehicle': {},
            'camera': {'gimbal': {'initialize_on_startup': True}},
            'cloud_arbitration': {'enabled': True}, 'runtime': {'archive_runs': True}}))
        self.default = {'version': 1, 'id': 'rosmaster_jetson_yolopv2', 'backend': 'rosmaster',
            'status': {'motion_calibrated': True}, 'runtime_overrides': {
                'perception': {'mode': 'yolopv2', 'yolopv2': {'img_size': 352}},
                'outer_loop': {'enabled': False}}}
        self.default_path = self.profiles / 'rosmaster_jetson_yolopv2.yaml'
        self.default_path.write_text(yaml.safe_dump(self.default))

    def launch(self, arguments):
        original_cwd = Path.cwd()
        os.chdir(self.root)
        try:
            with patch.object(perception_monitor, '__file__',
                              str(self.root / 'car/autodrive/web/perception_monitor.py')), \
                 patch.object(profile_loader, 'PROFILE_DIR', self.profiles), \
                 patch.object(sys, 'argv', ['perception_monitor'] + arguments), \
                 patch.object(perception_monitor.os, 'execv') as execute:
                perception_monitor.main()
                executable, argv = execute.call_args.args
        finally:
            os.chdir(original_cwd)
        self.assertEqual(executable, sys.executable)
        self.assertNotIn('--enable-motors', argv)
        self.assertNotIn('--field-trial', argv)
        self.assertIn('--no-run-archive', argv)
        generated = Path(argv[argv.index('--vehicle-profile') + 1])
        _, resolved = profile_loader.load_runtime_config(self.config, str(generated))
        self.assertFalse(resolved['vehicle']['status']['motion_calibrated'])
        self.assertFalse(resolved['camera']['gimbal']['initialize_on_startup'])
        self.assertFalse(resolved['cloud_arbitration']['enabled'])
        self.assertFalse(resolved['runtime']['archive_runs'])
        return resolved

    def test_default_keeps_existing_yolo_profile_and_read_only_guards(self):
        config = self.launch([])
        self.assertEqual(config['perception']['yolopv2']['img_size'], 352)
        self.assertFalse(config['outer_loop']['enabled'])

    def test_explicit_path_preserves_experiment_parameters_and_source_file(self):
        selected = deepcopy(self.default)
        selected['id'] = 'experiment_320'
        selected['runtime_overrides'] = {
            'perception': {'mode': 'yolopv2', 'yolopv2': {'img_size': 320, 'half': True},
                'semantic_outer_loop': {'enabled': True, 'route_side': 'left'},
                'semantic_stability': {'enabled': True, 'threshold': .71}},
            'centerline': {'lookahead_ratio': .67}, 'lcc': {'base_speed': .37},
            'wheels': {'pwm_limit': 30}, 'outer_loop': {'enabled': False},
            'camera': {'gimbal': {'initialize_on_startup': True, 'pan_angle': 25}},
            'cloud_arbitration': {'enabled': True, 'retry_seconds': 3},
            'runtime': {'archive_runs': True, 'route_hint': 'left'}}
        path = self.root / 'selected experiment.yaml'
        path.write_text(yaml.safe_dump(selected))
        before = path.read_bytes()
        config = self.launch(['--vehicle-profile', str(path)])
        self.assertEqual(config['perception']['yolopv2']['img_size'], 320)
        for section in ('perception', 'centerline', 'lcc', 'wheels', 'outer_loop'):
            self.assertEqual(config[section], selected['runtime_overrides'][section])
        self.assertEqual(config['camera']['gimbal']['pan_angle'], 25)
        self.assertEqual(config['cloud_arbitration']['retry_seconds'], 3)
        self.assertEqual(config['runtime']['route_hint'], 'left')
        self.assertEqual(path.read_bytes(), before)

    def test_profile_id_and_relative_path_use_existing_resolver(self):
        selected = deepcopy(self.default)
        selected['id'] = 'custom_route'
        selected['runtime_overrides']['perception'] = {'mode': 'classical'}
        selected['runtime_overrides']['outer_loop']['enabled'] = True
        path = self.profiles / 'custom_route.yaml'
        path.write_text(yaml.safe_dump(selected))
        for selector in ('custom_route', str(path.relative_to(self.root))):
            with self.subTest(selector=selector):
                config = self.launch(['--vehicle-profile', selector])
                self.assertEqual(config['perception']['mode'], 'classical')
                self.assertTrue(config['outer_loop']['enabled'])

    def test_missing_profile_is_rejected_without_falling_back(self):
        with self.assertRaises(FileNotFoundError):
            self.launch(['--vehicle-profile', str(self.root / 'missing.yaml')])

    def test_motion_flags_cannot_be_forwarded_to_onboard(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.launch(['--enable-motors'])
        self.assertEqual(error.exception.code, 2)

    def test_launcher_passes_selector_without_starting_runtime(self):
        script = self.root / 'run_vehicle_dashboard.sh'
        shutil.copy2(CAR.parent / script.name, script)
        capture = self.root / 'arguments.json'
        runner = self.root / 'capture-python'
        runner.write_text('#!' + sys.executable + '\nimport json, os, sys\n'
                          'open(os.environ["MONITOR_TEST_ARGUMENTS"], "w").write(json.dumps(sys.argv[1:]))\n')
        runner.chmod(0o755)
        result = subprocess.run(['bash', str(script), 'perception', '--vehicle-profile',
                                 'selected experiment.yaml'],
                                env=dict(os.environ, CAR_PYTHON=str(runner),
                                         MONITOR_TEST_ARGUMENTS=str(capture)),
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(capture.read_text()),
                         ['-B', '-m', 'autodrive.web.perception_monitor', '--vehicle-profile',
                          'selected experiment.yaml'])


if __name__ == '__main__':
    unittest.main()
