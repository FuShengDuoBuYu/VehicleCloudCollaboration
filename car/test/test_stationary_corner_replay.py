"""Recorded IMU inputs must remain exact and hardware-free during corner replay."""
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from car.autodrive.tools.replay_outer_loop import recorded_stationary_inputs, replay_recorded
from autodrive.control.lane_centering import DifferentialDriveCommand, LaneEstimate
from autodrive.perception.outer_loop import BoundaryTrackResult


def row(**changes):
    data = {'control_monotonic': '10.1', 'imu_yaw_rad': '-2.3',
            'imu_received_monotonic': '10.05', 'imu_valid': 'True',
            'imu_stale': 'False', 'imu_age_s': '.05',
            'frame_age_exact_s': '.04', 'stationary_external_allowed': 'True',
            'stationary_perception_budget_valid': 'True'}
    data.update(changes)
    return data


class RecordedCornerInputTests(unittest.TestCase):
    def test_reconstructs_exact_receipt_and_quality_flags(self):
        value = recorded_stationary_inputs(row())
        self.assertEqual(value['now'], 10.1)
        self.assertEqual(value['imu_sample']['value']['radians'], [0., 0., -2.3])
        self.assertEqual(value['imu_sample']['received_monotonic'], 10.05)
        self.assertTrue(value['imu_sample']['valid'])
        self.assertFalse(value['imu_sample']['stale'])
        self.assertEqual(value['frame_age_seconds'], .04)

    def test_preserves_missing_or_invalid_imu_without_fabricating_angles(self):
        value = recorded_stationary_inputs(row(imu_yaw_rad='', imu_received_monotonic='',
                                                imu_valid='False', imu_stale='True', imu_age_s=''))
        self.assertIsNone(value['imu_sample']['value']['radians'][2])
        self.assertIsNone(value['imu_sample']['received_monotonic'])
        self.assertFalse(value['imu_sample']['valid'])
        self.assertTrue(value['imu_sample']['stale'])

    def test_requires_recorded_fields_not_live_clock_or_default_imu(self):
        data = row(); del data['control_monotonic']
        with self.assertRaisesRegex(ValueError, 'stationary.*inputs'):
            recorded_stationary_inputs(data)

    def test_rejects_corrupt_numbers_and_boolean_strings(self):
        for name, bad in [('control_monotonic', 'nan'), ('imu_yaw_rad', 'inf'),
                          ('frame_age_exact_s', '-.01'), ('imu_valid', 'yes')]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                recorded_stationary_inputs(row(**{name: bad}))

    def test_preserves_external_and_perception_vetoes(self):
        value = recorded_stationary_inputs(row(stationary_external_allowed='False',
                                                stationary_perception_budget_valid='False'))
        self.assertFalse(value['external_motion_allowed'])
        self.assertFalse(value['perception_budget_valid'])

    def test_complete_recorded_turn_reuses_imu_and_never_opens_hardware(self):
        self.check_complete_recorded_turn()

    def test_recorded_final_application_veto_overrides_otherwise_valid_motion(self):
        self.check_complete_recorded_turn(final_veto=True)

    def check_complete_recorded_turn(self, final_veto=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root/'run'; inputs = run/'semantic_inputs'
            inputs.mkdir(parents=True)
            source_profile = Path(__file__).resolve().parents[1]/'control/vehicle_control/profiles/rosmaster_jetson_yolopv2_outer_trial.yaml'
            config = yaml.safe_load(source_profile.read_text())
            overrides = config['runtime_overrides']
            overrides['stationary_corner'] = {'enabled': True}
            overrides['wheels']['stationary_pivot_pwm'] = 30
            overrides['perception']['semantic_outer_loop']['stationary_corners'] = True
            profile = root/'profile.yaml'; profile.write_text(yaml.safe_dump(config))
            rows = []
            for i in range(26):
                now = 1000.+i*.05
                yaw = -min(max(0, i-11)*.4, math.pi/2)
                datum = row(control_monotonic=str(now), imu_yaw_rad=str(yaw),
                            imu_received_monotonic=str(now), imu_age_s='0')
                datum.update(timestamp_s=i*.05, sample=i, semantic_sequence=i,
                             semantic_result_age_exact_s=.01, semantic_result_age_s=.01,
                             semantic_fusion_source='yolopv2', action='stop')
                if final_veto:
                    datum.update(stationary_application_allowed='False' if i==25 else 'True',
                        stationary_application_monotonic=str(now+.01),
                        stationary_application_reason='late semantic expiry' if i==25 else '',
                        semantic_application_age_s='.31' if i==25 else '.02',
                        frame_application_age_s='.02',imu_application_age_s='.01')
                rows.append(datum)
                np.savez(inputs/('%08d.npz' % i), sequence=i, captured_at=now-.01,
                         road=np.ones((240,320), np.uint8), lane=np.zeros((240,320),np.uint8),
                         frame=np.zeros((480,640,3),np.uint8), detections_json='[]')
            with (run/'onboard_log.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
            (run/'status.json').write_text(json.dumps({'termination':{'stopped':True,'completed_at':100.},
                'last_result':{'sample':25}, 'diagnostics':{'archive':{'written_frames':26,'dropped_frames':0,'error':None}}}))
            hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
            def analyze(detector, estimator, controller, mapper, frame, *args, **kwargs):
                i = detector._consumer_result.sequence
                near_missing = 12 <= i <= 15
                front = .7 if i < 5 else .78 if i <= 11 else None
                boundary = BoundaryTrackResult(not near_missing, .9, np.ones((240,320),np.uint8),
                    'yolopv2', semantic_sequence=i, semantic_result_age_seconds=.01,
                    semantic_front_boundary_ratio=front, semantic_right_exit_observed=True,
                    semantic_hard_safe=True)
                estimate = LaneEstimate(not near_missing, .9 if not near_missing else 0)
                command = (DifferentialDriveCommand('stop',0,0,0,0,'near road missing') if near_missing
                           else DifferentialDriveCommand('forward',0,.2,.2,.9,'synthetic road'))
                return estimate,command,frame.copy(),0.,0.,boundary
            with patch('car.autodrive.tools.replay_outer_loop.analyze',side_effect=analyze), \
                 patch('autodrive.runtime.onboard.create_chassis') as chassis:
                summary = replay_recorded(run, root/'derived', str(profile))
                chassis.assert_not_called()
            commands = json.loads((root/'derived/commands.json').read_text())
            self.assertTrue(summary['stationary_corner_replayed'])
            self.assertIn('pivot-right',summary['actions'])
            self.assertTrue(all(c['pwm']==[30,30,-30,-30] for c in commands if c['action']=='pivot-right'))
            self.assertEqual(commands[15]['action'],'stop')
            self.assertEqual(commands[15]['stationary_corner']['stationary_phase'],'exit')
            self.assertEqual(commands[-1]['action'],'stop' if final_veto else 'forward')
            if final_veto:
                self.assertEqual(commands[-1]['pwm'],[0,0,0,0])
                self.assertEqual(summary['recorded_final_application_rows'],26)
                self.assertEqual(summary['final_application_veto_rows'],1)
                self.assertFalse(commands[-1]['stationary_corner']['stationary_startup_ready'])
            else:
                self.assertEqual(summary['recorded_final_application_rows'],0)
            for p,h in hashes.items(): self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),h)


if __name__ == '__main__':
    unittest.main()
