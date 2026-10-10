"""Offline bundle contracts using temporary videos, with no hardware imports."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from car.autodrive.simulation.export_bundle import export_bundle, pairs, read_run


class SimulationBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root/'run'
        self.run.mkdir()
        self.rows = [dict(sample=i, timestamp_s=10+i*.1, action='forward', reason='road',
                          front_left_pwm=30, rear_left_pwm=30,
                          front_right_pwm=30, rear_right_pwm=30,
                          confidence=.75, heading_error=.2) for i in range(3)]
        self.write_rows()
        self.write_video(3)
        self.viewer = self.root/'viewer.html'
        self.viewer.write_text('<!doctype html><title>Offline fixture</title>')

    def write_rows(self):
        keys = list(dict.fromkeys(key for row in self.rows for key in row))
        with (self.run/'onboard_log.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_video(self, count, name='raw.mp4', *, folder=None, fps=10., color_offset=0):
        writer = cv2.VideoWriter(str((folder or self.run)/name), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (32, 24))
        if not writer.isOpened():
            self.fail('temporary file video encoder unavailable')
        try:
            for i in range(count):
                writer.write(np.full((24, 32, 3), color_offset+i*40, dtype=np.uint8))
        finally:
            writer.release()

    def candidate(self, frames=None, input_run=None):
        path = self.root/'candidate'
        path.mkdir()
        commands = [dict(frame=i, action='stop', pwm=[0, 0, 0, 0], reason='candidate',
                         stationary_corner={'stationary_phase':'settle'})
                    for i in (range(3) if frames is None else frames)]
        (path/'commands.json').write_text(json.dumps(commands))
        (path/'summary.json').write_text(json.dumps({'input_run':str(input_run or self.run)}))
        return path

    def test_reads_source_and_candidate_with_distinct_capture_and_video_times(self):
        data = read_run(self.run, self.candidate())
        self.assertEqual(data['video_fps'], 10.)
        self.assertEqual([item['video_t'] for item in data['frames']], [0., .1, .2])
        self.assertEqual([item['capture_t'] for item in data['frames']], [10., 10.1, 10.2])
        self.assertEqual(data['frames'][1]['pwm'], [30, 30, 30, 30])
        self.assertEqual(data['frames'][1]['candidate']['pwm'], [0, 0, 0, 0])
        self.assertEqual(data['frames'][1]['candidate']['phase'], 'settle')

    def test_rejects_nonconsecutive_control_samples(self):
        self.rows[1]['sample'] = 4
        self.write_rows()
        with self.assertRaises(ValueError):
            read_run(self.run)

    def test_rejects_empty_control_log(self):
        (self.run/'onboard_log.csv').write_text('sample,timestamp_s\n')
        with self.assertRaises(ValueError):
            read_run(self.run)

    def test_rejects_video_shorter_than_control_log(self):
        self.write_video(2)
        with self.assertRaises(ValueError):
            read_run(self.run)

    def test_rejects_video_longer_than_control_log(self):
        self.write_video(4)
        with self.assertRaises(ValueError):
            read_run(self.run)

    def test_rejects_candidate_length_mismatch(self):
        with self.assertRaises(ValueError):
            read_run(self.run, self.candidate([0, 1]))

    def test_rejects_candidate_frame_sequence_mismatch(self):
        with self.assertRaises(ValueError):
            read_run(self.run, self.candidate([0, 1, 3]))

    def test_rejects_candidate_from_another_input_run(self):
        with self.assertRaises(ValueError):
            read_run(self.run, self.candidate(input_run=self.root/'other-run'))

    def test_missing_imu_encoder_and_semantic_numbers_remain_unknown(self):
        value = read_run(self.run)['frames'][0]
        for key in ('yaw_rad', 'imu_age', 'encoder', 'semantic_age', 'semantic_sequence', 'front_ratio'):
            self.assertIsNone(value[key], key)

    def test_invalid_imu_and_encoder_do_not_expose_untrusted_values(self):
        self.rows[0].update(imu_valid='False', imu_yaw_rad='1.5', encoder_valid='False',
                            encoder_1='10', encoder_2='20', encoder_3='30', encoder_4='40')
        self.write_rows()
        value = read_run(self.run)['frames'][0]
        self.assertIsNone(value['yaw_rad'])
        self.assertIsNone(value['encoder'])

    def test_valid_imu_and_encoder_keep_recorded_values_and_missing_channels(self):
        self.rows[0].update(imu_valid='True', imu_yaw_rad='-1.25', imu_age_s='.05',
                            encoder_valid='True', encoder_1='10', encoder_2='',
                            encoder_3='30', encoder_4='40', confidence='nan')
        self.write_rows()
        value = read_run(self.run)['frames'][0]
        self.assertEqual(value['yaw_rad'], -1.25)
        self.assertEqual(value['imu_age'], .05)
        self.assertEqual(value['encoder'], [10., None, 30., 40.])
        self.assertIsNone(value['confidence'])

    def test_export_links_media_and_hashes_sources_without_modifying_them(self):
        candidate = self.candidate()
        self.write_video(3, 'annotated.mp4')
        (self.run/'status.json').write_text('{"termination":{"stopped":true}}')
        (self.run/'config.yaml').write_text('motors_enabled: false\n')
        course = self.root/'course.png'
        course.write_bytes(b'course fixture')
        synthetic = self.root/'synthetic.json'
        synthetic.write_text('{"frames":[],"summary":{"kind":"fixture"}}')
        source_files = [p for directory in (self.run, candidate) for p in directory.iterdir()]
        source_files.extend((course, synthetic))
        before = {p:(p.read_bytes(), p.stat().st_mtime_ns) for p in source_files}
        output = self.root/'export'
        data = export_bundle(output, {'recorded':self.run}, {'recorded':candidate},
                             course, synthetic, self.viewer)
        self.assertEqual(len(data['runs']), 1)
        self.assertTrue(data['synthetic']['label'].startswith('SYNTHETIC'))
        self.assertTrue((output/'media/recorded-raw.mp4').is_symlink())
        self.assertEqual((output/'media/recorded-raw.mp4').resolve(), self.run/'raw.mp4')
        self.assertTrue((output/'media/recorded-annotated.mp4').is_symlink())
        self.assertTrue((output/'media/course.png').is_symlink())
        self.assertEqual((output/'index.html').read_bytes(), self.viewer.read_bytes())
        emitted = (output/'data.js').read_text()
        decoded = json.loads(emitted[len('window.VEHICLE_SIM_DATA='):-2])
        self.assertEqual(decoded['runs'][0]['frames'][0]['candidate']['action'], 'stop')
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual({item['path'] for item in manifest}, {str(p) for p in source_files})
        for item in manifest:
            self.assertEqual(item['sha256'], hashlib.sha256(Path(item['path']).read_bytes()).hexdigest())
        self.assertEqual(before, {p:(p.read_bytes(), p.stat().st_mtime_ns) for p in source_files})

    def test_invalid_candidate_does_not_create_partial_bundle(self):
        output = self.root/'export'
        with self.assertRaises(ValueError):
            export_bundle(output, {'recorded':self.run}, {'recorded':self.candidate([0])},
                          viewer_path=self.viewer)
        self.assertFalse(output.exists())

    def test_browser_media_links_aligned_derived_copy_and_preserves_source_provenance(self):
        browser = self.root/'browser'
        browser.mkdir()
        self.write_video(3, 'recorded-raw.mp4', folder=browser, color_offset=10)
        original = self.run/'raw.mp4'
        derived = browser/'recorded-raw.mp4'
        before = {p:(p.read_bytes(), p.stat().st_mtime_ns)
                  for p in (original, derived, self.run/'onboard_log.csv')}
        self.assertNotEqual(before[original][0], before[derived][0])
        output = self.root/'export'
        data = export_bundle(output, {'recorded':self.run}, viewer_path=self.viewer,
                             browser_media=browser)
        linked = output/'media/recorded-raw.mp4'
        self.assertTrue(linked.is_symlink())
        self.assertEqual(linked.resolve(), derived)
        self.assertEqual(data['runs'][0]['source_run'], str(self.run))
        manifest = json.loads((output/'manifest.json').read_text())
        record = next(item for item in manifest if item['path'] == str(original))
        self.assertEqual(record['sha256'], hashlib.sha256(before[original][0]).hexdigest())
        self.assertEqual(record['display_path'], str(derived))
        self.assertEqual(record['display_sha256'], hashlib.sha256(before[derived][0]).hexdigest())
        self.assertEqual(before, {p:(p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_browser_frame_count_mismatch_is_rejected_before_creating_output(self):
        browser = self.root/'browser'
        browser.mkdir()
        self.write_video(2, 'recorded-raw.mp4', folder=browser)
        original = (self.run/'raw.mp4').read_bytes()
        output = self.root/'export'
        with self.assertRaises(ValueError):
            export_bundle(output, {'recorded':self.run}, viewer_path=self.viewer,
                          browser_media=browser)
        self.assertFalse(output.exists())
        self.assertEqual((self.run/'raw.mp4').read_bytes(), original)

    def test_browser_fps_mismatch_is_rejected_before_creating_output(self):
        browser = self.root/'browser'
        browser.mkdir()
        self.write_video(3, 'recorded-raw.mp4', folder=browser, fps=12.)
        output = self.root/'export'
        with self.assertRaises(ValueError):
            export_bundle(output, {'recorded':self.run}, viewer_path=self.viewer,
                          browser_media=browser)
        self.assertFalse(output.exists())

    def test_export_hashes_original_runtime_configuration_snapshots(self):
        snapshots = []
        for name in ('runtime_config.yaml', 'resolved_runtime_config.yaml', 'vehicle_profile.yaml'):
            path = self.run/name
            path.write_text('snapshot_name: '+name+'\n')
            snapshots.append(path)
        before = {path:path.read_bytes() for path in snapshots}
        output = self.root/'export'
        export_bundle(output, {'recorded':self.run}, viewer_path=self.viewer)
        manifest = {item['path']:item for item in json.loads((output/'manifest.json').read_text())}
        self.assertTrue(set(map(str, snapshots)).issubset(manifest))
        for path in snapshots:
            self.assertEqual(manifest[str(path)]['sha256'], hashlib.sha256(before[path]).hexdigest())
            self.assertEqual(path.read_bytes(), before[path])

    def test_duplicate_run_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            pairs(['same=a', 'same=b'])


if __name__ == '__main__':
    unittest.main()
