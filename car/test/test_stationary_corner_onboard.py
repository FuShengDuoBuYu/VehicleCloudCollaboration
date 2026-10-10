"""Runtime corner integration uses supplied observations only, with no devices."""
import copy
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autodrive.runtime import onboard
from autodrive.control.drive_runtime import PerceptionMotionGate
from autodrive.control.lane_centering import DifferentialDriveCommand, LaneEstimate
from autodrive.control.stationary_corner import StationaryCornerConfig


def attitude(now, yaw=0.):
    return {'value': {'radians': [0., 0., yaw]}, 'valid': True, 'stale': False,
            'age_s': 0., 'received_monotonic': now}


class StationaryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(onboard, 'StationaryCornerRuntime'), 'missing shared runtime adapter')
        self.runtime = onboard.StationaryCornerRuntime(StationaryCornerConfig(enabled=True),
            PerceptionMotionGate(resume_valid_frames=5), pivot_verified=True)
        self.command = DifferentialDriveCommand('turn-right', .6, .5, .2, .9, 'local curve', .4)
        self.estimate = LaneEstimate(True, .9, heading_error=.1)

    def tick(self, now, sequence, front=.7, yaw=0., *, local=True, hard=True,
             boundary_changes=None, **changes):
        boundary = SimpleNamespace(valid=local, source='yolopv2', semantic_sequence=sequence,
            semantic_result_age_seconds=.01, semantic_front_boundary_ratio=front,
            semantic_right_exit_observed=True, semantic_hard_safe=hard)
        for name, value in (boundary_changes or {}).items():
            setattr(boundary, name, value)
        command = self.command if local else DifferentialDriveCommand('stop', 0., 0., 0., 0., 'near road missing')
        estimate = self.estimate if local else LaneEstimate(False, 0., reason='near road missing')
        arguments = dict(now=now, imu_sample=attitude(now, yaw), frame_age_seconds=.01,
                         perception_budget_valid=True, external_motion_allowed=True)
        arguments.update(changes)
        return self.runtime.filter(command, estimate, boundary, **arguments)

    def start_pivot(self):
        for sequence in range(1, 6):
            action = self.tick(sequence*.05, sequence)
            self.assertEqual(action.action, 'stop' if sequence < 5 else 'forward')
        self.assertEqual(self.tick(.30, 6, .78).action, 'stop')
        self.assertEqual(self.tick(.35, 7, .78).action, 'stop')
        self.assertEqual(self.tick(.59, 8, .78).action, 'stop')
        self.assertEqual(self.tick(.61, 9, .78).action, 'pivot-right')

    def test_initial_five_observations_and_approach_never_arc(self):
        self.start_pivot()
        self.assertEqual(self.runtime.get_state()['stationary_phase'], 'pivot-right')

    def test_nonpivot_veto_revokes_permission_and_requires_five_new_observations(self):
        for veto in ({'external_motion_allowed': False}, {'local': False}, {'hard': False},
                     {'frame_age_seconds': .3}, {'perception_budget_valid': False}):
            with self.subTest(veto=veto):
                self.setUp()
                for sequence in range(1, 6):
                    self.tick(sequence*.05, sequence)
                self.assertEqual(self.tick(.30, 6, **veto).action, 'stop')
                self.assertIs(self.runtime.get_state()['stationary_startup_ready'], False)
                self.assertEqual(self.tick(.35, 7, .78).action, 'stop')
                self.assertEqual(self.tick(.40, 8, .78).action, 'stop')
                # A cached second observation must not finish entry or renew
                # the permission that the preceding veto revoked.
                self.assertEqual(self.tick(.67, 8, .78,
                    boundary_changes={'semantic_result_age_seconds': .27}).action, 'stop')
                self.assertEqual(self.runtime.get_state()['motion_gate']['consecutive_valid'], 2)
                for sequence, now in ((9, .72), (10, .77), (11, .82), (12, .87)):
                    self.assertEqual(self.tick(now, sequence, .78).action, 'stop')
                self.assertEqual(self.tick(1.14, 12, .78,
                    boundary_changes={'semantic_result_age_seconds': .27}).action, 'pivot-right')

    def test_normal_settle_zero_keeps_current_permission(self):
        for sequence in range(1, 6):
            self.tick(sequence*.05, sequence)
        for sequence, now in ((6, .30), (7, .35), (8, .59)):
            self.assertEqual(self.tick(now, sequence, .78).action, 'stop')
            self.assertIs(self.runtime.get_state()['stationary_startup_ready'], True)
            self.assertIs(self.runtime.get_state()['motion_gate']['ready'], True)
        self.assertEqual(self.tick(.61, 9, .78).action, 'pivot-right')

    def test_cached_veto_observation_cannot_count_as_new_recovery_evidence(self):
        for sequence in range(1, 6):
            self.tick(sequence*.05, sequence)
        self.tick(.30, 6, external_motion_allowed=False)
        self.assertEqual(self.tick(.35, 6).action, 'stop')
        self.assertEqual(self.runtime.get_state()['motion_gate']['consecutive_valid'], 0)
        for sequence, now in ((7, .40), (8, .45), (9, .50), (10, .55), (11, .60)):
            action = self.tick(now, sequence)
            self.assertEqual(action.action, 'stop' if sequence < 11 else 'forward')

    def test_consumed_capture_time_revokes_aged_mask_despite_fresh_camera_frame(self):
        for sequence in range(1, 6):
            self.tick(sequence*.05, sequence)
        boundary = SimpleNamespace(valid=True, source='yolopv2', semantic_sequence=6,
            semantic_result_age_seconds=.29, semantic_captured_at=.71,
            semantic_front_boundary_ratio=.7, semantic_right_exit_observed=True,
            semantic_hard_safe=True)
        action = self.runtime.filter(self.command, self.estimate, boundary, now=1.05,
            imu_sample=attitude(1.05), frame_age_seconds=.10)
        self.assertEqual(action.action, 'stop')
        self.assertIs(self.runtime.get_state()['stationary_startup_ready'], False)
        self.assertAlmostEqual(boundary.semantic_result_age_seconds, .34)

    def test_application_veto_pauses_pivot_and_cached_semantic_cannot_resume_it(self):
        self.start_pivot()
        self.tick(.70, 10, None, -.5, local=False)
        self.assertEqual(self.runtime.apply_veto('late frame application veto').action, 'stop')
        state = self.runtime.get_state()
        self.assertEqual(state['stationary_decision'], 'stop')
        self.assertIs(state['stationary_application_allowed'], False)
        self.assertIs(state['stationary_startup_ready'], False)
        self.assertIs(state['controller']['paused'], True)
        self.assertAlmostEqual(state['controller']['yaw_progress_rad'], .5)
        self.assertAlmostEqual(state['controller']['pivot_started'], .61)
        self.assertEqual(self.tick(.75, 10, None, -.6, local=False).action, 'stop')
        self.assertEqual(self.tick(.80, 11, None, -.65, local=False).action, 'pivot-right')
        self.assertAlmostEqual(self.runtime.get_state()['controller']['yaw_progress_rad'], .65)
        self.assertAlmostEqual(self.runtime.get_state()['controller']['pivot_started'], .61)

    def test_short_active_pause_preserves_accumulated_yaw_and_original_deadline(self):
        self.start_pivot()
        self.assertEqual(self.tick(.70, 10, None, -.5, local=False).action, 'pivot-right')
        self.assertEqual(self.tick(.75, 11, None, -.6, local=False,
                                   external_motion_allowed=False).action, 'stop')
        self.assertEqual(self.tick(.85, 12, None, -.65, local=False).action, 'pivot-right')
        state = self.runtime.get_state()['controller']
        self.assertAlmostEqual(state['yaw_progress_rad'], .65)
        self.assertAlmostEqual(state['pivot_started'], .61)
        self.assertEqual(self.tick(.95, 13, None, -1.2, local=False).action, 'pivot-right')
        self.assertEqual(self.tick(1.05, 14, None, -math.pi/2, local=False).action, 'stop')
        self.assertEqual(self.runtime.get_state()['stationary_phase'], 'exit')

    def test_active_pause_cannot_extend_total_pivot_timeout(self):
        self.runtime = onboard.StationaryCornerRuntime(
            StationaryCornerConfig(enabled=True, max_pivot_seconds=1.),
            PerceptionMotionGate(resume_valid_frames=5), pivot_verified=True)
        self.start_pivot()
        self.tick(.70, 10, None, -.5, local=False)
        self.tick(.75, 11, None, -.6, local=False, external_motion_allowed=False)
        self.assertEqual(self.tick(1.62, 12, None, -.65, local=False).action, 'stop')
        self.assertEqual(self.runtime.get_state()['stationary_reason'], 'pivot-total-timeout')

    def test_active_verified_pivot_survives_near_road_failure_and_stops_at_yaw_target(self):
        self.start_pivot()
        for sequence, yaw in ((10, -.5), (11, -1.0)):
            self.assertEqual(self.tick(.7+(sequence-10)*.1, sequence, None, yaw, local=False).action,
                             'pivot-right')
        self.assertEqual(self.tick(.9, 12, None, -math.pi/2, local=False).action, 'stop')
        self.assertEqual(self.runtime.get_state()['stationary_phase'], 'exit')
        for i in range(3):
            self.assertEqual(self.tick(1.+i*.1, 13+i, None, -math.pi/2).action, 'stop')
            self.assertEqual(self.runtime.get_state()['stationary_phase'],
                             'exit' if i < 2 else 'cruise')
            self.assertEqual(self.runtime.get_state()['motion_gate']['consecutive_valid'], 0)
        self.assertEqual(self.tick(1.25, 15, None, -math.pi/2).action, 'stop')
        self.assertEqual(self.runtime.get_state()['motion_gate']['consecutive_valid'], 0)
        self.assertEqual(self.tick(1.30, 16, None, -math.pi/2).action, 'stop')
        self.assertEqual(self.tick(1.34, 16, None, -math.pi/2).action, 'stop')
        self.assertEqual(self.runtime.get_state()['motion_gate']['consecutive_valid'], 1)
        for i in range(4):
            action = self.tick(1.4+i*.1, 17+i, None, -math.pi/2)
            self.assertEqual(action.action, 'stop' if i < 3 else 'turn-right')

    def test_raw_hazard_camera_budget_and_external_veto_always_stop_pivot(self):
        for changes in ({'hard': False}, {'frame_age_seconds': .3},
                        {'perception_budget_valid': False}, {'external_motion_allowed': False}):
            with self.subTest(changes=changes):
                self.setUp(); self.start_pivot()
                self.assertEqual(self.tick(.7, 10, None, -.1, local=False, **changes).action, 'stop')

    def test_invalid_stale_and_future_imu_blocks_pivot(self):
        for change in ({'valid': False}, {'stale': True}, {'age_s': .3},
                       {'received_monotonic': .1}, {'received_monotonic': .8},
                       {'value': {'radians': [0., 0., float('nan')]}}):
            with self.subTest(change=change):
                self.setUp(); self.start_pivot()
                sample = attitude(.7, -.1); sample.update(change)
                self.assertEqual(self.tick(.7, 10, None, local=False, imu_sample=sample).action, 'stop')
                self.assertEqual(self.runtime.get_state()['stationary_phase'], 'blocked')

    def test_unverified_runtime_cannot_enter_pivot(self):
        self.runtime = onboard.StationaryCornerRuntime(StationaryCornerConfig(enabled=True),
            PerceptionMotionGate(resume_valid_frames=5), pivot_verified=False)
        for i in range(20):
            self.assertNotEqual(self.tick(i*.1, i+1, .78).action, 'pivot-right')

    def test_disabled_adapter_preserves_existing_gate_behavior(self):
        self.runtime = onboard.StationaryCornerRuntime(StationaryCornerConfig(),
            PerceptionMotionGate(resume_valid_frames=1))
        self.assertEqual(self.tick(.1, 1).action, 'turn-right')

    def test_logged_inputs_recreate_exact_command_sequence(self):
        self.start_pivot()
        self.tick(.7, 10, None, -.2, local=False)
        state = self.runtime.get_state()
        self.assertEqual(state['control_monotonic'], .7)
        self.assertEqual(state['imu_yaw_rad'], -.2)
        self.assertEqual(state['imu_received_monotonic'], .7)
        self.assertIs(state['imu_valid'], True)
        self.assertEqual(state['stationary_state_before'], 'pivot-right')
        self.assertEqual(state['stationary_decision'], 'pivot-right')
        self.assertIs(state['semantic_hard_safe'], True)
        self.assertIsNone(state['semantic_front_boundary_ratio'])

    def test_current_boundary_diagnostics_are_preserved_for_logs_and_status(self):
        diagnostics = {'semantic_front_observed': True, 'semantic_observed_front_ratio': .78,
            'semantic_front_spread_ratio': .02, 'semantic_right_exit_support_ratio': .81,
            'semantic_corner_reject_reason': 'fixture-current-front',
            'semantic_pivot_road_support_ratio': .91}
        self.tick(.1, 1, boundary_changes=diagnostics)
        state = self.runtime.get_state()
        for field, expected in diagnostics.items():
            with self.subTest(field=field):
                self.assertEqual(state.get(field), expected)
                self.assertIn(field, onboard.STATIONARY_LOG_FIELDS)


class StationaryCalibrationTests(unittest.TestCase):
    def config(self):
        _, config = onboard.load_config(onboard.DEFAULT_CONFIG, 'rosmaster_jetson_yolopv2_outer_trial')
        config['stationary_corner'] = {'enabled': True, 'yaw_sign': -1}
        config['wheels']['stationary_pivot_pwm'] = 30
        config['perception']['semantic_outer_loop']['stationary_corners'] = True
        return config

    @contextlib.contextmanager
    def main_fixture(self, root, *, stop_fails=False, runtime_fails=False,
                     failed_close=None, telemetry=None, failed_snapshot=False,
                     late_application=False):
        """Real CPU perception, archive/log workers, and virtual PWM only."""
        from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
        from car.test.test_yolopv2_primary import MaskModel
        source = root/'source.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (640,480))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.zeros((480,640,3),np.uint8))
        writer.release()
        resources = {'stop_attempts': [], 'telemetry_reads': 0}
        config = self.config()
        if late_application:
            config['perception']['yolopv2']['asynchronous'] = False
        class CapturedVideoSource(onboard.VideoSource):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                resources['source'] = self
        serial_file = (root/'simulated-serial.txt').open('w')
        resources['serial_file'] = serial_file
        def close_chassis(*, stop=True):
            self.assertIs(stop, False)
            serial_file.close()
        def telemetry_snapshot():
            resources['telemetry_reads'] += 1
            return copy.deepcopy(telemetry or {})
        # No poll_available method: the fixture never starts a UART publisher.
        chassis = SimpleNamespace(close=close_chassis,
            controller=SimpleNamespace(telemetry=telemetry_snapshot))
        build = onboard.build_components
        analyze = onboard.analyze
        analyze_count = 0
        def components(*args, **kwargs):
            result = build(*args, **kwargs)
            resources['detector'], resources['driver'], resources['watchdog'] = result[0], result[5], result[6]
            result[5].chassis = chassis
            original_stop = result[5].stop
            resources['original_stop'] = original_stop
            def stop(reason):
                resources['stop_attempts'].append(reason)
                if stop_fails and (reason == 'watchdog disarmed'
                                  or reason.startswith(('maximum samples', 'runtime error'))):
                    raise RuntimeError('injected shutdown stop failure')
                return original_stop(reason)
            result[5].stop = stop
            if late_application:
                heartbeat = result[6].heartbeat
                heartbeat_count = 0
                def delayed_heartbeat():
                    nonlocal heartbeat_count
                    heartbeat_count += 1
                    heartbeat()
                    if heartbeat_count >= 9 and heartbeat_count % 2:
                        onboard.time.sleep(.06)
                result[6].heartbeat = delayed_heartbeat
            return result
        def analysis(*args, **kwargs):
            nonlocal analyze_count
            analyze_count += 1
            if runtime_fails and analyze_count == 2:
                raise ValueError('injected control failure')
            result = analyze(*args, **kwargs)
            if late_application:
                if analyze_count >= 6:
                    result[5].semantic_captured_at = onboard.time.monotonic()-.29
                    result[5].semantic_result_age_seconds = .29
                result[5].semantic_hard_safe = True
                result[5].semantic_front_boundary_ratio = None
                result[5].valid = True
                result = (LaneEstimate(True, 1.),
                    DifferentialDriveCommand('forward', 0., .3, .3, 1., 'fixture ordinary road'),
                    *result[2:])
            return result
        def capture_worker(name, factory):
            def create(*args, **kwargs):
                resource = factory(*args, **kwargs)
                resources[name] = resource
                if name == 'RunArchive' and failed_snapshot:
                    def snapshot_failure(*args, **kwargs):
                        raise OSError('injected archive snapshot failure')
                    resource.snapshot_mapping = snapshot_failure
                if name == failed_close:
                    close = resource.close
                    def fail_after_close():
                        close()
                        raise RuntimeError('injected '+name+' close failure')
                    resource.close = fail_after_close
                return resource
            return create
        def detector(settings, **kwargs):
            return YOLOPv2SemanticDetector(settings, model=MaskModel(), **kwargs)
        argv = ['onboard', '--video', str(source), '--max-samples', '6' if late_application else '3',
                '--sample-every', '1', '--output-dir', str(root/'output')]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(onboard, 'VideoSource', CapturedVideoSource))
            stack.enter_context(patch.object(onboard, 'load_config',
                return_value=(onboard.DEFAULT_CONFIG, config)))
            stack.enter_context(patch('autodrive.perception.yolopv2_semantic.YOLOPv2SemanticDetector',
                                     side_effect=detector))
            stack.enter_context(patch.object(onboard, 'build_components', side_effect=components))
            stack.enter_context(patch.object(onboard, 'analyze', side_effect=analysis))
            for name in ('RunArchive', 'AsyncCsvLogger', 'LivePublisher', 'DebugFrameWriter'):
                stack.enter_context(patch.object(onboard, name,
                    side_effect=capture_worker(name, getattr(onboard, name))))
            try:
                yield resources
            finally:
                # RED failures must not leave fixture workers alive either.
                if 'driver' in resources:
                    resources['driver'].stop = resources['original_stop']
                    resources['watchdog'].close()
                    resources['detector'].close()
                for name in ('LivePublisher', 'DebugFrameWriter', 'AsyncCsvLogger', 'RunArchive'):
                    if name in resources:
                        try:
                            resources[name].close()
                        except RuntimeError:
                            pass
                serial_file.close()
                if 'source' in resources:
                    resources['source'].close()

    def assert_main_resources_closed(self, resources):
        self.assertTrue(resources['serial_file'].closed)
        self.assertTrue(resources['detector']._closed)
        self.assertFalse(resources['watchdog']._thread.is_alive())
        for name in ('AsyncCsvLogger', 'LivePublisher', 'DebugFrameWriter'):
            self.assertIsNone(resources[name]._thread)
        self.assertIsNone(resources['RunArchive']._writer_thread)
        self.assertFalse(resources['source'].capture.isOpened())

    def test_archive_snapshot_failure_closes_source_and_archive_before_component_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.main_fixture(root, failed_snapshot=True) as resources, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, 'injected archive snapshot failure'):
                    onboard.main()
                self.assertFalse(resources['source'].capture.isOpened())
                self.assertIsNone(resources['RunArchive']._writer_thread)
                self.assertNotIn('detector', resources)

    def test_final_application_veto_stops_mask_that_expires_after_initial_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.main_fixture(root, late_application=True) as resources, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(onboard.main(), 0)
                self.assert_main_resources_closed(resources)
                with (root/'output/onboard_log.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 6)
                for row in rows:
                    self.assertEqual(row['action'], 'stop')
                for row in rows[4:]:
                    self.assertIn('expired before application', row['reason'])
                    self.assertLess(float(row['semantic_result_age_exact_s']), .3)
                    self.assertGreater(float(row['semantic_application_age_s']), .3)
                    self.assertLess(float(row['frame_application_age_s']), .25)
                    self.assertEqual(row['stationary_application_allowed'], 'False')
                    self.assertEqual(row['stationary_startup_ready'], 'False')

    def test_shutdown_stop_exception_still_closes_serial_and_archive_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.main_fixture(root, stop_fails=True) as resources, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'injected shutdown stop failure'):
                    onboard.main()
                self.assert_main_resources_closed(resources)
                status = json.loads((root/'output/status.json').read_text())
                self.assertIs(status['termination']['stopped'], False)
                self.assertEqual(status['termination']['reason'], 'maximum samples reached (3)')
                self.assertTrue(status['termination']['cleanup_errors'])
                archive_status = json.loads(resources['RunArchive'].status_path.read_text())
                self.assertIs(archive_status['termination']['stopped'], False)
                self.assertIn('maximum samples reached (3)', resources['stop_attempts'])

    def test_runtime_exception_survives_cleanup_failure_after_all_resources_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.main_fixture(root, stop_fails=True, runtime_fails=True) as resources, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'injected control failure'):
                    onboard.main()
                self.assert_main_resources_closed(resources)
                status = json.loads((root/'output/status.json').read_text())
                self.assertIs(status['termination']['stopped'], False)
                self.assertIn('injected control failure', status['termination']['reason'])

    def test_repeated_sigint_during_shutdown_finishes_stop_and_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.main_fixture(root) as resources, contextlib.redirect_stdout(io.StringIO()):
                analyze = onboard.analyze
                calls = []
                def interrupt_control(*args, **kwargs):
                    calls.append(1)
                    if len(calls) == 3:
                        raise KeyboardInterrupt()
                    return analyze(*args, **kwargs)
                close = onboard.CommandWatchdog.close
                def repeated_interrupt(watchdog):
                    if not getattr(watchdog, '_test_signal_sent', False):
                        watchdog._test_signal_sent = True
                        self.assertIn('interrupted by operator', resources['stop_attempts'])
                        os.kill(os.getpid(), signal.SIGINT)
                    close(watchdog)
                previous_handler = signal.getsignal(signal.SIGINT)
                with patch.object(onboard, 'analyze', side_effect=interrupt_control), \
                     patch.object(onboard.CommandWatchdog, 'close', repeated_interrupt):
                    self.assertEqual(onboard.main(), 0)
                self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)
                self.assert_main_resources_closed(resources)
                status = json.loads((root/'output/status.json').read_text())
                self.assertIs(status['termination']['stopped'], True)
                self.assertEqual(status['termination']['reason'], 'interrupted by operator')
                self.assertEqual(status['termination']['cleanup_errors'], [])
                self.assertTrue(resources['RunArchive'].status_path.exists())

    def test_keyboard_interrupt_from_close_records_failure_and_closes_remaining_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.main_fixture(root) as resources, contextlib.redirect_stdout(io.StringIO()):
                close = onboard.CommandWatchdog.close
                def interrupt_after_close(watchdog):
                    close(watchdog)
                    if not getattr(watchdog, '_test_interrupt_sent', False):
                        watchdog._test_interrupt_sent = True
                        raise KeyboardInterrupt('injected repeated cleanup interruption')
                with patch.object(onboard.CommandWatchdog, 'close', interrupt_after_close):
                    with self.assertRaisesRegex(KeyboardInterrupt, 'repeated cleanup interruption'):
                        onboard.main()
                self.assert_main_resources_closed(resources)
                status = json.loads((root/'output/status.json').read_text())
                self.assertIs(status['termination']['stopped'], False)
                self.assertEqual(status['termination']['cleanup_errors'][0]['type'], 'KeyboardInterrupt')
                self.assertTrue(resources['RunArchive'].status_path.exists())

    def test_diagnostic_close_exception_does_not_skip_later_closes_or_final_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.main_fixture(root, failed_close='LivePublisher') as resources, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'injected LivePublisher close failure'):
                    onboard.main()
                self.assert_main_resources_closed(resources)
                status = json.loads((root/'output/status.json').read_text())
                self.assertIs(status['termination']['stopped'], True)
                self.assertTrue(status['termination']['cleanup_errors'])

    def test_main_logs_native_encoder_snapshot_once_per_control_frame(self):
        now = onboard.time.monotonic()
        telemetry = {'attitude': attitude(now, -.31),
            'encoder': {'value': {'native_ticks': [120, -30, 2147483640, -2147483640]},
                'received_monotonic': now-.02, 'age_s': .02, 'valid': False,
                'stale': True, 'sequence': 7, 'frame_type': 13},
            'imu': {'value': {'gyro_rad_s': [0., 0., .91]}, 'valid': True},
            'motion': {'value': {'battery_v': 9.7}, 'valid': True}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.main_fixture(root, telemetry=telemetry) as resources, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(onboard.main(), 0)
                self.assertEqual(resources['telemetry_reads'], 3)
                self.assert_main_resources_closed(resources)
                with (root/'output/onboard_log.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 3)
                for row in rows:
                    self.assertEqual([row.get('encoder_'+str(i)) for i in range(1, 5)],
                                     ['120', '-30', '2147483640', '-2147483640'])
                    self.assertAlmostEqual(float(row['encoder_received_monotonic']), now-.02)
                    self.assertEqual(row['encoder_age_s'], '0.02')
                    self.assertEqual(row['encoder_valid'], 'False')
                    self.assertEqual(row['encoder_stale'], 'True')
                    self.assertAlmostEqual(float(row['imu_yaw_rad']), -.31)

    def test_partial_builder_failure_closes_created_model_worker(self):
        from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
        from car.test.test_yolopv2_primary import MaskModel
        config = self.config()
        config['centerline']['route_hint_bias'] = .7
        created = []
        def detector(settings, **kwargs):
            result = YOLOPv2SemanticDetector(settings, model=MaskModel(), **kwargs)
            created.append(result)
            return result
        with patch('autodrive.perception.yolopv2_semantic.YOLOPv2SemanticDetector',
                   side_effect=detector):
            try:
                with self.assertRaises(ValueError):
                    onboard.build_components(config, False, None)
                self.assertEqual(len(created), 1)
                self.assertTrue(created[0]._closed)
                self.assertIsNone(created[0]._thread)
            finally:
                for result in created:
                    result.close()

    def test_partial_builder_cleanup_stop_failure_preserves_original_error_and_closes_fake_serial(self):
        from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
        from car.test.test_yolopv2_primary import MaskModel
        config = self.config()
        config['stationary_corner']['enabled'] = False
        config['safety']['watchdog_check_interval'] = 0
        created = []
        def detector(settings, **kwargs):
            result = YOLOPv2SemanticDetector(settings, model=MaskModel(), **kwargs)
            created.append(result)
            return result
        with tempfile.TemporaryDirectory() as temporary:
            handle = (Path(temporary)/'simulated-serial.txt').open('w')
            attempts = []
            def stop():
                attempts.append('stop')
                raise RuntimeError('injected partial-builder stop failure')
            def close(*, stop=True):
                attempts.append('close')
                handle.close()
                raise RuntimeError('injected partial-builder close failure')
            # All motor transport is replaced by this file-backed fake. No
            # serial device is opened and only a stop is ever proposed.
            chassis = SimpleNamespace(stop=stop, close=close)
            with patch('autodrive.perception.yolopv2_semantic.YOLOPv2SemanticDetector',
                       side_effect=detector), \
                 patch.object(onboard, 'validate_motion_configuration'), \
                 patch.object(onboard, 'create_chassis', return_value=chassis):
                try:
                    with self.assertRaisesRegex(ValueError, 'watchdog timings must be positive'):
                        onboard.build_components(config, True, None)
                    self.assertEqual(attempts, ['stop', 'close'])
                    self.assertTrue(handle.closed)
                    self.assertTrue(created[0]._closed)
                    self.assertIsNone(created[0]._thread)
                finally:
                    handle.close()
                    for result in created:
                        result.close()

    def test_enabled_motor_trial_rejects_bool_only_or_nonexistent_evidence(self):
        for calibration in ({}, {'verified': True, 'yaw_sign': -1},
                            {'verified': True, 'yaw_sign': -1, 'evidence': '/missing/summary.json'}):
            config = self.config(); config['stationary_corner_calibration'] = calibration
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                onboard.validate_field_trial_config(config)

    def test_bad_stationary_setup_rejected_before_model_or_chassis(self):
        config = self.config()
        config['perception']['semantic_outer_loop']['stationary_corners'] = False
        with patch('autodrive.perception.yolopv2_semantic.YOLOPv2SemanticDetector',
                   side_effect=ValueError('model must not be reached')) as model, \
             patch.object(onboard, 'create_chassis') as chassis, self.assertRaises(ValueError):
            onboard.build_components(config, False, None)
        model.assert_not_called(); chassis.assert_not_called()

    def test_evidence_needs_matching_direction_stop_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            artifacts = {}
            for name in ('raw.avi', 'telemetry.jsonl', 'frames.jsonl', 'uart_rx.bin'):
                (root/name).write_bytes(b'test-only evidence fixture')
                artifacts[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
            summary = {'status': 'pulse_recorded', 'direction': 'right', 'pwm': 30,
                'errors': [], 'yaw_delta_rad': -.5, 'final_pwm': [0, 0, 0, 0],
                'profile': copy.deepcopy(config['vehicle']), 'video_frames': 10,
                'before': {'attitude': attitude(1., 0.)},
                'after': {'attitude': attitude(2., -.5)},
                'stop': {'final_pwm': [0, 0, 0, 0], 'error': None, 'reason': 'duration reached',
                         'command_started': 1.1, 'command_completed': 1.11,
                         'stop_started': 1.9, 'stop_completed': 1.95},
                'artifacts_sha256': artifacts}
            path = root/'summary.json'; path.write_text(json.dumps(summary))
            config['stationary_corner_calibration'] = {'verified': True, 'yaw_sign': -1, 'evidence': str(path)}
            onboard.validate_field_trial_config(config)
            for change in ({'yaw_delta_rad': .5}, {'final_pwm': [30, 30, -30, -30]},
                           {'status': 'failed'}, {'errors': ['failed']}, {'artifacts_sha256': {}}):
                path.write_text(json.dumps(dict(summary, **change)))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    onboard.validate_field_trial_config(config)
            path.write_text(json.dumps(summary)); (root/'uart_rx.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                onboard.validate_field_trial_config(config)

    def test_dryrun_main_records_adapter_inputs_without_devices(self):
        from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
        from car.test.test_yolopv2_primary import MaskModel
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'video.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (640,480))
            self.assertTrue(writer.isOpened())
            for _ in range(10): writer.write(np.zeros((480,640,3),np.uint8))
            writer.release()
            config = self.config()
            argv = ['onboard', '--video', str(source), '--max-samples', '6',
                    '--no-run-archive', '--output-dir', str(root/'output')]
            def detector(settings, **kwargs):
                return YOLOPv2SemanticDetector(settings, model=MaskModel(), **kwargs)
            with patch.object(sys, 'argv', argv), \
                 patch.object(onboard, 'load_config', return_value=(onboard.DEFAULT_CONFIG, config)), \
                 patch('autodrive.perception.yolopv2_semantic.YOLOPv2SemanticDetector', side_effect=detector), \
                 patch.object(onboard, 'create_chassis') as chassis, \
                 patch('cloud_client.client.CloudClient') as cloud, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(onboard.main(), 0)
            chassis.assert_not_called(); cloud.assert_not_called()
            with (root/'output/onboard_log.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertIn('stationary_phase', rows[0])
            self.assertEqual(rows[0]['imu_valid'], 'False')
            self.assertEqual(rows[0]['imu_yaw_rad'], '')
            self.assertTrue(float(rows[0]['control_monotonic']) > 0)
            status = json.loads((root/'output/status.json').read_text())
            self.assertTrue(status['termination']['stopped'])
            self.assertIn('stationary_corner', status)
            self.assertEqual(status['stationary_corner']['imu_valid'], False)


if __name__ == '__main__':
    unittest.main()
