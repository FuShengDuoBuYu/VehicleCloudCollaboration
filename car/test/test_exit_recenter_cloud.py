"""Focused regressions from run 05a3; no hardware or network."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from car.test.test_cloud_recovery import evidence, STOP
from car.test.test_cloud_recovery_contract import recovery
from autodrive.control.cloud_recovery import RecoveryConfig, RecoveryEngine
from autodrive.perception.semantic_outer_route import SemanticOuterRoute
from autodrive.web.dashboard import DashboardState


class ExitCloudTests(unittest.TestCase):
    def sample(self, now, seq):
        e = evidence(now, seq)
        e['scene_support'] = e['observation']['mask'].copy()
        e['candidates'][0].update(margin_px=22.,heading_error=.2)
        return e

    def pending(self):
        engine = RecoveryEngine(RecoveryConfig(enabled=True), 'test')
        for seq, now in enumerate((1., 1.5, 2.1), 1):
            engine.tick(STOP, self.sample(now, seq), now)
        return engine

    def test_sloping_edge_does_not_override_recenter_with_corner_approach(self):
        route = SemanticOuterRoute([[.5, .7], [1., .7]], stationary_corners=True,
                                   visual_feedback=True)
        road = np.zeros((240, 320), np.uint8)
        for x in range(320):
            road[int(125 + x * .23):, x] = 1
        self.assertIsNone(route._front_corner_preview(road))
        self.assertEqual(route.corner_reject_reason, 'front-spread-too-large')
        self.assertFalse(route.front_observed)

    def test_mask_dropout_with_same_current_floor_and_candidates_keeps_request(self):
        engine = self.pending(); event = engine.request['context']['event_id']
        e = self.sample(2.3, 4); e['observation']['mask'][:] = 0
        e['candidates'][0].update(margin_px=21.,heading_error=.20625)
        self.assertEqual(engine.tick(STOP, e, 2.3).action, 'stop')
        self.assertEqual(engine.phase, 'CLOUD_PENDING')
        engine.receive(event, recovery(), 2.4)
        e = self.sample(2.5, 5); e['observation']['mask'][:] = 0
        self.assertEqual(engine.tick(STOP, e, 2.5).action, 'forward')

    def test_real_change_still_cancels_and_late_reply_cannot_drive(self):
        for changed in ('frame', 'floor', 'candidate', 'geometry', 'imu', 'hazard'):
            with self.subTest(changed=changed):
                engine = self.pending(); event = engine.request['context']['event_id']
                e = self.sample(2.3, 4); e['observation']['mask'][:] = 0
                if changed == 'frame': e['observation']['frame'][:] = 200
                if changed == 'floor': e['scene_support'][:] = 0
                if changed == 'candidate': e['candidates'] = []
                if changed == 'geometry': e['candidates'][0]['heading_error'] += .2
                if changed == 'imu': e['imu']['yaw_rad'] = .2
                if changed == 'hazard': e['hard_safe'] = False
                self.assertEqual(engine.tick(STOP, e, 2.3).action, 'stop')
                self.assertIsNone(engine.request)
                self.assertEqual(engine.get_state()['last_event_id'], event)
                self.assertEqual(engine.get_state()['response_status'], 'cancelled')
                engine.receive(event, recovery(), 2.4)
                self.assertNotEqual(engine.phase, 'REVALIDATE')

    def test_dashboard_preserves_cancelled_reply_without_active_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'runs'/'r1'
            event = '12345678-1234-4234-8234-123456789abc'
            folder = run/'cloud'/event; folder.mkdir(parents=True)
            (root/'status.json').write_text(json.dumps(dict(run_archive=str(run),
                cloud_arbitration=dict(enabled=True, phase='HOLD', event_id=None,
                    last_event_id=event, response_status='cancelled', last_event_reason='scene changed'))))
            (folder/'request.json').write_text(json.dumps(dict(context=dict(event_id=event))))
            (folder/'response.json').write_text(json.dumps(dict(event_id=event,
                result=dict(scene=recovery()), metadata=dict(secret='SECRET'))))
            cloud = DashboardState(root, root/'sensors').get_state()['cloud']
            self.assertIsNone(cloud['event_id'])
            self.assertEqual(cloud['last_event_id'], event)
            self.assertEqual(cloud['result']['candidate_id'], 'forward-white')
            self.assertEqual(cloud['arbitration']['response_status'], 'received_ignored')
            self.assertEqual(cloud['arbitration']['last_event_reason'], 'scene changed')
            self.assertNotIn('SECRET', json.dumps(cloud))
