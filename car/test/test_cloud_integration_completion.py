"""Small hardware-free regressions for deployment integration gaps."""
import json
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from autodrive.web.dashboard import DashboardState
from autodrive.control import recovery_worker
from autodrive.control.cloud_recovery import RecoveryConfig, RecoveryEngine, stopped


class CloudIntegrationTests(unittest.TestCase):
    def test_dashboard_reads_uuid_recovery_and_keeps_provider_secrets_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'runs'/'r1'
            event='12345678-1234-4234-8234-123456789abc'
            folder=run/'cloud'/event;folder.mkdir(parents=True)
            (root/'status.json').write_text(json.dumps(dict(run_archive=str(run),
                cloud_arbitration=dict(enabled=True,phase='PROBE_STEP',event_id=event,
                    reason='current candidate',selected_candidate_id='forward-white',
                    fusion=dict(source='cloud_recovery',candidate_id='forward-white',
                        policy_id='rule-recovery-v1',reason='current candidate',secret='SECRET')))))
            (folder/'request.json').write_text(json.dumps(dict(context=dict(event_id=event,stop_code='perception_stall'))))
            (folder/'response.json').write_text(json.dumps(dict(event_id=event,
                result=dict(scene=dict(schema_version='road-recovery-v1',assessment='white_marking',
                    recommendation='try_candidate',candidate_id='forward-white',uncertain=False,
                    hazards=[],reason='white crossing',raw_response='SECRET')))))
            cloud=DashboardState(root,root/'sensors').get_state()['cloud']
            self.assertEqual(cloud['status'],'probing')
            self.assertEqual(cloud['result']['candidate_id'],'forward-white')
            self.assertEqual(cloud['fusion']['source'],'cloud_recovery')
            self.assertNotIn('SECRET',json.dumps(cloud))

    def test_journal_does_not_block_control_and_drains_in_order(self):
        started=threading.Event();release=threading.Event()
        journal_class=getattr(recovery_worker,'AsyncRecoveryJournal',None)
        self.assertIsNotNone(journal_class)
        original=journal_class._write_record
        def slow(writer,*args):
            started.set();release.wait(2)
            return original(writer,*args)
        with tempfile.TemporaryDirectory() as tmp, patch.object(journal_class,'_write_record',slow):
            writer=journal_class(Path(tmp),queue_records=2)
            try:
                self.assertTrue(writer.submit('decisions.jsonl',dict(index=1)))
                self.assertTrue(started.wait(1))
                self.assertTrue(writer.submit('decisions.jsonl',dict(index=2)))
                self.assertTrue(writer.submit('applications.jsonl',dict(index=3)))
                self.assertFalse(writer.submit('decisions.jsonl',dict(index=4)))
                self.assertIsNotNone(writer.get_state()['error'])
            finally:release.set();writer.close()
            self.assertEqual([json.loads(x)['index'] for x in (Path(tmp)/'decisions.jsonl').read_text().splitlines()],[1,2])
            self.assertEqual(writer.get_state()['written'],3)

    def test_recorded_recovery_uses_recorded_reply_time_and_current_geometry(self):
        from autodrive.tools.recorded_recovery import RecordedRecoveryCoordinator
        config=RecoveryConfig(enabled=True)
        def evidence(now,seq):
            mask=np.ones((240,320),np.uint8)
            return dict(hard_safe=True,stop_code='perception_stall',
                observation=dict(sequence=seq,captured_at=now-.01,mask=mask,
                    frame=np.full((480,640,3),100,np.uint8)),
                candidates=[dict(candidate_id='forward-white',action='forward',assessment='white_marking')],
                imu=dict(yaw_rad=0.,imu_timestamp=now,imu_valid=True))
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);cloud=run/'cloud';cloud.mkdir()
            engine=RecoveryEngine(config,run.name)
            for seq,now in enumerate((1.,1.5,2.1),1):engine.tick(stopped('local'),evidence(now,seq),now)
            context=engine.request['context'];event=context['event_id'];folder=cloud/event;folder.mkdir()
            scene=dict(schema_version='road-recovery-v1',assessment='white_marking',
                recommendation='try_candidate',candidate_id='forward-white',uncertain=False,hazards=[],reason='white')
            manifest=[]
            for index,frame in enumerate(engine.request['frames']):
                path=folder/('input_%02d.png'%index);self.assertTrue(cv2.imwrite(str(path),frame))
                manifest.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            (folder/'request.json').write_text(json.dumps(dict(context=context,inputs=manifest)))
            (folder/'response.json').write_text(json.dumps(dict(event_id=event,worker_finished=2.19,result=dict(scene=scene))))
            (cloud/'decisions.jsonl').write_text(json.dumps(dict(now=2.3,received_advice=[dict(event_id=event,scene=scene,error=None,received_at=2.3)]))+'\n')
            replay=RecordedRecoveryCoordinator(config,run)
            for seq,now in enumerate((1.,1.5,2.1),1):self.assertEqual(replay.filter(stopped('local'),evidence(now,seq),now).action,'stop')
            self.assertEqual(replay.filter(stopped('local'),evidence(2.2,4),2.2).action,'stop')
            self.assertEqual(replay.filter(stopped('local'),evidence(2.3,5),2.3).action,'forward')
            self.assertEqual(replay.filter(stopped('local'),evidence(2.46,6),2.46).action,'stop')
            self.assertEqual(replay.get_state()['recorded_replay']['matched_events'],1)
            veto=RecordedRecoveryCoordinator(config,run)
            for seq,now in enumerate((1.,1.5,2.1),1):veto.filter(stopped('local'),evidence(now,seq),now)
            changed=evidence(2.3,4);changed['candidates']=[]
            self.assertEqual(veto.filter(stopped('local'),changed,2.3).action,'stop')
