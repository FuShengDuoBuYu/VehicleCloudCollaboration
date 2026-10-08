import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys
import tempfile
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class WorkerTests(unittest.TestCase):
    def test_submit_never_waits_for_network_and_keeps_only_latest_pending(self):
        from cloud_client import LatestSceneWorker, ImageFrame
        entered=threading.Event();release=threading.Event();finished=threading.Event();calls=[]
        class Client:
            def request_scene(self,frame,context):
                calls.append(context['event_id'])
                if len(calls)==1: entered.set();release.wait(3)
                else: finished.set()
                return SimpleNamespace(context=context,request_id=context['event_id'])
        worker=LatestSceneWorker(Client());self.addCleanup(lambda:worker.close(wait=True));self.addCleanup(release.set)
        frame=ImageFrame(b'image-bytes','frame.jpg')
        worker.submit(frame,{'event_id':'one','frame_id':'f1','candidate_version':'v1'})
        self.assertTrue(entered.wait(1))
        start=time.monotonic();worker.submit(frame,{'event_id':'two'});worker.submit(frame,{'event_id':'three'})
        self.assertLess(time.monotonic()-start,.1)
        release.set();self.assertTrue(finished.wait(1))
        deadline=time.monotonic()+1;outcome=None
        while time.monotonic()<deadline and outcome is None:
            outcome=worker.poll();threading.Event().wait(.005)
        self.assertEqual(calls,['one','three']);self.assertEqual(outcome.result.context['event_id'],'three')
        self.assertEqual(worker.counters['superseded_pending'],1);self.assertEqual(worker.counters['discarded_result'],1)
    def test_input_context_snapshot_and_failure_outcome_safe(self):
        from cloud_client import LatestSceneWorker, ImageFrame
        entered=threading.Event();release=threading.Event()
        class Client:
            def request_scene(self,frame,context):
                entered.set();release.wait(2)
                raise RuntimeError('private provider diagnostics')
        worker=LatestSceneWorker(Client());self.addCleanup(lambda:worker.close(wait=True));self.addCleanup(release.set)
        context={'event_id':'e','nested':{'value':1}}
        worker.submit(ImageFrame(b'x','x.jpg'),context);self.assertTrue(entered.wait(1));context['nested']['value']=9
        release.set();deadline=time.monotonic()+1;outcome=None
        while time.monotonic()<deadline and outcome is None:
            outcome=worker.poll();threading.Event().wait(.005)
        self.assertEqual(outcome.status,'failed');self.assertNotIn('private',outcome.error)
        self.assertEqual(outcome.context['nested']['value'],1)
    def test_reject_mutable_frame_and_use_after_close(self):
        from cloud_client import LatestSceneWorker, ImageFrame
        with self.assertRaises(ValueError): ImageFrame(bytearray(b'x'),'x.jpg')
        worker=LatestSceneWorker();worker.close()
        with self.assertRaises(RuntimeError): worker.submit(ImageFrame(b'x','x.jpg'))
    def test_poll_discards_each_mismatched_identity(self):
        from cloud_client import LatestSceneWorker, ImageFrame
        class Client:
            def request_scene(self,frame,context):
                return SimpleNamespace(context=context)
        identity={'event_id':'e','frame_id':'f','candidate_version':'v'}
        for field in identity:
            with self.subTest(field=field):
                worker=LatestSceneWorker(Client())
                try:
                    worker.submit(ImageFrame(b'x'),identity)
                    deadline=time.monotonic()+1
                    while time.monotonic()<deadline and not worker.counters.get('completed'):
                        threading.Event().wait(.005)
                    self.assertEqual(worker.counters.get('completed'),1)
                    expected=dict(identity);expected[field]='different'
                    self.assertIsNone(worker.poll(**expected))
                    self.assertEqual(worker.counters['discarded_result'],1)
                    self.assertIsNone(worker.poll(**identity))
                    self.assertIn('discarded_context_mismatch',[r['status'] for r in worker.drain_records()])
                finally:
                    worker.close(wait=True)
    def test_evidence_includes_obsolete_attempt_without_exposing_key(self):
        from cloud_client import LatestSceneWorker, ImageFrame, CloudSceneResult
        entered=threading.Event();release=threading.Event();calls=[]
        class Client:
            config=SimpleNamespace(api_key='test-private-key')
            def request_scene(self,frame,context):
                calls.append(context['event_id'])
                if len(calls)==1: entered.set();release.wait(2)
                return CloudSceneResult({'reason':'test-private-key'},'fake',{},'r',{})
        with tempfile.TemporaryDirectory() as folder:
            worker=LatestSceneWorker(Client(),evidence_dir=folder)
            try:
                worker.submit(ImageFrame(b'x'),{'event_id':'first'})
                self.assertTrue(entered.wait(1));worker.submit(ImageFrame(b'x'),{'event_id':'second'})
                release.set();deadline=time.monotonic()+1
                while time.monotonic()<deadline and worker.counters.get('completed',0)<1: threading.Event().wait(.01)
            finally: release.set();worker.close(wait=True)
            paths=list(Path(folder).glob('*.json'));self.assertEqual(len(paths),2)
            for path in paths: self.assertNotIn('test-private-key',path.read_text(encoding='utf-8'))


if __name__=='__main__': unittest.main()
