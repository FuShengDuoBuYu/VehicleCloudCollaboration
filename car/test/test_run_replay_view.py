import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    replay=importlib.import_module('autodrive.web.run_replay')
except ModuleNotFoundError as exc:
    if exc.name!='autodrive.web.run_replay':raise
    replay=None


class RunReplayTests(unittest.TestCase):
    def test_only_completed_session_archive_can_be_replayed_and_paths_are_confined(self):
        self.assertIsNotNone(replay,'session replay store missing')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);sid='a'*32;folder=root/sid;run=folder/'runtime/runs/run1';run.mkdir(parents=True)
            (folder/'session.json').write_text(json.dumps({'running':False,'run_archive':str(run)}))
            (run/'onboard_log.csv').write_text('sample,timestamp_s,control_monotonic,action,encoder_1\n0,0.0,10.0,forward,42\n1,0.1,10.1,stop,43\n')
            store=replay.RunReplayStore(root,root/'sensors')
            result=store.data(sid)
            self.assertEqual(result['frames'][1]['encoder_1'],43.)
            self.assertEqual(result['frames'][1]['action'],'stop')
            self.assertEqual(result['frames'][1]['timestamp_s'],.1)
            with self.assertRaises(ValueError):store.data('../'+sid)
            (folder/'session.json').write_text(json.dumps({'running':True,'run_archive':str(run)}))
            with self.assertRaises(ValueError):store.data(sid)
            (folder/'session.json').write_text(json.dumps({'running':False,'run_archive':str(root)}))
            with self.assertRaises(ValueError):store.data(sid)


if __name__=='__main__':unittest.main()
