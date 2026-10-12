"""Evidence metadata must remain truthful and never modify input runs."""
import json
import tempfile
import unittest
from pathlib import Path
from experiments.catalog import scan_runs, scan_dataset
from autodrive.runtime.provenance import source_provenance


class EvidenceTests(unittest.TestCase):
    def test_run_links_sensor_and_cloud_records_without_rewriting_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=root/'outputs/autodrive_sessions/s1';run=session/'runtime/runs/r1'
            event=run/'cloud/e1';event.mkdir(parents=True)
            (session/'session.json').write_text(json.dumps({'state':'finished','running':False}))
            (run/'status.json').write_text(json.dumps({'termination':{'stopped':True}}))
            (event/'request.json').write_text('{}')
            sensor=root/'outputs/vehicle_dashboard/sensors/recordings/r1';sensor.mkdir(parents=True)
            (sensor/'status.json').write_text('{broken')
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result=scan_runs(root)
            row=next(x for x in result['runs'] if x['run_id']=='r1')
            self.assertEqual(row['session_id'],'s1')
            self.assertEqual(row['cloud_events'][0]['response_present'],False)
            self.assertEqual(row['sensor_recording'],'outputs/vehicle_dashboard/sensors/recordings/r1')
            self.assertIsNone(row['source_commit'])
            self.assertTrue(any('sensor' in x['path'] for x in result['issues']))
            self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_broken_session_and_external_symlink_are_reported_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root=Path(tmp);base=root/'outputs/autodrive_sessions';base.mkdir(parents=True)
            bad=base/'broken';bad.mkdir();(bad/'session.json').write_text('[]')
            (base/'outside').symlink_to(other,target_is_directory=True)
            result=scan_runs(root)
            self.assertTrue(result['issues'])
            self.assertFalse(any(x.get('session_id')=='outside' for x in result['runs']))

    def test_intermediate_symlink_cannot_escape_evidence_root(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root=Path(tmp);outside=Path(other)
            (outside/'runs/stolen').mkdir(parents=True)
            (outside/'runs/stolen/status.json').write_text('{}')
            session=root/'outputs/autodrive_sessions/s1';session.mkdir(parents=True)
            (session/'runtime').symlink_to(outside,target_is_directory=True)
            result=scan_runs(root)
            self.assertEqual(result['runs'],[])
            self.assertTrue(result['issues'])

    def test_run_source_zip_matches_manifest_and_refuses_overwrite(self):
        import zipfile,hashlib
        from autodrive.runtime.provenance import write_run_provenance
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'car/autodrive/core.py';source.parent.mkdir(parents=True)
            source.write_text('original=True\n');run=root/'run';run.mkdir()
            (run/'resolved_runtime_config.yaml').write_text('version: 1\n')
            model=root/'model.pt';model.write_bytes(b'model')
            manifest=write_run_provenance(run,root,model)
            with zipfile.ZipFile(run/'source.zip') as bundle:
                self.assertEqual(bundle.read('car/autodrive/core.py'),source.read_bytes())
            self.assertEqual(manifest['model']['sha256'],hashlib.sha256(b'model').hexdigest())
            self.assertIn('resolved_runtime_config.yaml',manifest['snapshot_sha256'])
            with self.assertRaises(FileExistsError):write_run_provenance(run,root,model)

    def test_dataset_preserves_all_label_and_source_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ('Long-tail','Non-long-tail','CODA/images'):
                p=root/'car/longtail/dataset'/folder/'same.jpg';p.parent.mkdir(parents=True);p.write_bytes(b'original')
            result=scan_dataset(root)
            self.assertEqual(result['files'],3);self.assertEqual(result['unique_contents'],1)
            self.assertEqual(len(result['samples'][0]['aliases']),3)
            self.assertTrue(result['samples'][0]['label_conflict'])

    def test_provenance_hashes_uncommitted_sources_without_environment_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'car/autodrive/feature.py';p.parent.mkdir(parents=True);p.write_text('value=1\n')
            (root/'.env').write_text('TOKEN=DO_NOT_COPY')
            first=source_provenance(root)
            self.assertIsNone(first['git']['commit'])
            self.assertNotIn('DO_NOT_COPY',json.dumps(first))
            self.assertNotIn('.env',first['source_sha256'])
            p.write_text('value=2\n');second=source_provenance(root)
            self.assertNotEqual(first['source_tree_sha256'],second['source_tree_sha256'])

    def test_adaptive_models_are_both_hashed_and_tilde_is_expanded(self):
        from unittest.mock import patch
        import hashlib
        from autodrive.runtime.provenance import write_run_provenance
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            (root/'primary.onnx').write_bytes(b'fp32')
            (root/'int8.onnx').write_bytes(b'int8')
            with patch('pathlib.Path.expanduser', autospec=True, side_effect=lambda p: root/'int8.onnx' if str(p)=='~/int8.onnx' else p):
                result=write_run_provenance(run,root,'primary.onnx',
                    model_paths={'int8':'~/int8.onnx'})
            self.assertEqual(result['models']['primary']['sha256'],hashlib.sha256(b'fp32').hexdigest())
            self.assertEqual(result['models']['int8']['sha256'],hashlib.sha256(b'int8').hexdigest())

    def test_catalog_output_cannot_enter_symlinked_evidence_storage(self):
        from experiments.catalog import main
        from contextlib import redirect_stderr
        import io
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root=Path(tmp);outside=Path(other)
            (root/'outputs').symlink_to(outside,target_is_directory=True)
            for destination in (root/'outputs/catalog',outside/'catalog'):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['--root',str(root),'--output',str(destination)])
                self.assertFalse((outside/'catalog').exists())

    def test_source_snapshot_includes_public_launcher_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'tools').mkdir()
            (root/'vehicle').write_text('#!/bin/sh\n')
            (root/'tools/process_lifecycle.py').write_text('value=1\n')
            manifest=source_provenance(root)
            self.assertIn('vehicle',manifest['source_sha256'])
            self.assertIn('tools/process_lifecycle.py',manifest['source_sha256'])
