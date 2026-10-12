"""Catalog preserved evidence without importing the vehicle runtime.

python3 -m experiments.catalog --root /path/to/repo --output /new/catalog
Does not follow symlinks, infer missing commits, alter runs, or read secrets.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def _relative(path, root):
    return path.relative_to(root).as_posix()


def _safe(path, root):
    """Reject symlinks at every path component, including intermediate folders."""
    try:
        path.relative_to(root)
        path.resolve().relative_to(root)
    except (ValueError, OSError, RuntimeError):
        return False
    return not any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents)


def _read(path, root, issues):
    if not _safe(path, root):
        issues.append({'path': _relative(path, root), 'reason': 'symlink not followed'})
        return {}
    if not path.is_file():
        return {}
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError('metadata size exceeds 8 MiB')
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            raise ValueError('expected object')
        return value
    except (OSError, ValueError) as exc:
        issues.append({'path': _relative(path, root), 'reason': type(exc).__name__})
        return {}


def _children(folder, root, issues):
    if not _safe(folder, root):
        issues.append({'path': _relative(folder, root), 'reason': 'symlink not followed'})
        return []
    if not folder.is_dir():
        return []
    result = []
    for path in sorted(folder.iterdir()):
        if path.is_symlink():
            issues.append({'path': _relative(path, root), 'reason': 'symlink not followed'})
        elif path.is_dir():
            result.append(path)
    return result


def scan_runs(root):
    root = Path(root).resolve()
    issues, sessions, runs, seen = [], [], [], set()
    outputs = root / 'outputs'
    if outputs.is_symlink():
        return {'schema_version': 'vehicle-catalog-v1', 'sessions': [], 'runs': [],
                'sensor_recording_count': 0, 'matched_sensor_recordings': 0, 'unmatched_sensor_recordings': [],
                'issues': [{'path': 'outputs', 'reason': 'symlink not followed'}]}
    sensor_root = outputs / 'vehicle_dashboard/sensors/recordings'
    sensors = {p.name: p for p in _children(sensor_root, root, issues)}
    matched_sensors = set()

    def add_run(path, session_id=None):
        if path.is_symlink() or path in seen:
            return
        seen.add(path)
        status = _read(path / 'status.json', root, issues)
        provenance = _read(path / 'provenance.json', root, issues)
        events = []
        for event in _children(path / 'cloud', root, issues):
            # Do not copy model text or authentication metadata into the registry.
            events.append({'event_id': event.name, 'path': _relative(event, root),
                           'request_present': (event / 'request.json').is_file() and not (event / 'request.json').is_symlink(),
                           'response_present': (event / 'response.json').is_file() and not (event / 'response.json').is_symlink()})
        sensor = sensors.get(path.name)
        sensor_state = _read(sensor / 'status.json', root, issues) if sensor else {}
        if sensor:
            matched_sensors.add(sensor.name)
        termination = status.get('termination') if isinstance(status.get('termination'), dict) else {}
        git = provenance.get('git') if isinstance(provenance.get('git'), dict) else {}
        runs.append({'run_id': path.name, 'path': _relative(path, root), 'session_id': session_id,
                     'source_commit': git.get('commit'), 'source_tree_sha256': provenance.get('source_tree_sha256'),
                     'stopped_recorded': termination.get('stopped'),
                     'sensor_recording': _relative(sensor, root) if sensor else None,
                     'sensor_closed_recorded': sensor_state.get('closed'),
                     'config_present': _safe(path / 'resolved_runtime_config.yaml', root) and (path / 'resolved_runtime_config.yaml').is_file(),
                     'cloud_events': events,
                     'integrity': 'not_verified',
                     'provenance_status': 'manifest_present' if provenance else 'historical_source_requires_mapping'})

    for session in _children(outputs / 'autodrive_sessions', root, issues):
        state = _read(session / 'session.json', root, issues)
        sessions.append({'session_id': session.name, 'path': _relative(session, root),
                         'state_recorded': state.get('state'), 'running_recorded': state.get('running')})
        for path in _children(session / 'runtime/runs', root, issues):
            add_run(path, session.name)
    # Other historical experiments also own runs, outside the panel session tree.
    for base, dirs, _ in os.walk(outputs, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in ('venv', 'downloads', 'semantic_inputs', 'autodrive_sessions', '__pycache__')
                         and not (Path(base) / d).is_symlink())
        if Path(base).name == 'runs':
            for path in _children(Path(base), root, issues):
                add_run(path)
            dirs[:] = []
    counts = defaultdict(list)
    for run in runs:
        counts[run['run_id']].append(run['path'])
    for name, paths in counts.items():
        if len(paths) > 1:
            issues.append({'path': name, 'reason': 'duplicate run ID; use path-qualified identity', 'paths': paths})
    return {'schema_version': 'vehicle-catalog-v1', 'sessions': sessions, 'runs': runs,
            'sensor_recording_count': len(sensors), 'matched_sensor_recordings': len(matched_sensors),
            'unmatched_sensor_recordings': sorted(set(sensors) - matched_sensors), 'issues': issues}


def scan_dataset(root):
    root = Path(root).resolve()
    dataset = root / 'car/longtail/dataset'
    groups, issues = defaultdict(list), []
    if not _safe(dataset, root):
        raise ValueError('dataset symlink is not followed')
    for base, dirs, files in os.walk(dataset, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
        for name in sorted(files):
            path = Path(base) / name
            if path.is_symlink():
                issues.append({'path': _relative(path, root), 'reason': 'symlink not followed'})
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(dataset)
            label = relative.parts[0] if relative.parts[0] in ('Long-tail', 'Non-long-tail') else None
            groups[digest].append({'path': _relative(path, root), 'label': label,
                                  'source_group': relative.parts[0], 'bytes': path.stat().st_size})
    samples = []
    for digest, aliases in sorted(groups.items()):
        labels = sorted({x['label'] for x in aliases if x['label'] is not None})
        samples.append({'sha256': digest, 'aliases': aliases, 'labels': labels,
                        'label_conflict': len(labels) > 1, 'split': None})
    return {'schema_version': 'vehicle-dataset-catalog-v1', 'files': sum(map(len, groups.values())),
            'unique_contents': len(groups), 'samples': samples, 'issues': issues}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='new output directory, never an existing run')
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    for protected in (root / 'outputs', root / 'car/longtail/dataset'):
        resolved_protected = protected.resolve()
        if (output == protected or protected in output.parents
                or output == resolved_protected or resolved_protected in output.parents):
            parser.error('catalog output must be outside original evidence directories')
    if output.exists():
        parser.error('output already exists; choose a new catalog directory')
    runs, dataset = scan_runs(root), scan_dataset(root)
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (('runs.json', runs), ('datasets.json', dataset)):
        payload['generated_utc'] = datetime.now(timezone.utc).isoformat()
        payload['source_root'] = str(root)
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'sessions': len(runs['sessions']), 'runs': len(runs['runs']),
                      'sensor_matches': runs['matched_sensor_recordings'],
                      'dataset_files': dataset['files'], 'issues': len(runs['issues']) + len(dataset['issues'])}))


if __name__ == '__main__':
    main()
