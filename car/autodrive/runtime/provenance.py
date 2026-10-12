"""Versioned, secret-free source manifests for new runs (stdlib only).

Called during initialization, before the driver is constructed. Existing runs
are never rewritten. A source ZIP preserves uncommitted implementation bytes.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile
from datetime import datetime, timezone
from importlib import metadata

SOURCE_ROOTS = ('car/autodrive', 'car/control', 'car/cloud_client', 'car/longtail/detectors', 'tools')
SOURCE_SUFFIXES = {'.py', '.yaml', '.yml', '.html', '.js', '.css', '.sh'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_files(root):
    root = Path(root).resolve()
    for name in SOURCE_ROOTS:
        folder = root / name
        if (folder.is_symlink() or not folder.is_dir()
                or any(p.is_symlink() for p in folder.parents if p != root and root in p.parents)):
            continue
        for base, dirs, files in os.walk(folder, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__' and not (Path(base) / d).is_symlink())
            for name in sorted(files):
                path = Path(base) / name
                if not path.is_symlink() and path.suffix in SOURCE_SUFFIXES:
                    yield path
    for path in [root / 'vehicle', *sorted(root.glob('run*.sh'))]:
        if path.is_file() and not path.is_symlink():
            yield path


def _git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args],
                                       stderr=subprocess.DEVNULL, timeout=3).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_provenance(root):
    root = Path(root).resolve()
    hashes = {p.relative_to(root).as_posix(): file_sha256(p) for p in source_files(root)}
    status = _git(root, 'status', '--porcelain', '--untracked-files=normal')
    versions = {}
    for name in ('numpy', 'opencv-python', 'PyYAML', 'torch', 'websocket-client', 'pyserial',
                 'Pillow', 'smbus2', 'onnxruntime'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'schema_version': 'vehicle-source-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'git': {'commit': _git(root, 'rev-parse', 'HEAD'),
                'dirty': None if status is None else bool(status)},
        'source_sha256': hashes,
        'source_tree_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        'runtime': {'python': platform.python_version(), 'platform': platform.machine(),
                    'packages': versions},
        'scope': 'allowlisted implementation/configuration files; no environment values or secrets',
    }


def write_run_provenance(run_dir, root, model_path=None, *, model_paths=None):
    run_dir, root = Path(run_dir), Path(root).resolve()
    manifest = source_provenance(root)
    manifest['run_id'] = run_dir.name
    models = {'primary': model_path, **(model_paths or {})}
    manifest['models'] = {}
    for role, value in models.items():
        model = Path(value).expanduser() if value else None
        if model is not None and not model.is_absolute():
            model = root / model
        manifest['models'][role] = {
            'path': str(model) if model else None,
            'sha256': file_sha256(model) if model and model.is_file() else None}
    manifest['model'] = manifest['models']['primary']  # v1 consumer compatibility
    # Exclusive creation prevents accidental replacement of any historical run.
    archive = run_dir / 'source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, expected in manifest['source_sha256'].items():
            content = (root / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError('source changed during run initialization: ' + name)
            bundle.writestr(name, content)
    manifest['source_archive_sha256'] = file_sha256(archive)
    manifest['snapshot_sha256'] = {p.name: file_sha256(p) for p in sorted(run_dir.glob('*.yaml'))}
    with (run_dir / 'provenance.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return manifest
