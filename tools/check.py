#!/usr/bin/env python3
"""Repository checks. Default tests use fake devices; loopback is explicit.

Use the project's existing interpreter (see ./vehicle check). Never discovers
hardware tools, legacy scripts, outputs or model benchmarks.
"""
import argparse
import ast
import importlib.util
import hashlib
import json
import re
from urllib.parse import unquote, urlsplit
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / 'car', ROOT / 'car/control'):
    sys.path.insert(0, str(folder))
LOOPBACK = {'test_cloud_scene_http', 'test_vehicle_dashboard'}
WEBSOCKET = {'test_cloud_scene_ws'}
TORCH = {'test_yolopv2_torch_input'}


def documentation_checks():
    # Only maintained pages: immutable archives keep their original link context.
    pages = [ROOT / name for name in (
        'README.md', 'PROJECT_INDEX.md', 'experiments/README.md',
        'requirements/README.md', 'research/legacy/README.md',
        'car/autodrive/README.md', 'car/test/README.md', 'car/cloud_client/README.md',
        'car/autodrive/ON_CAR_CHECKLIST.md', 'car/control/README.md',
        'docs/outer-loop-record-replay.md', 'docs/vehicle-dashboard.md',
        'docs/history/README.md', 'docs/jetson-yolopv2.md', 'docs/visual-feedback-replay.md',
        'coordination/2026-10-12-repository-refactor.md',
        'coordination/2026-10-12-entrypoint-consolidation.md', 'AGENTS.md',
        '.agents/skills/vehicle-cloud-thesis/SKILL.md',
        '.agents/skills/thesis-handoff/SKILL.md')]
    for folder in ('docs/architecture', 'docs/operations'):
        pages.extend((ROOT / folder).glob('*.md'))
    for page in pages:
        content = re.sub(r'```.*?```', '', page.read_text(encoding='utf-8'), flags=re.S)
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', content):
            target = target.strip().strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith('#'):
                continue
            path = unquote(parsed.path)
            if path and not (page.parent / path).exists():
                raise ValueError('broken local link: {} -> {}'.format(page.relative_to(ROOT), target))
    migrations = json.loads((ROOT / 'coordination/refactor-2026-10-12/migrations.json').read_text())
    for record in migrations:
        target = ROOT / record['new_path']
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('archived original changed: ' + record['new_path'])
    print('Documentation: {} maintained pages; {} archive hashes'.format(len(pages), len(migrations)), flush=True)


def static_checks():
    documentation_checks()
    from tools.repository_layout import check_layout
    check_layout(ROOT)
    files = []
    for folder in ('car', 'experiments', 'tools', 'tests'):
        files.extend((ROOT / folder).rglob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    scripts = [ROOT / 'vehicle'] + list(ROOT.glob('*.sh')) + list((ROOT / 'deploy').glob('*.sh'))
    for path in scripts:
        subprocess.run(['bash', '-n', str(path)], check=True)
    print('Static checks: {} Python files, {} shell scripts'.format(len(files), len(scripts)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static', action='store_true', help='syntax, maintained document links and archive hashes; no runtime imports')
    parser.add_argument('--loopback', action='store_true', help='include local HTTP socket integration tests')
    parser.add_argument('--websocket-loopback', action='store_true', help='requires optional websockets test dependency')
    parser.add_argument('--torch', action='store_true', help='CPU tensor tests; no model load or GPU inference')
    args = parser.parse_args(argv)
    static_checks()
    if args.static:
        return 0
    required = ['numpy', 'cv2', 'yaml', 'websocket', 'torch']
    if args.websocket_loopback:
        required.append('websockets')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.error('missing test dependencies: ' + ', '.join(missing) + '; use existing project environment; no automatic installation')
    modules = []
    excluded = []
    for path in sorted((ROOT / 'car/test').glob('test_*.py')):
        name = path.stem
        if ((name in LOOPBACK and not args.loopback)
                or (name in WEBSOCKET and not args.websocket_loopback)
                or (name in TORCH and not args.torch)):
            excluded.append(name)
        else:
            modules.append('car.test.' + name)
    print('Explicitly excluded optional groups: ' + ', '.join(excluded), flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py'))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
