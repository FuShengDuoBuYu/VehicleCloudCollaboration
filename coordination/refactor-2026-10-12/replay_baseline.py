"""Reproduce pre-refactor behavior with the original recorded processing gate.

Analysis-only harness: no hardware/model/cloud; writes to an explicit new output.
The old replay measured the analyst machine. Replace ONLY analyze's four timer
reads with the archived allowance so CPU load cannot change this comparison.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-tree', required=True, type=Path)
parser.add_argument('--run', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
sys.path[:0] = [str(args.source_tree / 'car'), str(args.source_tree / 'car/control')]
from autodrive.tools import replay_outer_loop as replay
from autodrive.runtime import onboard
with (args.run / 'onboard_log.csv').open() as stream:
    rows = iter(list(csv.DictReader(stream)))
original = replay.analyze


def recorded_clock_analyze(*arguments, **kwargs):
    row = next(rows)
    allowance = row['stationary_perception_budget_valid']
    if allowance not in ('True', 'False'):
        raise ValueError('comparison requires an explicit archived budget flag')
    elapsed = 0. if allowance == 'True' else float(arguments[7]) + 1.
    reads = iter((0., elapsed, elapsed, elapsed))
    with patch.object(onboard, 'time', SimpleNamespace(monotonic=lambda: next(reads))):
        return original(*arguments, **kwargs)


replay.analyze = recorded_clock_analyze
print(json.dumps(replay.replay_recorded(args.run, args.output, profile='recorded'), indent=2))
