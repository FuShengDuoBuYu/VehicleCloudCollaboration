"""Compare all frame results, excluding only nondeterministic UUID identities."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('before', type=Path)
parser.add_argument('after', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()

def canonical(value):
    if isinstance(value, dict):
        return {k:canonical(v) for k,v in value.items()
                if k not in ('event_id', 'last_event_id', 'recorded_processing_budget_valid')}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value

before = json.loads((args.before / 'commands.json').read_text())
after = json.loads((args.after / 'commands.json').read_text())
left, right = canonical(before), canonical(after)
assert len(left) == len(right)
differences = [i for i,(a,b) in enumerate(zip(left,right)) if a != b]
a = json.loads((args.before / 'summary.json').read_text())
b = json.loads((args.after / 'summary.json').read_text())
inputs_equal = (a['csv_sha256'] == b['csv_sha256']
                and a['semantic_input_sha256'] == b['semantic_input_sha256']
                and a['configuration'] == b['configuration'])
result = dict(frames=len(left), different_frames=differences, inputs_equal=inputs_equal,
    csv_sha256=a['csv_sha256'], semantic_input_count=len(a['semantic_input_sha256']),
    processing_budget_unknown_rows=b.get('processing_budget_unknown_rows'),
    before_commands_sha256=hashlib.sha256((args.before/'commands.json').read_bytes()).hexdigest(),
    after_commands_sha256=hashlib.sha256((args.after/'commands.json').read_bytes()).hexdigest(),
    canonical_sha256=hashlib.sha256(json.dumps(left,sort_keys=True).encode()).hexdigest(),
    comparison='all frame results except random event UUIDs and new processing-budget metadata',
    timing='baseline uses archived gate via replay_baseline.py; refactor uses recorded budget natively',
    limitation='No model inference, physical movement, network or actuator timing validation')
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert inputs_equal and not differences, 'replay behavior differs'
