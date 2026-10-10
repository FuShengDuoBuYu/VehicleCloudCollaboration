#!/usr/bin/env python3
"""Replay a finished session offline and index its raw sensor evidence.

No camera, serial, ROS subscription, model inference, cloud or motors are
opened. Sources remain unchanged; derived output must be a new directory.
This replays observed inputs, not a counterfactual physical trajectory.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

CAR=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CAR))
from autodrive.tools.replay_outer_loop import replay_recorded, recorded_replay_config


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_sensor_journal(directory):
    root=Path(directory).resolve()
    try:state=json.loads((root/'status.json').read_text())
    except (OSError,ValueError):state={}
    events=[];counts=Counter();last={}
    try:
        with (root/'events.jsonl').open() as stream:
            for index,line in enumerate(stream):
                row=json.loads(line)
                if row.get('index')!=index:raise ValueError('sensor journal sequence gap')
                sensor=row.get('sensor')
                if sensor not in ('lidar','depth','uart_rx','uart_tx','telemetry'):
                    raise ValueError('unknown sensor journal stream')
                stamp=row.get('received_monotonic',row.get('sent_monotonic'))
                if type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp<0:
                    raise ValueError('invalid sensor timestamp')
                if stamp<last.get(sensor,0):raise ValueError('sensor time regression')
                if 'array_path' in row:
                    path=(root/row['array_path']).resolve()
                    if path.parent!=root or path.suffix!='.npy':
                        raise ValueError('sensor array path escaped archive')
                    if not path.is_file():raise ValueError('missing raw sensor array')
                row=dict(row,journal=str(root/'events.jsonl'),event_monotonic=stamp)
                events.append(row);counts[sensor]+=1;last[sensor]=stamp
    except FileNotFoundError:
        return [],dict(directory=str(root),complete=False,reason='raw journal missing',written={})
    complete=(state.get('closed') is True and state.get('dropped')==0
              and state.get('error') is None and state.get('written')==dict(counts))
    return events,dict(directory=str(root),complete=complete,written=dict(counts),
        dropped=state.get('dropped'),error=state.get('error'),closed=state.get('closed'))


def align_sensor_events(rows,events):
    grouped=defaultdict(list)
    for event in events:grouped[event['sensor']].append(event)
    for values in grouped.values():values.sort(key=lambda e:e['event_monotonic'])
    stamps={key:[event['event_monotonic'] for event in values] for key,values in grouped.items()}
    result=[]
    for row in rows:
        value=row.get('control_monotonic');when=None if value in (None,'') else float(value)
        if when is not None and (not math.isfinite(when) or when<0):
            raise ValueError('invalid control timestamp')
        matched={}
        if when is not None:
            for sensor,values in grouped.items():
                index=bisect_right(stamps[sensor],when)-1
                if index>=0:
                    event=values[index];age=when-event['event_monotonic']
                    matched[sensor]=dict(index=event['index'],journal=event['journal'],
                        age_s=age,fresh=age<=.5,array_path=event.get('array_path'),
                        source_stamp_s=event.get('source_stamp_s'))
        result.append(dict(sample=int(row['sample']),control_monotonic=when,sensors=matched))
    return result


def inventory(root):
    root=Path(root).resolve();result={}
    if not root.is_dir():return result
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        if root not in path.resolve().parents:raise ValueError('source inventory path escaped archive')
        result[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    return result


def replay_session(run,output,profile='recorded',allow_incomplete_sensors=False,expectations=None):
    run=Path(run).resolve();output=Path(output).resolve()
    if output==run or run in output.parents:raise ValueError('derived output must be outside the original run')
    status=json.loads((run/'status.json').read_text())
    if status.get('termination',{}).get('stopped') is not True:
        raise ValueError('session has not stopped')
    with (run/'onboard_log.csv').open() as stream:rows=list(csv.DictReader(stream))
    frozen,_=recorded_replay_config(run,'recorded')
    local=run/'sensors'
    ros_base=Path(frozen.get('runtime',{}).get('telemetry_dir','outputs/vehicle_dashboard/sensors'))
    if not ros_base.is_absolute():ros_base=CAR.parent/ros_base
    ros=(ros_base/'recordings'/run.name).resolve()
    if output==ros or ros in output.parents:
        raise ValueError('derived output must be outside the original ROS sensor archive')
    declared=status.get('sensor_recording',{}).get('ros_directory')
    if declared and Path(declared).resolve()!=ros:
        raise ValueError('ROS sensor directory does not match frozen configuration/run')
    events=[];health=[]
    for root in (local,ros):
        entries,state=read_sensor_journal(root);events.extend(entries);health.append(state)
    written=Counter(event['sensor'] for event in events)
    required=['uart_rx','telemetry','lidar','depth']
    pwm_keys=('front_left_pwm','rear_left_pwm','front_right_pwm','rear_right_pwm')
    has_pwm=any(any(key in row for key in pwm_keys) for row in rows)
    tx_required=(any(any(abs(float(row.get(key) or 0))>0 for key in pwm_keys) for row in rows)
                 if has_pwm else any(row.get('action') not in ('stop','stopped') for row in rows))
    # A run held at zero can legitimately have no post-initialization TX.
    if tx_required:required.append('uart_tx')
    complete=all(state['complete'] for state in health) and all(written[s]>0 for s in required)
    if not complete and not allow_incomplete_sensors:
        raise ValueError('raw sensor recording incomplete; historical partial replay requires --allow-incomplete-sensors')
    before=dict(runtime=inventory(run),ros=inventory(ros))
    output.mkdir(parents=True,exist_ok=False)
    with (output/'sensor_timeline.jsonl').open('x') as stream:
        for item in align_sensor_events(rows,events):stream.write(json.dumps(item,allow_nan=False)+'\n')
    summary=replay_recorded(run,output/'controller',profile,expectations)
    commands=json.loads((output/'controller/commands.json').read_text())
    changes=[dict(sample=item['frame'],recorded_action=item['recorded_action'],
        replay_action=item['action'],reason=item['reason'],pwm=item['pwm'])
        for item in commands if item['recorded_action']!=item['action']]
    (output/'decision_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    sources=['tools/replay_outer_loop.py','tools/replay_session.py','runtime/onboard.py',
        'control/visual_feedback.py','control/stationary_corner.py','control/drive_runtime.py',
        'control/lane_centering.py','perception/semantic_outer_route.py',
        'perception/semantic_stability.py','perception/yolopv2_semantic.py',
        'perception/yolopv2_fusion.py','perception/yolopv2_objects.py']
    manifest=dict(schema_version=1,input_run=str(run),created_at=time.time(),
        raw_sources=before,sensor_health=health,all_raw_streams_complete=complete,
        uart_tx_required=tx_required,
        historical_partial_replay=not complete,controller_summary=summary,
        changed_action_rows=len(changes),
        replay_code_sha256={p:sha256(CAR/'autodrive'/p) for p in sources},
        limitation='Recorded observations and receipt timestamps only. No altered vehicle trajectory, physical stopping, latency or model accuracy is simulated.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run');source.add_argument('--session')
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile',default='recorded',help='recorded snapshot, candidate profile ID or YAML')
    parser.add_argument('--allow-incomplete-sensors',action='store_true',help='Label legacy partial evidence explicitly')
    parser.add_argument('--expectations',help='JSON frame ranges and expected actions')
    args=parser.parse_args();run=args.run
    if args.session:
        folder=Path(args.session).resolve();state=json.loads((folder/'session.json').read_text())
        if state.get('running') is not False:raise ValueError('session is still running')
        run=Path(state['run_archive']).resolve()
        if run.parent!=(folder/'runtime/runs').resolve():raise ValueError('run path escaped session')
    expectations=json.loads(Path(args.expectations).read_text()) if args.expectations else None
    result=replay_session(run,args.output,args.profile,args.allow_incomplete_sensors,expectations)
    print(json.dumps(dict(output=str(Path(args.output).resolve()),frames=result['controller_summary']['frames'],
        changed_action_rows=result['changed_action_rows'],all_raw_streams_complete=result['all_raw_streams_complete']),indent=2))


if __name__=='__main__':main()
