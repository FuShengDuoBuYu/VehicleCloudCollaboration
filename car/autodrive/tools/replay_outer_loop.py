#!/usr/bin/env python3
"""Replay recorded camera frames through current YOLO/control, never hardware.

This deliberately uses synchronous inference to inspect each retained frame.
Recorded timestamps supply controller dt. It cannot reproduce asynchronous
deadline/dropout timing or prove actual stopping/line clearance.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import yaml

CAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAR))
from autodrive.runtime.onboard import (load_config, DEFAULT_CONFIG, build_components,
                                      analyze, PerceptionMotionGate)


def recorded_replay_config(run, profile):
    """Use an immutable run snapshot when requested, without today's defaults."""
    if profile == 'recorded':
        source = Path(run)/'resolved_runtime_config.yaml'
        if not source.is_file():
            raise ValueError('recorded replay requires resolved_runtime_config.yaml')
        config = yaml.safe_load(source.read_text())
        required = ('perception','camera','wheels','safety','perspective')
        if not isinstance(config,dict) or any(not isinstance(config.get(k),dict) for k in required):
            raise ValueError('invalid resolved runtime configuration')
        evidence = dict(selection='recorded',source=str(source),
                        sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    else:
        _, config = load_config(DEFAULT_CONFIG,profile)
        evidence = dict(selection=str(profile),source='current resolved candidate configuration',
                        sha256=hashlib.sha256(yaml.safe_dump(config,sort_keys=True).encode()).hexdigest())
    return config, evidence


def recorded_processing_budget(row, maximum_seconds):
    """Recorded allowance; None means missing evidence, never current CPU time."""
    exact = row.get('stationary_perception_budget_valid')
    if exact not in (None, ''):
        if exact not in ('True', 'False'):
            raise ValueError('invalid recorded processing budget flag')
        return exact == 'True'
    if row.get('reason') == 'inference exceeded safety limit':
        return False
    values = (row.get('inference_ms'), row.get('boundary_ms'))
    if any(value in (None, '') for value in values):
        return None
    times = [float(value) / 1000 for value in values]
    if not all(math.isfinite(value) and value >= 0 for value in times):
        raise ValueError('invalid recorded processing time')
    # Three-decimal millisecond CSVs cannot resolve a sub-microsecond threshold tie.
    total = sum(times)
    if abs(total - maximum_seconds) <= 1e-6:
        return None
    return total < maximum_seconds


def recorded_stationary_inputs(row):
    """Decode archived control/IMU evidence; never replace it with live time."""
    required = ('control_monotonic', 'imu_yaw_rad', 'imu_received_monotonic',
                'imu_valid', 'imu_stale', 'imu_age_s', 'frame_age_exact_s',
                'stationary_external_allowed', 'stationary_perception_budget_valid')
    if any(key not in row for key in required):
        raise ValueError('stationary corner replay requires recorded IMU/control inputs')
    def number(key, optional=False, nonnegative=False):
        value = row[key]
        if optional and value in (None, ''):
            return None
        value = float(value)
        if not math.isfinite(value) or (nonnegative and value < 0):
            raise ValueError('invalid stationary corner input: ' + key)
        return value
    def boolean(key):
        value = row[key]
        if value == 'True':
            return True
        if value == 'False':
            return False
        raise ValueError('invalid stationary corner boolean: ' + key)
    yaw = number('imu_yaw_rad', optional=True)
    return dict(now=number('control_monotonic', nonnegative=True),
                imu_sample={'value': {'radians': [0., 0., yaw]},
                            'received_monotonic': number('imu_received_monotonic', optional=True,
                                                         nonnegative=True),
                            'age_s': number('imu_age_s', optional=True, nonnegative=True),
                            'valid': boolean('imu_valid'), 'stale': boolean('imu_stale')},
                frame_age_seconds=number('frame_age_exact_s', nonnegative=True),
                external_motion_allowed=boolean('stationary_external_allowed'),
                perception_budget_valid=boolean('stationary_perception_budget_valid'))


def recorded_application_inputs(row):
    """Read optional final execution evidence; legacy timing stays unknown."""
    allowed=row.get('stationary_application_allowed')
    if allowed in (None,''):
        return None
    if allowed not in ('True','False'):
        raise ValueError('invalid recorded application permission')
    result={'allowed':allowed=='True','reason':row.get('stationary_application_reason') or ''}
    for key in ('stationary_application_monotonic','semantic_application_age_s',
                'frame_application_age_s','imu_application_age_s'):
        value=row.get(key)
        number=None if value in (None,'') else float(value)
        if number is not None and (not math.isfinite(number) or number<0):
            raise ValueError('invalid recorded application time: '+key)
        result[key]=number
    return result


def replay(run_dir, output_dir=None, profile='rosmaster_jetson_yolopv2_outer_trial',
           expectations=None, mode='auto'):
    run = Path(run_dir).resolve()
    if mode not in ('auto', 'recorded', 'video'):
        raise ValueError('replay mode must be auto, recorded or video')
    if mode == 'recorded' or (mode == 'auto' and (run/'semantic_inputs').is_dir()):
        return replay_recorded(run, output_dir, profile, expectations)
    video = run / 'raw.mp4'
    with (run / 'onboard_log.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    output = Path(output_dir) if output_dir else CAR.parent / 'outputs/outer_loop_replays' / (
        run.name + '-' + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    config, config_evidence = recorded_replay_config(run, profile)
    if config.get('stationary_corner', {}).get('enabled'):
        raise ValueError('stationary corner replay requires recorded masks and IMU; use --mode recorded')
    config['perception']['yolopv2']['asynchronous'] = False
    config['cloud_arbitration']['enabled'] = False
    config.setdefault('cloud_recovery',{})['enabled'] = False
    config['camera']['gimbal']['initialize_on_startup'] = False
    (output / 'replay_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('cannot read recorded video: ' + str(video))
    detector = driver = writer = watchdog = None
    results = []
    try:
        detector, estimator, controller, mapper, tracker, driver, watchdog = build_components(
            config, False, None)
        gate_config = config.get('safety', {})
        gate = PerceptionMotionGate(
            resume_valid_frames=int(gate_config.get('resume_valid_frames', 5)),
            resume_min_confidence=float(gate_config.get('resume_min_confidence', .5)),
            maximum_lateral_jump=float(gate_config.get('resume_maximum_lateral_jump', 0)),
            maximum_heading_jump=float(gate_config.get('resume_maximum_heading_jump', 0)),
            require_consistent_source=bool(gate_config.get('resume_require_consistent_source', False)))
        previous = None
        for index, recorded in enumerate(rows):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('recorded video ended before aligned CSV')
            timestamp = float(recorded['timestamp_s'])
            dt = None if previous is None else max(.001, timestamp-previous)
            previous = timestamp
            estimate, proposed, overlay, _, _, boundary = analyze(
                detector, estimator, controller, mapper, frame,
                str(config.get('runtime',{}).get('route_hint','center')), dt,
                float(config['safety'].get('maximum_inference_time', .25)), tracker,
                captured_at=time.monotonic())
            applied = gate.filter(proposed, estimate, boundary)
            state = driver.apply(applied)  # motors_enabled=False, no chassis
            pwm = [state[k] for k in ('front_left_pwm','rear_left_pwm',
                                      'front_right_pwm','rear_right_pwm')]
            if writer is None:
                writer = cv2.VideoWriter(str(output/'replay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                        max(1., cap.get(cv2.CAP_PROP_FPS)), (frame.shape[1], frame.shape[0]))
                if not writer.isOpened(): raise RuntimeError('cannot write replay video')
            cv2.putText(overlay, 'REPLAY: %s PWM %s' % (applied.action, pwm),
                        (8, 155), cv2.FONT_HERSHEY_SIMPLEX, .42, (0,255,255), 1)
            writer.write(overlay)
            results.append({'frame':index,'recorded_sample':recorded.get('sample'),
                'timestamp_s':timestamp,'recorded_action':recorded.get('action'),
                'proposal':proposed.action,'action':applied.action,'pwm':pwm,
                'reason':applied.reason,'valid':estimate.valid,'steering':applied.steering})
        if cap.read()[0]: raise RuntimeError('recorded video has frames missing from aligned CSV')
        if not results: raise RuntimeError('empty recorded run')
        checks = []
        for expected in expectations or []:
            selected = [r for r in results if expected['start_frame'] <= r['frame'] <= expected['end_frame']]
            checks.append({'expectation':expected,'frames':len(selected),
                           'passed':bool(selected) and all(r['action'] in expected['actions'] for r in selected)})
        summary = {'input_run':str(run),'output_dir':str(output),'frames':len(results),
            'mode':'synchronous current-model replay; motors disabled; cloud disabled',
            'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
            'configuration':config_evidence,
            'actions':dict(Counter(r['action'] for r in results)),
            'stop_reasons':dict(Counter(r['reason'] for r in results if r['action']=='stop')),
            'maximum_pwm':max(abs(v) for r in results for v in r['pwm']),
            'expectation_checks':checks,
            'expected_behavior_verified': all(c['passed'] for c in checks) if checks else None,
            'limitation':'No frame-labelled expectations supplied' if not checks else 'Expectations are supplied annotations, not physical driving results'}
        (output/'commands.json').write_text(json.dumps(results,indent=2)+'\n')
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary
    finally:
        if watchdog is not None: watchdog.close()
        if driver is not None: driver.stop('replay complete')
        if detector is not None: detector.close()
        cap.release()
        if writer is not None: writer.release()


def replay_recorded(run, output_dir=None, profile='rosmaster_jetson_yolopv2_outer_trial',
                    expectations=None):
    """Replay exact consumed masks, sequences and ages through current control.

    No model, camera, serial port or cloud client is opened. The model input is
    lossless, so compression and different asynchronous frame selection cannot
    silently change the evidence. This does not simulate actuator/archive/CPU
    timing, nor does it evaluate different model weights.
    """
    from autodrive.perception.yolopv2_semantic import YOLOPv2SemanticDetector
    from autodrive.perception.yolopv2_fusion import YOLOPv2FusionConfig, _MaskResult
    from autodrive.control.lane_centering import RoadCenterlineEstimator, LaneCenteringController, LCCConfig, DifferentialDriveCommand
    from autodrive.control.drive_runtime import SafeWheelDriver, WheelMappingConfig

    class RecordedDetector(YOLOPv2SemanticDetector):
        def predict_masks(self, frame, captured_at=None):
            return self._consumer_result.drivable_mask.copy(), self._consumer_result.lane_mask.copy()

    run = Path(run).resolve()
    with (run/'onboard_log.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    samples = [int(row['sample']) for row in rows]
    if (not samples or samples != list(range(len(samples)))
            or any(int(row.get('archive_dropped_frames') or 0) > 0 for row in rows)):
        raise ValueError('incomplete recorded control sequence; cannot verify stateful replay')
    if not (run/'status.json').is_file():
        raise ValueError('incomplete recorded run: final status is missing')
    recorded_status = json.loads((run/'status.json').read_text())
    termination = recorded_status.get('termination',{})
    archive_state = recorded_status.get('diagnostics',{}).get('archive',{})
    if (termination.get('stopped') is not True
            or not np.isfinite(float(termination.get('completed_at',0)))
            or float(termination.get('completed_at',0)) <= 0
            or recorded_status.get('last_result',{}).get('sample') != samples[-1]
            or archive_state.get('written_frames') != len(rows)
            or archive_state.get('dropped_frames',0) or archive_state.get('error')):
        raise ValueError('incomplete recorded archive/final state; cannot verify stateful replay')
    output = Path(output_dir) if output_dir else CAR.parent/'outputs/outer_loop_replays'/(
        run.name+'-recorded-'+str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    config, config_evidence = recorded_replay_config(run, profile)
    if config['perspective'].get('calibration') or config['perception']['mode'] != 'yolopv2':
        raise ValueError('recorded semantic replay requires the image-space YOLO profile')
    config['perception']['yolopv2']['asynchronous'] = False
    config['cloud_arbitration']['enabled'] = False
    config['camera']['gimbal']['initialize_on_startup'] = False
    stationary_enabled = config.get('stationary_corner', {}).get('enabled') is True
    recovery = None
    cloud_requested = config.get('cloud_recovery',{}).get('enabled') is True
    cloud_journal = run/'cloud/decisions.jsonl'
    if cloud_requested and cloud_journal.is_file():
        from autodrive.runtime.onboard import cloud_configuration
        from autodrive.tools.recorded_recovery import RecordedRecoveryCoordinator, read_records
        recovery = RecordedRecoveryCoordinator(cloud_configuration(config),run)
        applications = read_records(run/'cloud/applications.jsonl')
        if len(recovery.decisions)!=len(rows) or len(applications)!=len(rows):
            raise ValueError('incomplete cloud decision/application sequence')
        for index,(decision,application) in enumerate(zip(recovery.decisions,applications)):
            if ('decision_index' in decision and decision['decision_index']!=index
                    or 'decision_index' in application and application['decision_index']!=index):
                raise ValueError('cloud decision/application ordering mismatch')
        journal = recorded_status.get('cloud_arbitration',{}).get('journal')
        if journal is not None and (not journal.get('closed') or journal.get('error')
                or journal.get('written')!=journal.get('enqueued')):
            raise ValueError('incomplete cloud asynchronous journal')
    elif cloud_requested:
        if profile=='recorded' or recorded_status.get('cloud_arbitration',{}).get('enabled'):
            raise ValueError('cloud-enabled recording is missing its cloud journals')
        config['cloud_recovery']['enabled']=False
    stationary_inputs = ([recorded_stationary_inputs(row) for row in rows]
                         if stationary_enabled else None)
    application_inputs=([recorded_application_inputs(row) for row in rows]
                        if stationary_enabled else [None]*len(rows))
    (output/'replay_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    now = [0.]
    from autodrive.perception.semantic_outer_route import route_options
    outer_config=route_options(config)
    detector = RecordedDetector(YOLOPv2FusionConfig(**config['perception']['yolopv2']),
        model=object(), clock=lambda: now[0],
        output_width=config['perception']['mask_width'], output_height=config['perception']['mask_height'],
        outer_loop_config=outer_config,
        stability_config=config['perception'].get('semantic_stability'),
        track_color_config=config['perception'].get('track_colors'))
    estimator = RoadCenterlineEstimator(**config.get('centerline',{}))
    controller = LaneCenteringController(LCCConfig(**config.get('lcc',{})))
    driver = SafeWheelDriver(chassis=None, motors_enabled=False,
                             config=WheelMappingConfig(**config['wheels']))
    safety = config['safety']
    gate = PerceptionMotionGate(resume_valid_frames=safety.get('resume_valid_frames',5),
        resume_min_confidence=safety.get('resume_min_confidence',.5),
        maximum_lateral_jump=safety.get('resume_maximum_lateral_jump',0),
        maximum_heading_jump=safety.get('resume_maximum_heading_jump',0),
        require_consistent_source=safety.get('resume_require_consistent_source',False))
    stationary_runtime = None
    if stationary_enabled:
        from autodrive.runtime.onboard import StationaryCornerRuntime
        from autodrive.control.stationary_corner import StationaryCornerConfig
        stationary_runtime = StationaryCornerRuntime(
            StationaryCornerConfig(**config['stationary_corner']), gate,
            pivot_verified=True)  # Virtual wheel output only; no chassis exists.
    results, source_hashes, previous, writer = [], {}, None, None
    last_sequence, observation = None, None
    try:
        for index, recorded in enumerate(rows):
            sequence = int(recorded['semantic_sequence'])
            source = run/'semantic_inputs'/('%08d.npz' % sequence)
            if sequence != last_sequence:
                with np.load(source,allow_pickle=False) as data:
                    observation = {key:data[key].copy() for key in data.files}
                if int(observation['sequence']) != sequence:
                    raise ValueError('semantic archive sequence mismatch')
                source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
                last_sequence = sequence
            captured = float(observation['captured_at'])
            exact_age = recorded.get('semantic_result_age_exact_s')
            age = float(exact_age or recorded['semantic_result_age_s'])
            if not np.isfinite([captured,age]).all() or age < 0:
                raise ValueError('invalid archived observation time')
            # Preserve explicit failure from legacy rounded CSVs. Otherwise a
            # recorded .30004 -> .3000 could incorrectly become fresh in replay.
            limit = detector.config.max_result_age_seconds
            if 'stale result' in recorded.get('semantic_fusion_source',''):
                age = max(age,limit+1e-6)
            elif not exact_age and abs(age-limit) <= .00005:
                # Legacy non-stale label resolves this rounded threshold tie.
                age = limit-1e-6
            now[0] = captured + age
            frame = observation['frame']
            inference = float(recorded.get('semantic_inference_ms') or 0)/1000
            result = _MaskResult(sequence,captured,captured+inference,inference,
                recorded.get('semantic_precision') or 'fp16',observation['road'],observation['lane'],
                tuple(json.loads(str(observation['detections_json']))),frame)
            detector._consumer_result = detector._latest = result
            detector._error = ('recorded inference error' if 'inference error' in
                               recorded.get('semantic_fusion_source','') else None)
            timestamp = float(recorded['timestamp_s'])
            dt = None if previous is None else max(.001,timestamp-previous)
            previous = timestamp
            estimate, proposed, overlay, _, _, boundary = analyze(detector,estimator,controller,
                None,frame,str(config.get('runtime',{}).get('route_hint','center')),dt,
                math.inf,captured_at=captured)
            # Host processing time is not experiment evidence. Reapply the recorded
            # gate, and conservatively stop if a legacy row lacks this evidence.
            processing_budget = recorded_processing_budget(recorded,
                float(safety.get('maximum_inference_time', .25)))
            if processing_budget is False or (processing_budget is None and proposed.action != 'stop'):
                proposed = DifferentialDriveCommand('stop', 0., 0., 0., estimate.confidence,
                    'inference exceeded safety limit' if processing_budget is False
                    else 'recorded processing budget unavailable')
            applied = (stationary_runtime.filter(proposed, estimate, boundary,
                                                **stationary_inputs[index])
                       if stationary_runtime is not None else gate.filter(proposed,estimate,boundary))
            if recovery is not None:
                from autodrive.control.cloud_recovery import build_evidence
                inputs=stationary_inputs[index]
                control_now=float(recovery.decisions[index]['now'])
                if abs(control_now-inputs['now'])>1e-6:
                    raise ValueError('cloud journal and local control clocks do not match')
                if recovery.decisions[index].get('sequence')!=sequence:
                    raise ValueError('cloud journal and semantic sequence do not match')
                evidence=build_evidence(detector.get_consumed_observation(),boundary,applied,
                    now=control_now,frame_age=inputs['frame_age_seconds'],
                    external_allowed=inputs['external_motion_allowed'],
                    perception_valid=inputs['perception_budget_valid'],
                    pivot_verified=stationary_runtime.pivot_verified,
                    imu_sample=inputs['imu_sample'],local_state=stationary_runtime.get_state())
                applied=recovery.filter(applied,evidence,control_now,
                    local_deadline=stationary_runtime.controller.get_state()['motion_deadline'])
                detector.update_route_state(boundary,applied)
            application=application_inputs[index]
            if application is not None and not application['allowed']:
                applied=stationary_runtime.apply_veto(application['reason'] or 'recorded final application veto',
                    now=application.get('stationary_application_monotonic'))
                if recovery is not None:
                    applied=recovery.veto(applied.reason,application.get('stationary_application_monotonic') or control_now)
            if recovery is not None:
                from autodrive.runtime.onboard import apply_runtime_command
                applied_at=(application or {}).get('stationary_application_monotonic') or control_now
                with patch('autodrive.control.drive_runtime.time.monotonic',return_value=applied_at):
                    applied,state=apply_runtime_command(driver,applied,stationary_runtime,recovery)
            else:state = driver.apply(applied)
            pwm = [state[key] for key in ('front_left_pwm','rear_left_pwm','front_right_pwm','rear_right_pwm')]
            if writer is None:
                writer = cv2.VideoWriter(str(output/'replay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                    float(config['camera']['fps']),(frame.shape[1],frame.shape[0]))
                if not writer.isOpened(): raise RuntimeError('cannot write replay video')
            cv2.putText(overlay,'RECORDED MASKS: %s PWM %s' % (applied.action,pwm),
                        (8,155),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,255),1)
            writer.write(overlay)
            results.append({'frame':index,'recorded_sample':recorded.get('sample'),
                'semantic_sequence':sequence,'semantic_age_s':age,'timestamp_s':timestamp,
                'recorded_action':recorded.get('action'),'proposal':proposed.action,'action':applied.action,
                'pwm':pwm,'reason':applied.reason,'valid':estimate.valid,'steering':applied.steering,
                'stability':dict(detector.stabilizer.state),
                'track_colors': {'enabled':boundary.semantic_track_colors,
                    'allowed_white_pixels':boundary.semantic_allowed_white_pixels,
                    'yellow_pixels':boundary.semantic_yellow_pixels,
                    'ego_yellow_ratio':boundary.ego_yellow_ratio},
                'stationary_corner': (None if stationary_runtime is None
                                      else stationary_runtime.get_state()),
                'recorded_processing_budget_valid':processing_budget,
                'recorded_application':application,
                'fusion':None if recovery is None else asdict(recovery.fusion.last_decision),
                'cloud_recovery':None if recovery is None else recovery.get_state(),
                'recorded_pwm':[float(recorded.get(key) or 0) for key in
                    ('front_left_pwm','rear_left_pwm','front_right_pwm','rear_right_pwm')]})
        if not results: raise RuntimeError('empty recorded run')
        checks = []
        for expected in expectations or []:
            selected = [r for r in results if expected['start_frame'] <= r['frame'] <= expected['end_frame']]
            checks.append({'expectation':expected,'frames':len(selected),
                           'passed':bool(selected) and all(r['action'] in expected['actions'] for r in selected)})
        summary = {'input_run':str(run),'output_dir':str(output),'frames':len(results),
            'mode':'recorded lossless inputs/masks and reply timing; motors/network disabled',
            'cloud_recovery_replayed':recovery is not None,
            'cloud_replay':None if recovery is None else recovery.get_state()['recorded_replay'],
            'cloud_limitation':('No recorded cloud evidence; local-only replay' if cloud_requested and recovery is None else None),
            'matched_action_rows':sum(r['action']==r['recorded_action'] for r in results),
            'matched_pwm_rows':sum(r['pwm']==r['recorded_pwm'] for r in results),
            'unique_semantic_inputs':len(source_hashes),'semantic_input_sha256':source_hashes,
            'configuration':config_evidence,
            'sequence_complete':True,
            'csv_sha256':hashlib.sha256((run/'onboard_log.csv').read_bytes()).hexdigest(),
            'actions':dict(Counter(r['action'] for r in results)),
            'stop_reasons':dict(Counter(r['reason'] for r in results if r['action']=='stop')),
            'maximum_pwm':max(abs(v) for r in results for v in r['pwm']),
            'processing_timing_source':'recorded gate or recorded milliseconds; missing/ambiguous evidence stops',
            'processing_budget_unknown_rows':sum(r['recorded_processing_budget_valid'] is None for r in results),
            'stationary_corner_replayed': stationary_enabled,
            'recorded_final_application_rows':sum(item is not None for item in application_inputs),
            'final_application_veto_rows':sum(item is not None and not item['allowed']
                                              for item in application_inputs),
            'final_application_limit':('Final application evidence is consumed only where explicitly recorded; '
                                       'legacy missing application times/vetoes remain unknown. '
                                       'Changed processing or actuator timing is not simulated.'),
            'expectation_checks':checks,
            'expected_behavior_verified':all(c['passed'] for c in checks) if checks else None,
            'limitation':'Current postprocessing/control only; no new model inference or actuator/archive/CPU timing simulation. Physical motion not verified.'}
        (output/'commands.json').write_text(json.dumps(results,indent=2)+'\n')
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary
    finally:
        driver.stop('recorded replay complete'); detector.close()
        if writer is not None: writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run')
    source.add_argument('--status')
    parser.add_argument('--after',type=float,default=0.)
    parser.add_argument('--output')
    parser.add_argument('--profile', default='rosmaster_jetson_yolopv2_outer_trial',
                        help='Profile ID/YAML, or recorded to use the frozen runtime configuration')
    parser.add_argument('--mode',choices=['auto','recorded','video'],default='auto',
                        help='auto prefers lossless recorded masks; video re-runs the GPU model on compressed video')
    parser.add_argument('--expectations',help='JSON array of start_frame/end_frame/actions/source annotations')
    args = parser.parse_args()
    run = args.run
    if args.status:
        status = json.loads(Path(args.status).read_text())
        if (status.get('termination',{}).get('completed_at',0) < args.after
                or status.get('vehicle',{}).get('field_trial') is not True):
            raise ValueError('no completed field-trial run from this cycle')
        run = status['run_archive']
    expectations = json.loads(Path(args.expectations).read_text()) if args.expectations else None
    print(json.dumps(replay(run,args.output,profile=args.profile,
                           expectations=expectations,mode=args.mode),indent=2),flush=True)


if __name__ == '__main__': main()
