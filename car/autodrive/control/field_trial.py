"""Explicit experimental access after wheel mapping and grounded start/stop.

The normal calibrated-motion gate stays unchanged. This restricted path is for
the operator-authorized Jetson/raw-image YOLO trials, with full recording.
"""
import math
import hashlib
import json
from pathlib import Path

from .stationary_corner import StationaryCornerConfig


def validate_stationary_corner_config(config, *, motors_enabled=False):
    """Validate the opt-in before models/devices; physical output needs pulse evidence."""
    settings = StationaryCornerConfig(**config.get('stationary_corner', {}))
    perception = config.get('perception', {})
    outer = perception.get('semantic_outer_loop', {})
    surface=perception.get('track_colors',{}).get('surface_assist',False)
    if (type(surface) is not bool or surface and (not settings.enabled or not settings.visual_feedback
            or not settings.visual_corner_guard
            or perception.get('track_colors',{}).get('enabled') is not True)):
        raise ValueError('track surface assist requires visual corner guard and matched track colors')
    if not settings.enabled:
        return settings
    if settings.visual_corner_guard and (perception.get('track_colors',{}).get('enabled') is not True
                                        or outer.get('visual_feedback') is not True):
        raise ValueError('visual corner guard requires matched track colors and visual route selection')
    wheels = config.get('wheels', {})
    if (perception.get('mode') != 'yolopv2' or outer.get('enabled') is not True
            or outer.get('stationary_corners') is not True
            or config.get('outer_loop', {}).get('enabled', False)
            or config.get('safety', {}).get('corner_continuation', {}).get('enabled', False)):
        raise ValueError('stationary corners require current YOLO stationary outer-route observations')
    if (type(wheels.get('stationary_pivot_pwm')) is not int
            or wheels['stationary_pivot_pwm'] != 30 or wheels.get('pwm_limit', 0) < 30
            or wheels.get('transition_time', .25) != 0):
        raise ValueError('stationary corners require explicit 30 PWM and nonblocking transition_time=0')
    minimum_observations = 2 if settings.visual_feedback else 5
    observations = config.get('safety', {}).get('resume_valid_frames', 0)
    if type(observations) is not int or observations < minimum_observations:
        raise ValueError(f'stationary corners require at least {minimum_observations} startup/recovery observations')
    if motors_enabled:
        _validate_stationary_calibration(config, settings)
    return settings


def _validate_stationary_calibration(config, settings):
    calibration = config.get('stationary_corner_calibration', {})
    if (calibration.get('verified') is not True
            or type(calibration.get('yaw_sign')) is not int
            or calibration['yaw_sign'] != settings.yaw_sign
            or not isinstance(calibration.get('evidence'), str)):
        raise ValueError('stationary pivot requires verified calibration evidence and matching yaw_sign')
    path = Path(calibration['evidence']).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        stop = record['stop']
        delta = record['yaw_delta_rad']
        direction = record['direction']
        zero = [0, 0, 0, 0]
        if (record['status'] != 'pulse_recorded' or record['errors'] != []
                or type(record['pwm']) is not int or record['pwm'] != 30
                or record['final_pwm'] != zero or stop['final_pwm'] != zero
                or stop.get('error') is not None or stop['reason'] != 'duration reached'
                or direction not in ('left', 'right')
                or type(delta) not in (int, float) or not math.isfinite(delta)
                or not math.radians(1.) <= abs(delta) < math.pi
                or (1 if delta > 0 else -1) * (1 if direction == 'right' else -1) != settings.yaw_sign
                or type(record['video_frames']) is not int or record['video_frames'] < 1):
            raise ValueError('pulse does not establish the configured pivot/yaw direction and stopped output')
        expected = config['vehicle']
        if record['profile']['backend'] != expected['backend']:
            raise ValueError('calibration backend differs')
        for key in ('motor_order', 'wheel_signs', 'serial_port', 'car_type'):
            if record['profile']['chassis'][key] != expected['chassis'][key]:
                raise ValueError('calibration chassis differs: ' + key)
        before, after = record['before']['attitude'], record['after']['attitude']
        for sample in (before, after):
            values = sample['value']['radians']
            if (sample['valid'] is not True or sample['stale'] is not False
                    or len(values) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError('invalid recorded attitude')
        measured = math.atan2(math.sin(after['value']['radians'][2]-before['value']['radians'][2]),
                              math.cos(after['value']['radians'][2]-before['value']['radians'][2]))
        if not math.isclose(measured, delta, abs_tol=1e-6):
            raise ValueError('summary yaw differs from recorded attitude')
        stamps = (before['received_monotonic'], stop['command_started'], stop['command_completed'],
                  stop['stop_started'], stop['stop_completed'], after['received_monotonic'])
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in stamps)
                or list(stamps) != sorted(stamps)):
            raise ValueError('calibration command/stop/attitude timeline invalid')
        for name in ('raw.avi', 'telemetry.jsonl', 'frames.jsonl', 'uart_rx.bin'):
            artifact = path.parent / name
            if (not artifact.is_file() or artifact.stat().st_size == 0
                    or hashlib.sha256(artifact.read_bytes()).hexdigest() != record['artifacts_sha256'][name]):
                raise ValueError('calibration artifact missing or changed: ' + name)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError('invalid stationary calibration evidence: ' + str(exc)) from exc


class TrialMotionWindow:
    """One wall-clock window beginning with the first nonzero wheel output.

    The runtime checks it before/after perception. Existing stale-command
    watchdog remains the independent fallback if the control loop stalls.
    Stops during a trial do not reset its deadline or grant more movement.
    """

    def __init__(self, duration_seconds=0.0):
        if (isinstance(duration_seconds, bool)
                or not isinstance(duration_seconds, (int, float))
                or not math.isfinite(duration_seconds) or duration_seconds < 0):
            raise ValueError('motion window must be finite and nonnegative')
        self.duration_seconds = float(duration_seconds)
        self.started_at = None

    def record_output(self, pwm, now):
        if self.started_at is None and any(pwm):
            self.started_at = now

    def expired(self, now):
        return (self.duration_seconds > 0 and self.started_at is not None
                and now - self.started_at >= self.duration_seconds)


def validate_field_trial_config(config):
    corner_settings = validate_stationary_corner_config(config, motors_enabled=True)
    vehicle = config.get('vehicle', {})
    status = vehicle.get('status', {})
    if vehicle.get('backend') != 'rosmaster':
        raise ValueError('field trial requires the verified Rosmaster backend')
    for key in ('wheel_mapping_verified', 'ground_forward_stop_verified'):
        if status.get(key) is not True:
            raise ValueError('field trial requires verified ' + key)
    chassis = vehicle.get('chassis', {})
    if (chassis.get('motor_order') != [0, 1, 2, 3]
            or chassis.get('wheel_signs') != [1, 1, 1, 1]
            or chassis.get('serial_port') != '/dev/myserial'):
        raise ValueError('field trial differs from verified wheel mapping/port')
    wheels = config.get('wheels', {})
    for key in ('pwm_limit', 'front_left_base_pwm', 'rear_left_base_pwm',
                'front_right_base_pwm', 'rear_right_base_pwm', 'tight_turn_outside_pwm'):
        if type(wheels.get(key)) is not int or wheels[key] != 30:
            raise ValueError('field trial requires 30 PWM: ' + key)
    if wheels.get('tight_turn_inside_pwm') != 0:
        raise ValueError('field trial requires non-reversing inside stop')
    perception = config.get('perception', {})
    yolo = perception.get('yolopv2', {})
    if (perception.get('mode') != 'yolopv2' or yolo.get('enabled') is not True
            or yolo.get('required_for_motion') is not True
            or yolo.get('drivable_only') is not False or yolo.get('detect_objects') is not True
            or not str(yolo.get('device', '')).startswith('cuda')
            or perception.get('semantic_outer_loop', {}).get('enabled') is not True):
        raise ValueError('field trial requires full GPU YOLO and outer route selection')
    if config.get('perspective', {}).get('calibration') is not None:
        raise ValueError('field trial geometry uses original camera coordinates')
    if config.get('outer_loop', {}).get('enabled') or config.get('safety', {}).get('corner_continuation', {}).get('enabled'):
        raise ValueError('field trial cannot use classical or blind corner continuation')
    camera = config.get('camera', {})
    if (camera.get('index') != 0 or camera.get('width') != 640 or camera.get('height') != 480
            or camera.get('rotation_degrees', 0) != 0
            or camera.get('flip_horizontal', False) is not False
            or camera.get('flip_vertical', False) is not False
            or camera.get('gimbal', {}).get('initialize_on_startup') is not False):
        raise ValueError('field trial camera differs from recorded fixed-camera geometry')
    if config.get('runtime', {}).get('archive_runs', True) is not True:
        raise ValueError('field trial requires video and log archival')
    # Preserve finite, short freshness and stale-command limits.
    safety = config.get('safety', {})
    # The visual profile uses the measured capture-to-application budget;
    # legacy trials keep their original stricter source-age bound.
    yolo_age_limit = .6 if corner_settings.enabled and corner_settings.visual_feedback else .3
    for value, limit in ((yolo.get('max_result_age_seconds', 0), yolo_age_limit),
                         (camera.get('stale_timeout', 0), .25),
                         (safety.get('watchdog_timeout', 0), .4)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= limit:
            raise ValueError('field trial freshness/watchdog limit is invalid')


def field_trial_recording_ready(archive_state, log_state):
    for state in (archive_state, log_state):
        if state.get('error'):
            raise RuntimeError('field trial recording failed: ' + str(state['error']))
        if state.get('dropped_frames', 0) or state.get('dropped_rows', 0):
            raise RuntimeError('field trial recording dropped frames/rows; stop before further motion')
    return archive_state.get('written_frames', 0) > 0
