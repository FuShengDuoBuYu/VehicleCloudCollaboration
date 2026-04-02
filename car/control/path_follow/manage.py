#!/usr/bin/env python3
"""
Scripts to record a path by driving a donkey car
and using an autopilot to drive the recorded path.

Usage:
    manage.py (drive) [--js] [--log=INFO] [--camera=(single|stereo)] [--myconfig=<filename>]

Options:
    -h --help           Show this screen.
    --js                Use physical joystick.
    --myconfig=filename Specify myconfig file to use.
                        [default: myconfig.py]
"""
import importlib.util
import logging
import os
import sys

APP_DIR = os.path.dirname(os.path.realpath(__file__))
DONKEYCAR_SRC = os.path.abspath(os.path.join(APP_DIR, '..', 'donkeycar'))
if DONKEYCAR_SRC not in sys.path:
    sys.path.insert(0, DONKEYCAR_SRC)

from docopt import docopt
import donkeycar as dk
from donkeycar.templates import path_follow as pf
from donkeycar.templates.complete import add_drivetrain as complete_add_drivetrain
from donkeycar.parts.path import PID_Pilot as BasePIDPilot


class EndStopPIDPilot(BasePIDPilot):
    AUTO_STOP_AT_END = True
    END_MARGIN_POINTS = 1

    def run(self, cte, throttles, closest_pt_idx):
        if self.AUTO_STOP_AT_END and throttles is not None and closest_pt_idx is not None:
            last_idx = len(throttles) - 1
            if last_idx >= 0 and closest_pt_idx >= max(0, last_idx - self.END_MARGIN_POINTS):
                logging.info("Reached end of recorded path. Stopping vehicle.")
                return 0.0, 0.0
        return super().run(cte, throttles, closest_pt_idx)


class UserThrottleLimiter:
    def __init__(self, scale):
        self.scale = float(scale)

    def run(self, mode, steering, throttle):
        steering = 0.0 if steering is None else float(steering)
        throttle = 0.0 if throttle is None else float(throttle)
        if mode == 'user':
            throttle *= self.scale
        return steering, throttle


def add_drivetrain_compat(V, cfg):
    """Use ROSPBOT backend when requested; otherwise fallback to default drivetrain."""
    if getattr(cfg, 'DRIVE_TRAIN_TYPE', None) == 'ROSPBOT':
        user_scale = float(getattr(cfg, 'USER_MODE_THROTTLE_SCALE', 1.0))
        if user_scale < 1.0:
            V.add(UserThrottleLimiter(user_scale),
                  inputs=['user/mode', 'steering', 'throttle'],
                  outputs=['drive/steering', 'drive/throttle'])
            steer_in, thr_in = 'drive/steering', 'drive/throttle'
        else:
            steer_in, thr_in = 'steering', 'throttle'

        rospbot_file = os.path.abspath(os.path.join(APP_DIR, '..', 'train_car', 'rospbot_drive.py'))
        spec = importlib.util.spec_from_file_location('rospbot_drive', rospbot_file)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        V.add(module.RospbotDrive(cfg), inputs=[steer_in, thr_in])
        return

    complete_add_drivetrain(V, cfg)


# Patch path_follow template behavior.
pf.add_drivetrain = add_drivetrain_compat
pf.PID_Pilot = EndStopPIDPilot


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(config_path=os.path.join(APP_DIR, 'config.py'),
                         myconfig=args['--myconfig'])

    EndStopPIDPilot.AUTO_STOP_AT_END = bool(getattr(cfg, 'AUTO_STOP_AT_PATH_END', True))
    EndStopPIDPilot.END_MARGIN_POINTS = int(getattr(cfg, 'PATH_END_STOP_MARGIN_POINTS', 1))

    log_level = args['--log'] or 'INFO'
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=numeric_level)

    if args['drive']:
        pf.drive(cfg, use_joystick=args['--js'], camera_type=args['--camera'])
