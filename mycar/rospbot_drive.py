#!/usr/bin/env python3
import os
import sys

_UTILS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from Raspbot_Lib import Raspbot


class RospbotDrive:
    def __init__(self, cfg):
        self.bot = Raspbot()
        self.max_speed = int(getattr(cfg, "ROSPBOT_MAX_SPEED", 180))
        self.turn_scale = float(getattr(cfg, "ROSPBOT_TURN_SCALE", 0.7))
        self.deadzone = float(getattr(cfg, "ROSPBOT_DEADZONE", 0.02))

    @staticmethod
    def _clip(value, low, high):
        return max(low, min(high, value))

    def _set_wheels(self, left, right):
        left = int(self._clip(left, -255, 255))
        right = int(self._clip(right, -255, 255))
        self.bot.Ctrl_Muto(0, left)
        self.bot.Ctrl_Muto(1, left)
        self.bot.Ctrl_Muto(2, right)
        self.bot.Ctrl_Muto(3, right)

    def run(self, steering, throttle):
        steering = float(steering) if steering is not None else 0.0
        throttle = float(throttle) if throttle is not None else 0.0

        if abs(steering) < self.deadzone:
            steering = 0.0
        if abs(throttle) < self.deadzone:
            throttle = 0.0

        speed = int(self._clip(throttle, -1.0, 1.0) * self.max_speed)
        turn = int(self._clip(steering, -1.0, 1.0) * self.max_speed * self.turn_scale)

        left = speed + turn
        right = speed - turn
        self._set_wheels(left, right)

    def shutdown(self):
        self._set_wheels(0, 0)
