import os
import sys
import threading
import time


UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from Raspbot_Lib import Raspbot


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class RospbotChassis:
    def __init__(self):
        self.bot = Raspbot()
        self.current_left = 0
        self.current_right = 0
        self._lock = threading.Lock()

    def set_wheels(self, left_speed, right_speed):
        left_speed = int(clamp(left_speed, -255, 255))
        right_speed = int(clamp(right_speed, -255, 255))
        with self._lock:
            self.current_left = left_speed
            self.current_right = right_speed
            self.bot.Ctrl_Muto(0, left_speed)
            self.bot.Ctrl_Muto(1, left_speed)
            self.bot.Ctrl_Muto(2, right_speed)
            self.bot.Ctrl_Muto(3, right_speed)

    def ramp_to(self, left_target, right_target, transition_time, stop_event=None):
        left_target = int(clamp(left_target, -255, 255))
        right_target = int(clamp(right_target, -255, 255))

        if transition_time <= 0:
            self.set_wheels(left_target, right_target)
            return

        with self._lock:
            start_left = self.current_left
            start_right = self.current_right

        start = time.monotonic()
        while True:
            if stop_event and stop_event.is_set():
                return
            elapsed = time.monotonic() - start
            if elapsed >= transition_time:
                break
            ratio = elapsed / transition_time
            left_now = start_left + (left_target - start_left) * ratio
            right_now = start_right + (right_target - start_right) * ratio
            self.set_wheels(left_now, right_now)
            time.sleep(0.02)

        self.set_wheels(left_target, right_target)

    def hold(self, duration, stop_event=None):
        start = time.monotonic()
        while True:
            if stop_event and stop_event.is_set():
                return False
            if time.monotonic() - start >= duration:
                return True
            time.sleep(0.05)

    def stop(self):
        for _ in range(3):
            self.set_wheels(0, 0)
            time.sleep(0.05)
