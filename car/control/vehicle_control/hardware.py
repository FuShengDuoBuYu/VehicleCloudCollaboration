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
        self.current_front_left = 0
        self.current_rear_left = 0
        self.current_front_right = 0
        self.current_rear_right = 0
        self._lock = threading.Lock()

    def set_four_wheels(
        self,
        front_left,
        rear_left,
        front_right,
        rear_right,
    ):
        front_left = int(clamp(front_left, -255, 255))
        rear_left = int(clamp(rear_left, -255, 255))
        front_right = int(clamp(front_right, -255, 255))
        rear_right = int(clamp(rear_right, -255, 255))
        with self._lock:
            self.current_front_left = front_left
            self.current_rear_left = rear_left
            self.current_front_right = front_right
            self.current_rear_right = rear_right
            self.current_left = int(round((front_left + rear_left) * 0.5))
            self.current_right = int(round((front_right + rear_right) * 0.5))
            # Verified on this chassis: 0=front-left, 1=rear-left,
            # 2=front-right, 3=rear-right.
            self.bot.Ctrl_Muto(0, front_left)
            self.bot.Ctrl_Muto(1, rear_left)
            self.bot.Ctrl_Muto(2, front_right)
            self.bot.Ctrl_Muto(3, rear_right)

    def set_wheels(self, left_speed, right_speed):
        left_speed = int(clamp(left_speed, -255, 255))
        right_speed = int(clamp(right_speed, -255, 255))
        self.set_four_wheels(
            left_speed,
            left_speed,
            right_speed,
            right_speed,
        )

    def ramp_four_to(
        self,
        front_left_target,
        rear_left_target,
        front_right_target,
        rear_right_target,
        transition_time,
        stop_event=None,
    ):
        targets = tuple(
            int(clamp(value, -255, 255))
            for value in (
                front_left_target,
                rear_left_target,
                front_right_target,
                rear_right_target,
            )
        )
        if transition_time <= 0:
            self.set_four_wheels(*targets)
            return

        with self._lock:
            starts = (
                self.current_front_left,
                self.current_rear_left,
                self.current_front_right,
                self.current_rear_right,
            )

        start = time.monotonic()
        while True:
            if stop_event and stop_event.is_set():
                return
            elapsed = time.monotonic() - start
            if elapsed >= transition_time:
                break
            ratio = elapsed / transition_time
            current = tuple(
                start_value + (target - start_value) * ratio
                for start_value, target in zip(starts, targets)
            )
            self.set_four_wheels(*current)
            time.sleep(0.02)
        self.set_four_wheels(*targets)

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
            self.set_four_wheels(0, 0, 0, 0)
            time.sleep(0.05)
