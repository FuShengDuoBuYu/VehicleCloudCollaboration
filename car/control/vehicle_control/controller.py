import threading
from dataclasses import asdict

from .hardware import RospbotChassis
from .settings import CRUISE_CONFIG, LANE_CHANGE_CONFIG


class VehicleController:
    def __init__(self, chassis=None, cruise_config=CRUISE_CONFIG, lane_change_config=LANE_CHANGE_CONFIG):
        self.chassis = chassis or RospbotChassis()
        self.cruise_config = cruise_config
        self.lane_change_config = lane_change_config
        self._action_lock = threading.Lock()
        self._action_thread = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._current_action = "stopped"
        self._status = "idle"

    def _set_state(self, action, status):
        with self._state_lock:
            self._current_action = action
            self._status = status

    def get_state(self):
        with self._state_lock:
            return {
                "current_action": self._current_action,
                "status": self._status,
                "cruise": asdict(self.cruise_config),
                "lane_change": asdict(self.lane_change_config),
            }

    def _cancel_active_action(self, keep_wheels=False):
        thread = None
        with self._action_lock:
            if self._action_thread and self._action_thread.is_alive():
                self._stop_event.set()
                thread = self._action_thread
            self._action_thread = None
        if thread:
            thread.join(timeout=2.0)
        self._stop_event = threading.Event()
        if not keep_wheels:
            self.chassis.stop()

    def _base_targets(self, cfg):
        return int(cfg.speed + cfg.left_trim), int(cfg.speed + cfg.right_trim)

    def _forward_floor(self, target, floor_scale):
        if target > 0:
            return max(int(abs(target) * floor_scale), 1)
        if target < 0:
            return -max(int(abs(target) * floor_scale), 1)
        return 0

    def _build_arc_targets(self, lane_left, lane_right, steer_delta, direction):
        cfg = self.lane_change_config
        min_left = self._forward_floor(lane_left, cfg.min_turn_speed_scale)
        min_right = self._forward_floor(lane_right, cfg.min_turn_speed_scale)
        return_delta = max(1, int(steer_delta * cfg.return_steer_scale))

        right_phase_1_left = lane_left + steer_delta
        right_phase_1_right = max(min_right, lane_right - steer_delta)
        right_phase_2_left = max(min_left, lane_left - return_delta)
        right_phase_2_right = lane_right + return_delta

        if direction == "left":
            return (
                right_phase_1_right,
                right_phase_1_left,
                right_phase_2_right,
                right_phase_2_left,
            )

        return (
            right_phase_1_left,
            right_phase_1_right,
            right_phase_2_left,
            right_phase_2_right,
        )

    def drive_forward(self):
        self._cancel_active_action(keep_wheels=True)
        base_left, base_right = self._base_targets(self.cruise_config)
        self._set_state("forward", "running")
        self.chassis.ramp_to(base_left, base_right, self.cruise_config.ramp_time)
        self._set_state("forward", "cruising")

    def stop(self):
        self._cancel_active_action(keep_wheels=False)
        self._set_state("stopped", "idle")

    def lane_left(self):
        self._start_lane_change("left")

    def lane_right(self):
        self._start_lane_change("right")

    def _start_lane_change(self, direction):
        self._cancel_active_action(keep_wheels=True)
        self._set_state(f"lane-{direction}", "queued")
        thread = threading.Thread(target=self._lane_change_sequence, args=(direction,), daemon=True)
        with self._action_lock:
            self._action_thread = thread
        thread.start()

    def _lane_change_sequence(self, direction):
        cfg = self.lane_change_config
        stop_event = self._stop_event
        action_name = f"lane-{direction}"
        self._set_state(action_name, "running")

        base_left, base_right = self._base_targets(cfg)
        brake_left = int(base_left * cfg.brake_speed_scale)
        brake_right = int(base_right * cfg.brake_speed_scale)
        lane_base_left = int(base_left * cfg.lane_speed_scale)
        lane_base_right = int(base_right * cfg.lane_speed_scale)
        phase_1_left, phase_1_right, phase_2_left, phase_2_right = self._build_arc_targets(
            lane_base_left,
            lane_base_right,
            cfg.steer_delta,
            direction,
        )

        self.chassis.ramp_to(base_left, base_right, cfg.ramp_time, stop_event)
        self.chassis.ramp_to(brake_left, brake_right, cfg.lane_transition_time, stop_event)
        if not self.chassis.hold(cfg.brake_time, stop_event):
            return
        if not self.chassis.hold(cfg.pre_lane_time, stop_event):
            return
        self.chassis.ramp_to(phase_1_left, phase_1_right, cfg.lane_transition_time, stop_event)
        if not self.chassis.hold(cfg.turn_time, stop_event):
            return
        self.chassis.ramp_to(phase_2_left, phase_2_right, cfg.lane_transition_time, stop_event)
        if not self.chassis.hold(cfg.return_time, stop_event):
            return
        self.chassis.ramp_to(brake_left, brake_right, cfg.lane_transition_time, stop_event)
        if not self.chassis.hold(cfg.post_brake_time, stop_event):
            return
        self.chassis.ramp_to(base_left, base_right, cfg.recover_transition_time, stop_event)
        if not self.chassis.hold(cfg.settle_time, stop_event):
            return
        self._set_state(action_name, "completed")
        self._set_state("forward", "cruising")

    def execute(self, action):
        actions = {
            "forward": self.drive_forward,
            "stop": self.stop,
            "lane-left": self.lane_left,
            "lane-right": self.lane_right,
        }
        if action not in actions:
            raise ValueError(f"unsupported action: {action}")
        actions[action]()
