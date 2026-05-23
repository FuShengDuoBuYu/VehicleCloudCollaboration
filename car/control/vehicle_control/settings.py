from dataclasses import dataclass


@dataclass(frozen=True)
class CruiseConfig:
    speed: int = 18
    left_trim: int = 10
    right_trim: int = 0
    ramp_time: float = 0.25


@dataclass(frozen=True)
class LaneChangeConfig:
    speed: int = 22
    left_trim: int = 10
    right_trim: int = 0
    ramp_time: float = 0.25
    approach_time: float = 0.55
    steer_delta: int = 18
    brake_speed_scale: float = 0.05
    brake_time: float = 0.25
    lane_speed_scale: float = 0.70
    min_turn_speed_scale: float = 0.6
    return_steer_scale: float = 1
    pre_lane_time: float = 0.05
    turn_time: float = 1.05
    return_time: float = 0.75
    settle_time: float = 0.6
    lane_transition_time: float = 0.80
    recover_transition_time: float = 0.45
    post_brake_time: float = 0.15


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 20
    jpeg_quality: int = 80


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080


CRUISE_CONFIG = CruiseConfig()
LANE_CHANGE_CONFIG = LaneChangeConfig()
CAMERA_CONFIG = CameraConfig()
SERVER_CONFIG = ServerConfig()
