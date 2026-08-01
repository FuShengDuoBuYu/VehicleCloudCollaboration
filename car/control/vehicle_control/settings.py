from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 20
    jpeg_quality: int = 80


CAMERA_CONFIG = CameraConfig()
