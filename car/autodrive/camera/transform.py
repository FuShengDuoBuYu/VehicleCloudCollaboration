"""Deterministic image-orientation transforms for onboard camera frames."""

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraTransformConfig:
    """Correct sensor roll or mirroring without pretending to correct camera yaw."""

    rotation_degrees: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def __post_init__(self):
        if self.rotation_degrees not in {0, 90, 180, 270}:
            raise ValueError("camera rotation_degrees must be one of 0, 90, 180, 270")

    @classmethod
    def from_mapping(cls, value):
        value = value or {}
        return cls(
            rotation_degrees=int(value.get("rotation_degrees", 0)),
            flip_horizontal=bool(value.get("flip_horizontal", False)),
            flip_vertical=bool(value.get("flip_vertical", False)),
        )


def transform_frame(frame, config: CameraTransformConfig):
    """Return a transformed frame; a zero transform returns the original object."""
    if frame is None:
        return None
    output = frame
    rotation_codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if config.rotation_degrees:
        output = cv2.rotate(output, rotation_codes[config.rotation_degrees])
    if config.flip_horizontal and config.flip_vertical:
        output = cv2.flip(output, -1)
    elif config.flip_horizontal:
        output = cv2.flip(output, 1)
    elif config.flip_vertical:
        output = cv2.flip(output, 0)
    return output
