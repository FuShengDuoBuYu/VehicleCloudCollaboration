"""Perspective calibration for camera-view-independent lane geometry."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..control.lane_centering import LaneEstimate


def camera_pose_from_mapping(camera_config):
    """Return the image geometry and PTZ pose that define a calibration."""
    camera = camera_config or {}
    rotation = int(camera.get("rotation_degrees", 0))
    if rotation not in (0, 90, 180, 270):
        raise ValueError("camera.rotation_degrees must be 0, 90, 180, or 270")
    capture_width = int(camera.get("width", 640))
    capture_height = int(camera.get("height", 480))
    if capture_width < 1 or capture_height < 1:
        raise ValueError("camera width and height must be positive")
    if rotation in (90, 270):
        image_width, image_height = capture_height, capture_width
    else:
        image_width, image_height = capture_width, capture_height
    gimbal = camera.get("gimbal") or {}
    return {
        "image_width": image_width,
        "image_height": image_height,
        "rotation_degrees": rotation,
        "flip_horizontal": bool(camera.get("flip_horizontal", False)),
        "flip_vertical": bool(camera.get("flip_vertical", False)),
        "gimbal_initialize_on_startup": bool(
            gimbal.get("initialize_on_startup", False)
        ),
        "pan_angle": (
            None if gimbal.get("pan_angle") is None else int(gimbal["pan_angle"])
        ),
        "tilt_angle": (
            None if gimbal.get("tilt_angle") is None else int(gimbal["tilt_angle"])
        ),
    }


def validate_calibration_camera_pose(calibration_path, camera_config):
    """Reject a calibration captured with different image/PTZ geometry."""
    path = Path(calibration_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not data.get("calibrated", False):
        raise ValueError(f"perspective calibration is not marked calibrated: {path}")
    actual = data.get("camera_pose")
    if not isinstance(actual, dict):
        raise ValueError(
            "perspective calibration has no camera_pose metadata; recapture "
            "and recalibrate after fixing the camera gimbal"
        )
    expected = camera_pose_from_mapping(camera_config)
    mismatches = []
    for key, expected_value in expected.items():
        if key not in actual or actual[key] != expected_value:
            mismatches.append(
                f"{key}: calibration={actual.get(key)!r}, runtime={expected_value!r}"
            )
    if mismatches:
        raise ValueError(
            "perspective calibration does not match the runtime camera pose: "
            + "; ".join(mismatches)
        )
    return data


class PerspectiveMapper:
    """Warp masks into bird's-eye view and map debug geometry back to camera view."""

    def __init__(self, source_points, destination_points):
        self.source_points = self._validate_points(source_points, "source_points")
        self.destination_points = self._validate_points(
            destination_points, "destination_points"
        )

    @staticmethod
    def _validate_points(points, name):
        array = np.asarray(points, dtype=np.float32)
        if array.shape != (4, 2):
            raise ValueError(f"{name} must contain four [x, y] points")
        if not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"{name} must use normalized coordinates in [0, 1]")
        return array

    @classmethod
    def from_yaml(cls, path):
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not data.get("calibrated", False):
            raise ValueError(
                f"perspective calibration is not marked calibrated: {path}"
            )
        return cls(data.get("source_points"), data.get("destination_points"))

    def _matrices(self, shape):
        height, width = shape[:2]
        scale = np.array([max(1, width - 1), max(1, height - 1)], dtype=np.float32)
        source = self.source_points * scale
        destination = self.destination_points * scale
        matrix = cv2.getPerspectiveTransform(source, destination)
        inverse = cv2.getPerspectiveTransform(destination, source)
        return matrix, inverse

    def warp_mask(self, mask):
        array = np.asarray(mask)
        matrix, _ = self._matrices(array.shape)
        height, width = array.shape[:2]
        return cv2.warpPerspective(
            array,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def camera_mask(self, mask):
        """Map a bird's-eye binary mask back into the camera image plane."""
        array = np.asarray(mask)
        _, inverse = self._matrices(array.shape)
        height, width = array.shape[:2]
        return cv2.warpPerspective(
            array,
            inverse,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def camera_points(self, points, shape):
        array = np.asarray(points)
        if array.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        _, inverse = self._matrices(shape)
        mapped = cv2.perspectiveTransform(
            array.astype(np.float32).reshape(-1, 1, 2), inverse
        ).reshape(-1, 2)
        height, width = shape[:2]
        mapped[:, 0] = np.clip(mapped[:, 0], 0, width - 1)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, height - 1)
        return np.rint(mapped).astype(np.int32)

    def camera_estimate(self, estimate: LaneEstimate, shape) -> LaneEstimate:
        lookahead = None
        if estimate.lookahead_point is not None:
            mapped = self.camera_points(
                np.asarray([estimate.lookahead_point]), shape
            )
            lookahead = (int(mapped[0, 0]), int(mapped[0, 1]))
        return replace(
            estimate,
            lookahead_point=lookahead,
            centerline=self.camera_points(estimate.centerline, shape),
            left_boundary=self.camera_points(estimate.left_boundary, shape),
            right_boundary=self.camera_points(estimate.right_boundary, shape),
        )
