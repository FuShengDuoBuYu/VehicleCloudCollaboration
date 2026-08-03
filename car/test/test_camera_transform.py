#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


CAR_DIR = Path(__file__).resolve().parents[1]
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.camera.transform import CameraTransformConfig, transform_frame


class CameraTransformTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    def test_rotation_is_clockwise(self):
        transformed = transform_frame(
            self.frame,
            CameraTransformConfig(rotation_degrees=90),
        )
        np.testing.assert_array_equal(
            transformed,
            np.array([[4, 1], [5, 2], [6, 3]], dtype=np.uint8),
        )

    def test_horizontal_flip(self):
        transformed = transform_frame(
            self.frame,
            CameraTransformConfig(flip_horizontal=True),
        )
        np.testing.assert_array_equal(
            transformed,
            np.array([[3, 2, 1], [6, 5, 4]], dtype=np.uint8),
        )

    def test_invalid_rotation_is_rejected(self):
        with self.assertRaises(ValueError):
            CameraTransformConfig(rotation_degrees=45)


if __name__ == "__main__":
    unittest.main()
