import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longtail.detectors.yolopv2_detector import YOLOPv2Detector


class TorchInputTests(unittest.TestCase):
    def test_matching_thread_pool_is_not_reconfigured(self):
        from autodrive.perception.yolopv2_fusion import YOLOPv2FusionConfig, YOLOPv2FusionDetector
        with tempfile.NamedTemporaryFile() as weights:
            config = YOLOPv2FusionConfig(weights=weights.name, asynchronous=False,
                                         torch_num_threads=0, torch_interop_threads=torch.get_num_interop_threads())
            with patch('longtail.detectors.yolopv2_detector.YOLOPv2Detector'), patch('torch.set_num_interop_threads') as setter:
                first = YOLOPv2FusionDetector(config); first.close()
                second = YOLOPv2FusionDetector(config); second.close()
                setter.assert_not_called()

    def test_camera_aspect_and_rgb_order_are_preserved(self):
        detector = YOLOPv2Detector({'img_size': 640, 'preserve_aspect_ratio': True})
        image = np.zeros((480, 640, 3), np.uint8); image[:, :, 0] = 255
        tensor, padding = detector._preprocess_image(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, 480, 640))
        self.assertEqual(padding, (0, 0, 0, 0))
        self.assertEqual(float(tensor[0, 0, 0, 0]), 0.)
        self.assertEqual(float(tensor[0, 2, 0, 0]), 1.)

    def test_preserve_aspect_rejects_legacy_fixed_crop(self):
        with self.assertRaises(ValueError):
            YOLOPv2Detector({'preserve_aspect_ratio': True, 'fast_mask': False})

    def test_half_cpu_is_rejected_before_model_load(self):
        with self.assertRaises(ValueError):
            YOLOPv2Detector({'device': 'cpu', 'half': True})

    def test_nonfinite_segmentation_is_not_converted_to_valid_mask(self):
        detector = YOLOPv2Detector({'img_size': 64, 'preserve_aspect_ratio': True})
        class InvalidModel:
            def __call__(self, value):
                return None, torch.full((1, 2, 64, 64), float('nan')), torch.zeros(1, 1, 64, 64)
        detector.model = InvalidModel()
        with self.assertRaises(ValueError):detector.predict_masks(np.zeros((64, 64, 3), np.uint8))


if __name__ == '__main__':unittest.main()
