import importlib
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ObjectDecodeTests(unittest.TestCase):
    def decode(self, heads, anchors, **kwargs):
        module = importlib.import_module('autodrive.perception.yolopv2_objects')
        return module.decode_objects(heads, anchors, network_shape=(32, 32),
                                     padding=(0, 0, 0, 0), **kwargs)

    def fixture(self):
        heads = [np.full((1, 255, 1, 1), -20., np.float32) for _ in range(3)]
        anchors = [np.full((1, 3, 1, 1, 2), 4., np.float32) for _ in range(3)]
        return heads, anchors

    def test_decodes_stride_anchor_and_confidence_and_deduplicates(self):
        heads, anchors = self.fixture()
        for offset in (0, 85):
            heads[0][0, offset:offset+4] = 0
            heads[0][0, offset+4] = 4
            heads[0][0, offset+5] = 4
        objects = self.decode(heads, anchors)
        self.assertEqual(len(objects), 1)
        np.testing.assert_allclose(objects[0]['box'], [.0625, .0625, .1875, .1875])
        self.assertEqual(objects[0]['class_id'], 0)
        self.assertAlmostEqual(objects[0]['confidence'], .96435, places=4)

    def test_no_detection_stays_empty(self):
        self.assertEqual(self.decode(*self.fixture()), [])

    def test_nonfinite_head_fails_instead_of_silent_empty_detections(self):
        heads, anchors = self.fixture();heads[1][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):self.decode(heads, anchors)


if __name__ == '__main__':unittest.main()
