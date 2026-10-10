"""CPU stored-mask analysis: rendering must not precede control execution."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autodrive.control.lane_centering import LaneEstimate, DifferentialDriveCommand
from autodrive.runtime.onboard import analyze


class RenderTimingTests(unittest.TestCase):
    def test_expensive_image_render_can_be_deferred_until_after_control(self):
        road=np.ones((24,32),np.uint8)
        detector=SimpleNamespace(predict_masks=lambda frame:(road,np.zeros_like(road)))
        estimator=SimpleNamespace(estimate=lambda *a:LaneEstimate(True,.9))
        controller=SimpleNamespace(update=lambda *a:DifferentialDriveCommand('forward',0.,.3,.3,.9))
        jobs=[]
        with patch('autodrive.runtime.onboard.render_debug_frame',return_value='rendered') as renderer:
            est,command,annotated,*_=analyze(detector,estimator,controller,None,
                np.zeros((48,64,3),np.uint8),'center',.1,.25,deferred_render=jobs.append)
            self.assertTrue(est.valid)
            self.assertEqual(command.action,'forward')
            self.assertIsNone(annotated)
            renderer.assert_not_called()
            self.assertEqual(len(jobs),1)
            self.assertEqual(jobs[0](),'rendered')
            renderer.assert_called_once()


if __name__=='__main__':unittest.main()
