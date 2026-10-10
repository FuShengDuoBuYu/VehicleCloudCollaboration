"""Short causal smoothing of road support, never filling a current exclusion.

An asymmetric EMA suppresses transient road expansion. Current road/lane
rejections take effect immediately, so this cannot hide a new mask hole or
repair a false negative. Abrupt scene changes and capture gaps reset history.
Only distinct consumed model observations advance the filter.
"""
import math

import numpy as np


class SemanticStabilizer:
    def __init__(self, enabled=True, time_constant_seconds=.12,
                 max_gap_seconds=.25, threshold=.65, reset_iou=.8):
        numbers = (time_constant_seconds, max_gap_seconds, threshold, reset_iou)
        if (type(enabled) is not bool or not all(math.isfinite(v) for v in numbers)
                or not 0 < time_constant_seconds <= .3
                or not 0 < max_gap_seconds <= .3
                or not .5 <= threshold < 1 or not 0 < reset_iou < 1):
            raise ValueError('invalid semantic stabilization parameters')
        self.enabled = enabled
        self.tau = float(time_constant_seconds)
        self.max_gap = float(max_gap_seconds)
        self.threshold = float(threshold)
        self.reset_iou = float(reset_iou)
        self.reset()

    def reset(self):
        self._probability = self._raw = self._output = None
        self._sequence = self._captured_at = None
        self.state = {'mode': 'reset', 'changed_ratio': 0., 'raw_iou': None}

    def update(self, corridor, sequence, captured_at):
        raw = (np.asarray(corridor) > 0)
        if raw.ndim != 2 or not math.isfinite(captured_at):
            raise ValueError('stabilization requires a 2D mask and finite capture time')
        if not self.enabled:
            self.state = {'mode': 'disabled', 'changed_ratio': 0., 'raw_iou': None}
            return raw.astype(np.uint8)
        if self._sequence == sequence and self._output is not None:
            return (self._output & raw).astype(np.uint8)
        restart = self._raw is None or self._raw.shape != raw.shape
        iou = None
        if not restart:
            dt = captured_at - self._captured_at
            union = np.count_nonzero(raw | self._raw)
            iou = float(np.count_nonzero(raw & self._raw) / max(1, union))
            restart = dt <= 0 or dt > self.max_gap or iou < self.reset_iou
        if restart:
            self._probability = raw.astype(np.float32)
        else:
            alpha = 1. - math.exp(-dt / self.tau)
            self._probability += alpha * (raw.astype(np.float32) - self._probability)
        self._output = raw & (self._probability >= self.threshold)
        self._raw = raw.copy()
        self._sequence, self._captured_at = sequence, captured_at
        self.state = {'mode': 'initialized' if restart else 'filtered',
                      'changed_ratio': float(np.mean(raw != self._output)), 'raw_iou': iou}
        return self._output.astype(np.uint8)
