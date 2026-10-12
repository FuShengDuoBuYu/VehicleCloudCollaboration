"""Compatibility facade for established wheel-control imports.

New code imports wheel_driver, motion_gate, corner_continuation or watchdog.
No hardware opens at import time.
"""
import time
from .wheel_driver import WheelMappingConfig, SafeWheelDriver
from .motion_gate import PerceptionMotionGate
from .corner_continuation import CornerContinuationConfig, CornerContinuationGate
from .watchdog import CommandWatchdog

__all__ = ['WheelMappingConfig', 'SafeWheelDriver', 'PerceptionMotionGate',
           'CornerContinuationConfig', 'CornerContinuationGate', 'CommandWatchdog']
