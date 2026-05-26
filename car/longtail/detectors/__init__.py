"""
Long-tail scene detection modules
"""

from .base_detector import BaseDetector
from .clip_detector import CLIPDetector
from .yolov8_detector import YOLOv8Detector
from .yolopv2_detector import YOLOPv2Detector

__all__ = [
    'BaseDetector',
    'CLIPDetector',
    'YOLOv8Detector',
    'YOLOPv2Detector',
]
