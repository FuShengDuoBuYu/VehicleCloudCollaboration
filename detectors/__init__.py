"""
Long-tail scene detection modules
"""

from .base_detector import BaseDetector
from .clip_detector import CLIPDetector
from .yoloworld_detector import YOLOWorldDetector
from .yolov8_detector import YOLOv8Detector
from .yolopv2_detector import YOLOPv2Detector
from .yolov8m_detector import YOLOv8MultiTaskDetector

__all__ = [
    'BaseDetector',
    'CLIPDetector',
    'YOLOWorldDetector',
    'YOLOv8Detector',
    'YOLOPv2Detector',
    #'YOLOv8MultiTaskDetector',
]
