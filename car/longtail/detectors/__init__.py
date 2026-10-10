"""
Long-tail scene detection modules
"""

from .base_detector import BaseDetector


def __getattr__(name):
    # Import only the selected model; a YOLOPv2 deployment needs no CLIP/YOLOv8.
    from importlib import import_module
    modules = {"CLIPDetector": "clip_detector", "YOLOv8Detector": "yolov8_detector",
               "YOLOPv2Detector": "yolopv2_detector"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module("." + modules[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    'BaseDetector',
    'CLIPDetector',
    'YOLOv8Detector',
    'YOLOPv2Detector',
]
