# Base detector interface for long-tail scene detection
# All detectors should inherit from this class and implement the detect() method

from abc import ABC, abstractmethod
from typing import Dict, Any
import time


class BaseDetector(ABC):
    """
    Base class for all detectors
    Each detector returns a score in [0, 1] indicating the likelihood of long-tail scenario
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.inference_time = 0.0
    
    @abstractmethod
    def detect(self, image_path: str) -> float:
        """
        Detect if the image is a long-tail scenario
        Returns a float in [0, 1]
        """
        pass
    
    def __call__(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        score = self.detect(image_path)
        self.inference_time = time.time() - start_time
        return {
            'detector': self.name,
            'score': float(score),
            'inference_time': self.inference_time
        }
    
    def get_fps(self) -> float:
        if self.inference_time > 0:
            return 1.0 / self.inference_time
        return 0.0
import time


class BaseDetector(ABC):
    """
    Base class for all detectors
    Each detector returns a score in [0, 1] indicating the likelihood of long-tail scenario
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.inference_time = 0.0
    
    @abstractmethod
    def detect(self, image_path: str) -> float:
        """
        Detect if the image is a long-tail scenario
        Returns a float in [0, 1]
        """
        pass
    
    def __call__(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        score = self.detect(image_path)
        self.inference_time = time.time() - start_time
        return {
            'detector': self.name,
            'score': float(score),
            'inference_time': self.inference_time
        }
    
    def get_fps(self) -> float:
        if self.inference_time > 0:
            return 1.0 / self.inference_time
        return 0.0
