# Base detector interface for long-tail scene detection

from abc import ABC, abstractmethod
import time
from typing import Any, Dict


class BaseDetector(ABC):
    """Base class for detectors that score long-tail likelihood in [0, 1]."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.inference_time = 0.0

    @abstractmethod
    def detect(self, image_path: str) -> float:
        """Return a long-tail likelihood score in [0, 1]."""
        raise NotImplementedError

    def __call__(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        score = float(self.detect(image_path))
        if score < 0.0 or score > 1.0:
            raise ValueError(f"{self.name} returned invalid score: {score}")
        self.inference_time = time.time() - start_time
        return {
            "detector": self.name,
            "score": score,
            "inference_time": self.inference_time,
        }

    def get_fps(self) -> float:
        if self.inference_time > 0:
            return 1.0 / self.inference_time
        return 0.0
