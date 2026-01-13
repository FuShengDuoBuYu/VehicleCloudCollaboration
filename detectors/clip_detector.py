# CLIP-based semantic detector for long-tail scenarios

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any
import numpy as np

from .base_detector import BaseDetector


class CLIPDetector(BaseDetector):
    """Uses CLIP to detect long-tail scenarios via semantic analysis"""
    
    DEFAULT_LABELS = [
        "normal road driving scene",
        "ordinary street scene",
        "highway driving scene",
        "pedestrian", "car", "truck", "bus", "motorcycle", "bicycle",
        "traffic cone", "traffic bollard", "road barrier", "barricade",
        "construction sign", "detour sign", "lane closed sign",
        "temporary traffic sign", "warning sign", "arrow board", "road work",
        "traffic light", "stop sign", "speed limit sign",
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_name = self.config.get('model_name', 'openai/clip-vit-base-patch32')
        self.labels = self.config.get('labels', self.DEFAULT_LABELS)
        self.max_prob_threshold = self.config.get('max_prob_threshold', 0.22)
        self.entropy_threshold = self.config.get('entropy_threshold', 0.72)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()
    
    def _calculate_entropy(self, probs: torch.Tensor) -> float:
        eps = 1e-8
        entropy = float(-(probs * (probs + eps).log()).sum().item())
        max_entropy = np.log(len(probs))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def detect(self, image_path: str) -> float:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=self.labels, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0].float()
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max().item()
        norm_entropy = self._calculate_entropy(probs)
        if norm_entropy > self.entropy_threshold:
            score = min(1.0, norm_entropy)
        elif max_prob < self.max_prob_threshold:
            score = 1.0 - max_prob
        else:
            score = norm_entropy * 0.5
        return float(np.clip(score, 0.0, 1.0))
