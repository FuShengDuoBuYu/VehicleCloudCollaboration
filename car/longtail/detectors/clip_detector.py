# CLIP-based semantic detector for long-tail scenarios

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any
import numpy as np

from .base_detector import BaseDetector


class CLIPDetector(BaseDetector):
    """Uses CLIP to detect long-tail scenarios via semantic analysis"""
    
    # Labels are arranged to highlight long-tail scenarios first, especially cones
    DEFAULT_LABELS = [
        # Long-tail: construction with cones (most typical)
        "road construction with traffic cones",
        "traffic cones on road",
        "construction zone cones",
        "cones blocking lane",
        "orange safety cones",
        "temporary cones in lane",
        "work zone with cones",
        # Long-tail: visibility/clarity issues
        "heavy fog low visibility",
        "sudden fog on road",
        "blurry image scene",
        "motion blur driving scene",
        "camera defocus road scene",
        # Long-tail: traffic disorder/accident
        "chaotic traffic scene",
        "traffic accident scene",
        "multiple vehicles disorderly",
        "wrong-way driving detected",
        # Long-tail: temporary/barricades
        "road closure barricade",
        "temporary detour sign",
        "lane closed sign",
        "arrow board road work",
        # Normal baselines
        "normal road driving scene",
        "ordinary street scene",
        "highway driving scene",
        "clear weather good visibility",
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
        top_idx = int(probs.argmax().item())
        top_label = self.labels[top_idx].lower()
        norm_entropy = self._calculate_entropy(probs)
        # Base scoring from uncertainty
        if norm_entropy > self.entropy_threshold:
            score = min(1.0, norm_entropy)
        elif max_prob < self.max_prob_threshold:
            score = 1.0 - max_prob
        else:
            score = norm_entropy * 0.5
        # Explicit boost for typical long-tail cues (including traffic cones)
        long_tail_keywords = [
            "cone", "traffic cone", "orange cone", "construction", "work zone", 
            "barricade", "fog", "low visibility", "blurry", "motion blur", 
            "accident", "chaotic", "wrong-way", "detour", "lane closed"
        ]
        
        if any(k in top_label for k in long_tail_keywords):
            score = max(score, 0.90)  # Unified high score for all long-tail scenarios
            
        return float(np.clip(score, 0.0, 1.0))
