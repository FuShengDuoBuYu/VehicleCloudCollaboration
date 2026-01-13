# YOLOPv2 detector for road scene analysis (heuristic-based)

from typing import Dict, Any
import numpy as np
import cv2

from .base_detector import BaseDetector


class YOLOPv2Detector(BaseDetector):
    """Heuristic-based analysis for unusual scenes"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.weights_path = self.config.get('weights_path', 'YOLOPv2/data/weights/yolopv2.pt')
        self.img_size = self.config.get('img_size', 640)
        self.conf_threshold = self.config.get('conf_threshold', 0.3)
        self.use_full_model = self.config.get('use_full_model', False)
        self.model = None
    
    def _analyze_image_heuristics(self, image_path: str) -> float:
        img = cv2.imread(image_path)
        if img is None:
            return 0.5
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(gray)
        brightness_score = 0.0
        if brightness < 50 or brightness > 200:
            brightness_score = 0.4
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_ratio = np.sum(orange_mask > 0) / orange_mask.size
        orange_score = min(1.0, orange_ratio * 20)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = 0.0
        if edge_density > 0.15 or edge_density < 0.03:
            edge_score = 0.3
        saturation = hsv[:, :, 1]
        sat_std = np.std(saturation)
        sat_score = 0.0
        if sat_std < 20:
            sat_score = 0.3
        total_score = 0.25 * brightness_score + 0.35 * orange_score + 0.25 * edge_score + 0.15 * sat_score
        return float(np.clip(total_score, 0.0, 1.0))
    
    def detect(self, image_path: str) -> float:
        return self._analyze_image_heuristics(image_path)
