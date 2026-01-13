# YOLOv8 detector for traffic signs and unusual objects

from typing import Dict, Any
import numpy as np

from .base_detector import BaseDetector


class YOLOv8Detector(BaseDetector):
    """Uses YOLOv8 to detect traffic signs and unusual objects"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        try:
            from ultralytics import YOLO
            model_path = self.config.get('model_path', 'yolov8n/yolov8n.pt')
            self.conf_threshold = self.config.get('conf_threshold', 0.25)
            self.model = YOLO(model_path)
            self.unusual_indicators = self.config.get('unusual_indicators', [
                'stop sign', 'parking meter', 'fire hydrant',
            ])
        except ImportError:
            print("Warning: ultralytics not available, YOLOv8Detector will return 0.5")
            self.model = None
    
    def detect(self, image_path: str) -> float:
        if self.model is None:
            return 0.5
        results = self.model.predict(source=image_path, conf=self.conf_threshold, verbose=False)
        if len(results) == 0:
            return 0.0
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return 0.3
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        total_objects = len(classes)
        unusual_count = 0
        avg_conf = np.mean(confidences) if len(confidences) > 0 else 0.0
        for cls_id in classes:
            cls_name = result.names.get(int(cls_id), "").lower()
            if any(indicator in cls_name for indicator in self.unusual_indicators):
                unusual_count += 1
        conf_score = 1.0 - avg_conf if avg_conf < 0.5 else 0.0
        unusual_score = min(1.0, unusual_count / max(1, total_objects))
        count_score = 0.0
        if total_objects < 2:
            count_score = 0.4
        elif total_objects > 20:
            count_score = 0.3
        score = 0.4 * conf_score + 0.4 * unusual_score + 0.2 * count_score
        return float(np.clip(score, 0.0, 1.0))
