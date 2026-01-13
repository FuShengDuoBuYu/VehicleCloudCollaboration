# YOLO-World detector for long-tail scenarios

from typing import Dict, Any
import numpy as np

from .base_detector import BaseDetector


class YOLOWorldDetector(BaseDetector):
    """Uses YOLO-World to detect long-tail objects"""
    
    LONG_TAIL_CLASSES = [
        "traffic cone", "traffic bollard", "road barrier", "traffic barrier",
        "barricade", "construction sign", "road work sign", "detour sign",
        "lane closed sign", "temporary sign", "arrow board", "warning sign",
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        try:
            from ultralytics import YOLO
            model_path = self.config.get('model_path', 'yolov8x-worldv2.pt')
            self.conf_threshold = self.config.get('conf_threshold', 0.20)
            self.model = YOLO(model_path)
            all_classes = self.config.get('classes', [
                "person", "car", "truck", "bus", "motorcycle", "bicycle", "scooter",
            ] + self.LONG_TAIL_CLASSES + [
                "traffic light", "stop sign", "speed limit sign",
                "pedestrian crossing sign", "no entry sign", "yield sign",
            ])
            self.model.set_classes(all_classes)
            self.long_tail_set = set(self.LONG_TAIL_CLASSES)
        except ImportError:
            print("Warning: ultralytics not available, YOLOWorldDetector will return 0.5")
            self.model = None
    
    def detect(self, image_path: str) -> float:
        if self.model is None:
            return 0.5
        results = self.model.predict(source=image_path, imgsz=640, conf=self.conf_threshold, verbose=False)
        if len(results) == 0:
            return 0.0
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return 0.0
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        long_tail_count = 0
        long_tail_conf_sum = 0.0
        total_objects = len(classes)
        for cls_id, conf in zip(classes, confidences):
            cls_name = result.names.get(int(cls_id), "").lower()
            if any(lt_class in cls_name for lt_class in self.long_tail_set):
                long_tail_count += 1
                long_tail_conf_sum += conf
        if long_tail_count == 0:
            return 0.0
        ratio_score = min(1.0, long_tail_count / max(1, total_objects))
        conf_score = long_tail_conf_sum / long_tail_count if long_tail_count > 0 else 0.0
        score = 0.6 * ratio_score + 0.4 * conf_score
        return float(np.clip(score, 0.0, 1.0))
