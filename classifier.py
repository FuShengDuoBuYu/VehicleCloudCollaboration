# Long-tail scenario classifier with weighted ensemble of detectors

from typing import Dict, List, Any
import time

from detectors import (
    CLIPDetector,
    YOLOWorldDetector,
    YOLOv8Detector,
    YOLOPv2Detector,
    # YOLOv8MultiTaskDetector,
)


class LongTailClassifier:
    """Ensemble classifier combining multiple detectors"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
        self.detectors: List[tuple] = []  # (detector, weight)
        self._initialize_detectors()
        self.total_inference_time = 0.0
        self.last_results = None
    
    def _initialize_detectors(self):
        detector_configs = self.config.get('detectors', [])
        if not detector_configs:
            detector_configs = [
                {'type': 'clip', 'weight': 0.35, 'config': {}},
                {'type': 'yoloworld', 'weight': 0.30, 'config': {}},
                {'type': 'yolov8', 'weight': 0.20, 'config': {}},
                {'type': 'yolopv2', 'weight': 0.15, 'config': {}},
                # {'type': 'yolov8m', 'weight': 0.4, 'config': {}},
            ]
        detector_map = {
            'clip': CLIPDetector,
            'yoloworld': YOLOWorldDetector,
            'yolov8': YOLOv8Detector,
            'yolopv2': YOLOPv2Detector,
            # 'yolov8m': YOLOv8MultiTaskDetector,
        }
        for det_config in detector_configs:
            det_type = det_config.get('type', '').lower()
            weight = det_config.get('weight', 0.25)
            det_specific_config = det_config.get('config', {})
            if det_type in detector_map:
                try:
                    detector = detector_map[det_type](det_specific_config)
                    self.detectors.append((detector, weight))
                    print(f"Initialized {detector.__class__.__name__} with weight {weight}")
                except Exception as e:
                    print(f"Warning: Failed to initialize {det_type}: {e}")
        self._normalize_weights()
    
    def _normalize_weights(self):
        if not self.detectors:
            return
        total_weight = sum(weight for _, weight in self.detectors)
        if total_weight > 0:
            self.detectors = [ (detector, weight / total_weight) for detector, weight in self.detectors ]
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        start_time = time.time()
        individual_results = []
        weighted_score = 0.0
        for detector, weight in self.detectors:
            try:
                result = detector(image_path)
                individual_results.append(result)
                weighted_score += result['score'] * weight
            except Exception as e:
                print(f"Warning: {detector.__class__.__name__} failed: {e}")
                individual_results.append({
                    'detector': detector.__class__.__name__,
                    'score': 0.5,
                    'inference_time': 0.0,
                    'error': str(e)
                })
        self.total_inference_time = time.time() - start_time
        is_long_tail = weighted_score >= self.threshold
        result = {
            'is_long_tail': bool(is_long_tail),
            'score': float(weighted_score),
            'threshold': self.threshold,
            'individual_scores': individual_results,
            'inference_time': self.total_inference_time,
            'fps': (1.0 / self.total_inference_time) if self.total_inference_time > 0 else 0.0,
        }
        self.last_results = result
        return result
    
    def print_summary(self):
        if self.last_results is None:
            print("No predictions made yet")
            return
        print("\n" + "="*70)
        print("LONG-TAIL SCENARIO CLASSIFICATION SUMMARY")
        print("="*70)
        print(f"Result: {'LONG-TAIL ⚠️' if self.last_results['is_long_tail'] else 'NORMAL ✓'}")
        print(f"Confidence Score: {self.last_results['score']:.3f}")
        print(f"Threshold: {self.last_results['threshold']:.3f}")
        print(f"\nTotal Inference Time: {self.last_results['inference_time']:.3f}s")
        print(f"Overall FPS: {self.last_results['fps']:.2f}")
        print("\n" + "-"*70)
        print("Individual Detector Results:")
        print("-"*70)
        for result in self.last_results['individual_scores']:
            detector_name = result['detector']
            score = result['score']
            inf_time = result.get('inference_time', 0)
            fps = 1.0 / inf_time if inf_time > 0 else 0
            weight = 0.0
            for det, w in self.detectors:
                if det.__class__.__name__ == detector_name:
                    weight = w
                    break
            print(f"{detector_name:<20} | Score: {score:.3f} | Weight: {weight:.2f} | Time: {inf_time:.3f}s | FPS: {fps:.2f}")
        print("="*70 + "\n")
