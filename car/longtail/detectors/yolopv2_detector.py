# YOLOPv2 detector for road scene analysis

from typing import Dict, Any
import numpy as np
import cv2
import torch

from .base_detector import BaseDetector


class YOLOPv2Detector(BaseDetector):
    """Analyzes road geometry using YOLOPv2 drivable-area and lane masks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.weights_path = self.config.get('weights_path', 'YOLOPv2/data/weights/yolopv2.pt')
        self.img_size = self.config.get('img_size', 640)
        self.conf_threshold = self.config.get('conf_threshold', 0.3)
        self.use_full_model = self.config.get('use_full_model', False)
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.geometry_mode = self.config.get('geometry_mode', 'weighted')
        self.geometry_weights = self.config.get('geometry_weights', {})
        self.linear_coefficients = self.config.get('linear_coefficients', [])
        self.linear_intercept = float(self.config.get('linear_intercept', 0.0))
        self.linear_temperature = float(self.config.get('linear_temperature', 1.0))
        self.model = None
        if self.use_full_model:
            self.model = torch.jit.load(self.weights_path, map_location=self.device)
            self.model.to(self.device).eval()
    
    def _analyze_image_heuristics(self, image_path: str) -> float:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"failed to read image: {image_path}")
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

    def _letterbox(self, img: np.ndarray) -> np.ndarray:
        shape = img.shape[:2]
        new_shape = (self.img_size, self.img_size)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    def _preprocess_for_model(self, image_path: str) -> torch.Tensor:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"failed to read image: {image_path}")
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = self._letterbox(img)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(self.device).float()
        tensor /= 255.0
        return tensor.unsqueeze(0)

    def _predict_masks(self, image_path: str) -> tuple:
        tensor = self._preprocess_for_model(image_path)
        with torch.no_grad():
            _, seg, lane = self.model(tensor)
        da_predict = seg[:, :, 12:372, :]
        da_predict = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear')
        _, da_mask = torch.max(da_predict, 1)
        lane_predict = lane[:, :, 12:372, :]
        lane_predict = torch.nn.functional.interpolate(lane_predict, scale_factor=2, mode='bilinear')
        lane_mask = torch.round(lane_predict).squeeze(1)
        return (
            da_mask.int().squeeze().cpu().numpy().astype(np.uint8),
            lane_mask.int().squeeze().cpu().numpy().astype(np.uint8),
        )

    def _geometry_features(self, drivable_mask: np.ndarray, lane_mask: np.ndarray) -> Dict[str, float]:
        h, w = drivable_mask.shape
        lower_start = int(h * 0.42)
        lower = drivable_mask[lower_start:, :]
        lane_lower = lane_mask[lower_start:, :]
        center_band = lower[:, int(w * 0.38):int(w * 0.62)]
        near_band = drivable_mask[int(h * 0.72):, :]

        rows = []
        centers = []
        widths = []
        for y in range(lower_start, h, 6):
            xs = np.where(drivable_mask[y] > 0)[0]
            if len(xs) < 8:
                rows.append(y)
                centers.append(0.5)
                widths.append(0.0)
                continue
            left = float(xs[0]) / w
            right = float(xs[-1]) / w
            rows.append(y)
            centers.append((left + right) * 0.5)
            widths.append(right - left)

        widths_arr = np.array(widths, dtype=np.float32)
        centers_arr = np.array(centers, dtype=np.float32)
        valid = widths_arr > 0.02
        if np.count_nonzero(valid) >= 3:
            valid_widths = widths_arr[valid]
            valid_centers = centers_arr[valid]
            width_jump = float(np.percentile(np.abs(np.diff(valid_widths)), 95))
            center_jump = float(np.percentile(np.abs(np.diff(valid_centers)), 95))
            width_std = float(np.std(valid_widths))
        else:
            width_jump = 1.0
            center_jump = 1.0
            width_std = 1.0

        lane_ratio = float(np.mean(lane_mask > 0))
        lane_lower_ratio = float(np.mean(lane_lower > 0))
        drivable_ratio = float(np.mean(drivable_mask > 0))
        lower_drivable_ratio = float(np.mean(lower > 0))
        near_drivable_ratio = float(np.mean(near_band > 0))
        center_drivable_ratio = float(np.mean(center_band > 0))
        bottom_center_ratio = float(np.mean(drivable_mask[int(h * 0.82):, int(w * 0.40):int(w * 0.60)] > 0))

        return {
            "drivable_ratio": drivable_ratio,
            "lower_drivable_ratio": lower_drivable_ratio,
            "near_drivable_ratio": near_drivable_ratio,
            "center_drivable_ratio": center_drivable_ratio,
            "bottom_center_ratio": bottom_center_ratio,
            "lane_ratio": lane_ratio,
            "lane_lower_ratio": lane_lower_ratio,
            "width_jump": width_jump,
            "center_jump": center_jump,
            "width_std": width_std,
        }

    def _score_geometry(self, features: Dict[str, float]) -> float:
        if self.geometry_mode == "linear":
            return self._score_geometry_linear(features)

        weights = {
            "near_missing": self.geometry_weights.get("near_missing", 0.22),
            "center_missing": self.geometry_weights.get("center_missing", 0.18),
            "width_jump": self.geometry_weights.get("width_jump", 0.24),
            "center_jump": self.geometry_weights.get("center_jump", 0.16),
            "lane_anomaly": self.geometry_weights.get("lane_anomaly", 0.10),
            "width_std": self.geometry_weights.get("width_std", 0.10),
        }
        near_missing = np.clip((0.18 - features["near_drivable_ratio"]) / 0.18, 0.0, 1.0)
        center_missing = np.clip((0.16 - features["bottom_center_ratio"]) / 0.16, 0.0, 1.0)
        width_jump = np.clip(features["width_jump"] / 0.24, 0.0, 1.0)
        center_jump = np.clip(features["center_jump"] / 0.18, 0.0, 1.0)
        lane_anomaly = max(
            np.clip((0.008 - features["lane_lower_ratio"]) / 0.008, 0.0, 1.0),
            np.clip((features["lane_lower_ratio"] - 0.08) / 0.08, 0.0, 1.0),
        )
        width_std = np.clip(features["width_std"] / 0.28, 0.0, 1.0)
        score = (
            weights["near_missing"] * near_missing
            + weights["center_missing"] * center_missing
            + weights["width_jump"] * width_jump
            + weights["center_jump"] * center_jump
            + weights["lane_anomaly"] * lane_anomaly
            + weights["width_std"] * width_std
        )
        return float(np.clip(score, 0.0, 1.0))

    def _score_geometry_linear(self, features: Dict[str, float]) -> float:
        values = np.array(
            [
                features["near_drivable_ratio"],
                features["bottom_center_ratio"],
                features["width_jump"],
                features["center_jump"],
                features["lane_lower_ratio"],
                features["width_std"],
                features["drivable_ratio"],
                features["lower_drivable_ratio"],
                features["center_drivable_ratio"],
                features["lane_ratio"],
            ],
            dtype=np.float32,
        )
        coefficients = np.array(self.linear_coefficients, dtype=np.float32)
        if coefficients.shape != values.shape:
            raise ValueError(
                "YOLOPv2 linear coefficients must contain 10 values matching "
                "near_drivable,bottom_center,width_jump,center_jump,lane_lower,"
                "width_std,drivable,lower_drivable,center_drivable,lane_ratio"
            )
        margin = float(np.dot(values, coefficients) + self.linear_intercept)
        score = 1.0 / (1.0 + np.exp(-self.linear_temperature * margin))
        return float(np.clip(score, 0.0, 1.0))

    def _analyze_full_model(self, image_path: str) -> float:
        drivable_mask, lane_mask = self._predict_masks(image_path)
        features = self._geometry_features(drivable_mask, lane_mask)
        return self._score_geometry(features)
    
    def detect(self, image_path: str) -> float:
        if self.use_full_model:
            return self._analyze_full_model(image_path)
        return self._analyze_image_heuristics(image_path)
