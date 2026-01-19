import os
import sys
import cv2
import torch
import numpy as np
from typing import Dict, Any
from .base_detector import BaseDetector

class YOLOv8MultiTaskDetector(BaseDetector):
    """基于 YOLOv8 多任务模型的长尾场景检测器，集成了：目标检测异常评分 + 道路分割异常评分"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.lt_config = self.config.get('long_tail_config')
        try:
            # 确保能找到项目自带的 ultralytics 库
            project_root = os.getcwd()

            rel_ultralytics_path = self.config.get('ultralytics_path')
            custom_ultralytics_path = os.path.join(project_root, rel_ultralytics_path)
            if custom_ultralytics_path not in sys.path:
                sys.path.insert(0, custom_ultralytics_path)
            from ultralytics import YOLO

            model_path = self.config.get('model_path')
            self.model = YOLO(model_path, task='multi')
            self.model.model.names[0] = 'car'  # 修正类别名
            self.imgsz = self.config.get('imgsz', (384, 672))

        except ImportError:
            print("Warning: ultralytics (multi-task version) not available.")
            self.model = None

    def _get_det_anomaly_score(self, det_result, img_shape):
        """内部函数：计算目标检测维度异常分"""
        boxes = det_result.boxes
        if boxes is None or len(boxes) == 0:
            return 0.0

        confidences = boxes.conf
        h, w = img_shape
        score = 0.0

        # a. 置信度模糊评分
        c_min, c_max = self.lt_config["conf_range"]
        low_conf_count = torch.sum((confidences > c_min) & (confidences < c_max)).item()
        score += min(low_conf_count / (self.lt_config["low_conf_count_limit"] * 2), 0.4)

        # b. 密度评分
        if len(boxes) > self.lt_config["density_limit"]:
            score += min(len(boxes) / (self.lt_config["density_limit"] * 2), 0.3)

        # c. 尺寸与类别评分
        img_area = h * w
        max_box_ratio = max([(b[2] * b[3]) / img_area for b in boxes.xywh]) if len(boxes) > 0 else 0
        if max_box_ratio > self.lt_config["size_ratio_limit"] or any(
                cls in self.lt_config["rare_classes"] for cls in boxes.cls):
            score += 0.3

        return min(score, 1.0)

    def _get_seg_anomaly_score(self, d_mask, l_mask):
        """内部函数：计算分割维度异常分"""
        h, w = d_mask.shape
        total_pixels = h * w
        score = 0.0

        # 路面面积与破碎度
        d_binary = (d_mask > 0).astype(np.uint8)
        d_contours, _ = cv2.findContours(d_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_d_fragments = [c for c in d_contours if cv2.contourArea(c) > 500]

        drivable_area_ratio = np.count_nonzero(d_mask) / total_pixels
        if drivable_area_ratio < self.lt_config["drivable_area_min"]:
            score += 0.3

        if len(valid_d_fragments) > self.lt_config["drivable_fragment_limit"]:
            score += min(len(valid_d_fragments) / 10, 0.3)

        # 车道线碎裂度与缺失
        lane_binary = (l_mask > 0).astype(np.uint8)
        l_contours, _ = cv2.findContours(lane_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_l_fragments = [c for c in l_contours if cv2.contourArea(c) > 50]

        if len(valid_l_fragments) > self.lt_config["lane_fragment_limit"]:
            score += 0.2

        lane_area_ratio = np.count_nonzero(l_mask) / total_pixels
        if lane_area_ratio < self.lt_config["lane_width_min_ratio"] and drivable_area_ratio > 0.3:
            score += 0.2

        return min(score, 1.0)

    def detect(self, image_path: str) -> float:
        if self.model is None:
            return 0.5

        # 1. 运行推理 (Multi-Task 模式)
        results = self.model.predict(source=image_path, imgsz=self.imgsz, conf=0.1, verbose=False)

        # 2. 解析多任务输出
        # 注意：多任务推理一张图会产生3个结果项：[Det, Drivable_Tensor, Lane_Tensor]
        if len(results) < 3:
            return 0.0

        det_result = results[0][0] if isinstance(results[0], list) else results[0]
        d_mask_entry = results[1]
        l_mask_entry = results[2]

        # 3. 数据预处理
        orig_img = det_result.orig_img
        h, w = orig_img.shape[:2]

        d_tensor = d_mask_entry[0] if isinstance(d_mask_entry, list) else d_mask_entry
        l_tensor = l_mask_entry[0] if isinstance(l_mask_entry, list) else l_mask_entry

        d_mask = d_tensor.detach().cpu().numpy().squeeze()
        l_mask = l_tensor.detach().cpu().numpy().squeeze()

        # 缩放掩码到原图大小用于评估
        d_mask_res = cv2.resize(d_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        l_mask_res = cv2.resize(l_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 4. 计算子分数
        det_score = self._get_det_anomaly_score(det_result, (h, w))
        seg_score = self._get_seg_anomaly_score(d_mask_res, l_mask_res)

        # 5. 加权融合总分
        final_score = (det_score * self.lt_config["weight_det"]) + \
                      (seg_score * self.lt_config["weight_seg"])

        return float(np.clip(final_score, 0.0, 1.0))