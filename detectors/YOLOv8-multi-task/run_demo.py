import sys
import os
import cv2
import torch
import numpy as np

project_root = os.getcwd()
sys.path.insert(0, os.path.join(project_root, "ultralytics"))
from ultralytics import YOLO

# ==================== 模型相关参数配置 ====================
MODEL_PATH = os.path.join(project_root, "weights", "v4.pt")
SOURCE_PATH = 'test_input_single'

# ==================== 长尾场景阈值配置 ====================
LONG_TAIL_CONFIG = {
    #1.目标检测相关指标
    # 置信度“犹豫”区间：模型在此区间内认为“有物体但看不清”，常用于捕捉光照不足、遮挡或远距离的模糊目标。
    "conf_range": (0.2, 0.45),
    # 犹豫目标数量阈值：单张图中处于上述区间的目标数超过此值，则认为场景复杂度过高或模型感知能力下降，判定为异常风险。
    "low_conf_count_limit": 5,
    # 目标密度阈值：单图中检测到的总目标数（如车、人等）上限。超过此值通常代表极度拥堵或模型产生大量冗余错误框（幻觉）。
    "density_limit": 10,
    # 尺寸占比阈值：单个检测框占全图面积的最大比例。用于识别传感器遮挡、镜头污染或物体距离摄像头过近（近距离冲突场景）。
    "size_ratio_limit": 0.6,
    # 罕见类别清单：定义在常规道路场景中极少出现的类别 ID（如火车、异形车）。一旦出现，即刻触发长尾场景判别。
    "rare_classes": [9],

    #2.道路分割相关指标 (Segmentation Metrics) ---
    # 可行驶区域最小比例：路面掩码像素占全图的最小百分比。低于此值可能意味着车辆正在急转弯、进入断头路或前方被大货车完全遮挡。
    "drivable_area_min": 0.15,
    # 可行驶区域破碎度上限：正常路面应为1块，超过3块通常意味着路面被障碍物切断或感知严重失真。
    "drivable_fragment_limit": 3,
    # 车道线破碎度上限：允许的车道线连通域（碎块）数量。超过此值通常意味着路面有积水反光、阴影干扰或车道线磨损严重，导致感知失效。
    "lane_fragment_limit": 5,
    # 车道线最小宽度比：车道线像素占全图的最小比例。在路面正常但车道线像素极少时，用于识别磨损严重或无标线的长尾路段。
    "lane_width_min_ratio": 0.02,

    #3.融合权重配置 (Weighted Fusion) ---
    # 目标检测权重：综合评分中来自检测任务贡献的比例。
    "weight_det": 0.3,
    # 分割结果权重：综合评分中来自路面与车道线分割贡献的比例。
    "weight_seg": 0.7
}


# ======= 1. 目标检测异常评分函数 ==========
def get_det_anomaly_score(det_result, img_shape, config=LONG_TAIL_CONFIG):
    boxes = det_result.boxes
    if boxes is None or len(boxes) == 0:
        return 0.0, "Empty"

    confidences = boxes.conf
    h, w = img_shape
    score = 0.0
    reasons = []

    # a. 置信度模糊评分 (Max 0.4)
    c_min, c_max = config["conf_range"]
    low_conf_count = torch.sum((confidences > c_min) & (confidences < c_max)).item()
    conf_score = min(low_conf_count / (config["low_conf_count_limit"] * 2), 0.4)
    if conf_score > 0:
        score += conf_score
        reasons.append(f"Uncertainty({low_conf_count})")

    # b. 密度评分 (Max 0.3)
    density_score = min(len(boxes) / (config["density_limit"] * 2), 0.3)
    if len(boxes) > config["density_limit"]:
        score += density_score
        reasons.append(f"Density({len(boxes)})")

    # c. 尺寸与类别评分 (Max 0.3)
    img_area = h * w
    max_box_ratio = max([(b[2] * b[3]) / img_area for b in boxes.xywh]) if len(boxes) > 0 else 0
    if max_box_ratio > config["size_ratio_limit"] or any(cls in config["rare_classes"] for cls in boxes.cls):
        score += 0.3
        reasons.append("Size/Class-Anomaly")

    return min(score, 1.0), "|".join(reasons)


# ======= 2. 分割结果异常评分函数 ==========
def get_seg_anomaly_score(d_mask, l_mask, config=LONG_TAIL_CONFIG):
    h, w = d_mask.shape
    total_pixels = h * w
    score = 0.0
    reasons = []

    # --- a. 路面面积与破碎度检测 (Drivable Area Analysis) ---
    d_binary = (d_mask > 0).astype(np.uint8)
    d_contours, _ = cv2.findContours(d_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤微小噪声，只保留有效路面块
    valid_d_fragments = [c for c in d_contours if cv2.contourArea(c) > 500]

    # 面积萎缩评分
    drivable_area_ratio = np.count_nonzero(d_mask) / total_pixels
    if drivable_area_ratio < config["drivable_area_min"]:
        score += 0.3
        reasons.append("Road-Shrink")

    # 路面破碎度评分 (核心新增)
    if len(valid_d_fragments) > config["drivable_fragment_limit"]:
        # 碎块越多，分数越高，最高贡献 0.3
        frag_score = min(len(valid_d_fragments) / 10, 0.3)
        score += frag_score
        reasons.append(f"Road-Fragmented({len(valid_d_fragments)})")

    # --- b. 车道线碎裂度检测 (Lane Analysis) ---
    lane_binary = (l_mask > 0).astype(np.uint8)
    l_contours, _ = cv2.findContours(lane_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_l_fragments = [c for c in l_contours if cv2.contourArea(c) > 50]

    if len(valid_l_fragments) > config["lane_fragment_limit"]:
        score += 0.2
        reasons.append(f"Lane-Fragmented")

    # --- c. 车道丢失检测 ---
    lane_area_ratio = np.count_nonzero(l_mask) / total_pixels
    if lane_area_ratio < config["lane_width_min_ratio"] and drivable_area_ratio > 0.3:
        score += 0.2
        reasons.append("Lane-Lost")

    return min(score, 1.0), "|".join(reasons)


# ================= 模型加载 ======================
model = YOLO(MODEL_PATH, task='multi')
model.model.names[0] = 'car'
results = model.predict(source=SOURCE_PATH, imgsz=(384, 672), conf=0.2, device=0)

# ================= 结果处理循环 ===================
for i in range(0, len(results), 3):
    try:
        det_result = results[i][0]
        d_mask_entry = results[i + 1]
        l_mask_entry = results[i + 2]

        orig_img = det_result.orig_img
        h, w = orig_img.shape[:2]
        canvas = orig_img.copy()

        # --- 数据转换 ---
        d_tensor = d_mask_entry[0] if isinstance(d_mask_entry, list) else d_mask_entry
        l_tensor = l_mask_entry[0] if isinstance(l_mask_entry, list) else l_mask_entry
        d_mask = d_tensor.detach().cpu().numpy().squeeze()
        l_mask = l_tensor.detach().cpu().numpy().squeeze()
        d_mask_resized = cv2.resize(d_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        l_mask_resized = cv2.resize(l_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- 分数计算 ---
        det_score, det_reasons = get_det_anomaly_score(det_result, (h, w))
        seg_score, seg_reasons = get_seg_anomaly_score(d_mask_resized, l_mask_resized)

        # 加权总分
        final_score = (det_score * LONG_TAIL_CONFIG["weight_det"]) + \
                      (seg_score * LONG_TAIL_CONFIG["weight_seg"])

        # --- 渲染展示 ---
        canvas[d_mask_resized > 0] = [0, 255, 0]
        canvas[l_mask_resized > 0] = [255, 0, 0]
        final_show = cv2.addWeighted(orig_img, 0.7, canvas, 0.3, 0)
        final_show = det_result.plot(img=final_show, line_width=2)

        # 打印分数和信息
        overlay = final_show.copy()
        cv2.rectangle(overlay, (20, 15), (600, 135), (0, 0, 0), -1)  # 黑色实心矩形
        # 将遮罩层与原图融合，实现半透明效果 (0.5 为透明度)
        final_show = cv2.addWeighted(overlay, 0.5, final_show, 0.5, 0)
        text_color = (0, int(255 * (1 - final_score)), int(255 * final_score))
        cv2.putText(final_show, f"TOTAL ANOMALY: {final_score:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(final_show, f"Det Score: {det_score:.2f} ({det_reasons})", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(final_show, f"Seg Score: {seg_score:.2f} ({seg_reasons})", (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        print(f"正在处理: {os.path.basename(det_result.path)} -> Score: {final_score:.2f}")

        cv2.imshow("Multi-Task Result", final_show)
        if cv2.waitKey(0) & 0xFF == ord('q'): break

    except Exception as e:
        print(f"Error at index {i}: {e}")

cv2.destroyAllWindows()