"""Decode the official V0.0.1 trace heads, without a torchvision dependency.

Protocol reference: CAIC-AD/YOLOPv2 utils/utils.py, split_for_trace_model.
Boxes returned here are normalized to the unpadded camera image.
"""
import numpy as np


def decode_objects(heads, anchors, network_shape, padding, confidence=.3, iou=.45):
    if len(heads) != 3 or len(anchors) != 3:
        raise ValueError("YOLOPv2 requires three detection scales")
    candidates = []
    top, bottom, left, right = padding
    height = network_shape[0] - top - bottom
    width = network_shape[1] - left - right
    if min(height, width) <= 0:
        raise ValueError("invalid unpadded image size")
    for raw, anchor, stride in zip(heads, anchors, (8, 16, 32)):
        raw = np.asarray(raw)
        if raw.ndim != 4 or raw.shape[:2] != (1, 255) or not np.isfinite(raw).all():
            raise ValueError("invalid YOLOPv2 detection head")
        ny, nx = raw.shape[2:]
        values = raw.reshape(3, 85, ny, nx).transpose(0, 2, 3, 1)
        locations = np.where(values[..., 4] > np.log(confidence / (1 - confidence)))
        selected = values[locations].astype(np.float32)
        if not len(selected):
            continue
        probabilities = 1 / (1 + np.exp(-np.clip(selected, -80, 80)))
        classes = probabilities[:, 5:].argmax(axis=1)
        scores = probabilities[:, 4] * probabilities[np.arange(len(selected)), classes + 5]
        anchor = np.asarray(anchor, dtype=np.float32).reshape(3, 2)
        if not np.isfinite(anchor).all() or np.any(anchor <= 0):
            raise ValueError("invalid YOLOPv2 anchors")
        xy = (probabilities[:, :2] * 2 - .5 + np.stack((locations[2], locations[1]), axis=1)) * stride
        wh = (probabilities[:, 2:4] * 2) ** 2 * anchor[locations[0]]
        boxes = np.concatenate((xy - wh/2, xy + wh/2), axis=1)
        boxes = np.clip((boxes - [left, top, left, top]) / [width, height, width, height], 0, 1)
        for box, score, cls in zip(boxes, scores, classes):
            if score >= confidence and box[2] > box[0] and box[3] > box[1]:
                candidates.append({'box': box.tolist(), 'confidence': float(score), 'class_id': int(cls)})
    candidates = sorted(candidates, key=lambda d: d['confidence'], reverse=True)[:1000]
    kept = []
    for candidate in candidates:
        a = np.asarray(candidate['box']); area = (a[2]-a[0])*(a[3]-a[1])
        duplicate = False
        for previous in kept:
            if candidate['class_id'] != previous['class_id']:
                continue
            b = np.asarray(previous['box'])
            intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
            union = area + (b[2]-b[0])*(b[3]-b[1]) - intersection
            if intersection / max(union, 1e-12) > iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
            if len(kept) == 100:
                break
    return kept
