#!/usr/bin/env python3
"""Replay images/videos through YOLOPv2 and a hardware-free LCC controller."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


AUTODRIVE_DIR = Path(__file__).resolve().parent
CAR_DIR = AUTODRIVE_DIR.parent
LONGTAIL_DIR = CAR_DIR / "longtail"
REPO_ROOT = CAR_DIR.parent
for path in (CAR_DIR, LONGTAIL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from autodrive.lane_centering import LaneCenteringController, RoadCenterlineEstimator
from autodrive.perspective import PerspectiveMapper
from autodrive.visualization import render_debug_frame
from detectors.yolopv2_detector import YOLOPv2Detector


DEFAULT_WEIGHTS = REPO_ROOT / "car" / "longtail" / "models" / "yolopv2.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline YOLOPv2 road-center and lane-centering replay"
    )
    parser.add_argument("inputs", nargs="+", help="Input images or videos")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "autodrive_offline"),
        help="Directory for annotated media, CSV logs, and summary JSON",
    )
    parser.add_argument(
        "--weights",
        default=os.environ.get("CAR_LONGTAIL_YOLOPV2_WEIGHTS", str(DEFAULT_WEIGHTS)),
        help="YOLOPv2 TorchScript weights",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--img-size", type=int, default=640, help="YOLOPv2 letterbox size"
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=6,
        help="Run every Nth video frame; output FPS is reduced by the same factor",
    )
    parser.add_argument(
        "--route-hint",
        choices=["left", "center", "right"],
        default="center",
        help="Preferred branch when the drivable mask splits",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop each video after this many inferred frames; 0 means all",
    )
    parser.add_argument(
        "--camera-view",
        choices=["external", "onboard"],
        default="external",
        help="External view disables LCC; onboard view enables offline LCC geometry",
    )
    parser.add_argument(
        "--calibration",
        help="Perspective-calibration YAML for onboard bird's-eye transformation",
    )
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help="Safety alias that disables LCC regardless of camera view",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Also save every inferred video frame as a JPEG",
    )
    return parser


def make_detector(weights: Path, device: str, img_size: int) -> YOLOPv2Detector:
    if not weights.exists():
        raise FileNotFoundError(f"YOLOPv2 weights not found: {weights}")
    return YOLOPv2Detector(
        {
            "weights_path": str(weights),
            "img_size": img_size,
            "device": device,
            "use_full_model": True,
            "fast_mask": True,
        }
    )


def analyze_frame(
    detector,
    estimator,
    controller,
    frame,
    route_hint,
    dt,
    perception_only=False,
    perspective_mapper=None,
):
    started = time.monotonic()
    drivable_mask, lane_mask = detector.predict_masks(frame)
    inference_ms = (time.monotonic() - started) * 1000.0
    control_drivable_mask = drivable_mask
    control_lane_mask = lane_mask
    if perspective_mapper is not None:
        control_drivable_mask = perspective_mapper.warp_mask(drivable_mask)
        control_lane_mask = perspective_mapper.warp_mask(lane_mask)
    estimate = estimator.estimate(
        control_drivable_mask, control_lane_mask, route_hint
    )
    command = controller.update(estimate, dt=dt)
    if perception_only:
        command = type(command)(
            "perception-only",
            0.0,
            0.0,
            0.0,
            estimate.confidence,
            "external-camera footage; LCC command disabled",
        )
    display_estimate = estimate
    if perspective_mapper is not None:
        display_estimate = perspective_mapper.camera_estimate(
            estimate, drivable_mask.shape
        )
    annotated = render_debug_frame(
        frame,
        drivable_mask,
        lane_mask,
        display_estimate,
        command,
        inference_ms,
        show_control=not perception_only,
    )
    perception = {
        "drivable_ratio": float(np.mean(drivable_mask > 0)),
        "lane_ratio": float(np.mean(lane_mask > 0)),
    }
    return annotated, estimate, command, inference_ms, perception


def record_from_result(
    frame_index,
    timestamp_s,
    estimate,
    command,
    inference_ms,
    perception,
):
    control_enabled = command.action != "perception-only"
    left_pwm, right_pwm = command.as_pwm() if control_enabled else (None, None)
    lookahead_x = None
    lookahead_y = None
    if control_enabled and estimate.lookahead_point is not None:
        lookahead_x, lookahead_y = estimate.lookahead_point
    return {
        "frame_index": int(frame_index),
        "timestamp_s": round(float(timestamp_s), 4),
        "control_enabled": control_enabled,
        "valid": bool(estimate.valid) if control_enabled else None,
        "confidence": (
            round(float(estimate.confidence), 5) if control_enabled else None
        ),
        "lateral_error": (
            round(float(estimate.lateral_error), 5) if control_enabled else None
        ),
        "heading_error": (
            round(float(estimate.heading_error), 5) if control_enabled else None
        ),
        "lookahead_x_mask": lookahead_x,
        "lookahead_y_mask": lookahead_y,
        "action": command.action,
        "steering": round(float(command.steering), 5) if control_enabled else None,
        "left_speed": (
            round(float(command.left_speed), 5) if control_enabled else None
        ),
        "right_speed": (
            round(float(command.right_speed), 5) if control_enabled else None
        ),
        "left_pwm": left_pwm,
        "right_pwm": right_pwm,
        "inference_ms": round(float(inference_ms), 3),
        "drivable_ratio": round(float(perception["drivable_ratio"]), 5),
        "lane_ratio": round(float(perception["lane_ratio"]), 5),
        "reason": command.reason,
    }


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_contact_sheet(sample_dir: Path, output_path: Path) -> None:
    sample_paths = sorted(sample_dir.glob("frame_*.jpg"))
    if not sample_paths:
        return
    thumbnails = []
    thumb_width = 480
    for sample_path in sample_paths:
        image = cv2.imread(str(sample_path))
        if image is None:
            continue
        thumb_height = max(1, int(image.shape[0] * thumb_width / image.shape[1]))
        thumbnails.append(
            cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        )
    if not thumbnails:
        return
    columns = 2
    rows = (len(thumbnails) + columns - 1) // columns
    tile_height = max(image.shape[0] for image in thumbnails)
    sheet = np.zeros((rows * tile_height, columns * thumb_width, 3), dtype=np.uint8)
    for index, image in enumerate(thumbnails):
        row, column = divmod(index, columns)
        y = row * tile_height
        x = column * thumb_width
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"failed to write contact sheet: {output_path}")


def process_image(
    path,
    output_dir,
    detector,
    estimator,
    route_hint,
    perception_only,
    perspective_mapper,
):
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"failed to read image: {path}")
    controller = LaneCenteringController()
    annotated, estimate, command, inference_ms, perception = analyze_frame(
        detector,
        estimator,
        controller,
        frame,
        route_hint,
        None,
        perception_only,
        perspective_mapper,
    )
    mode = "perception" if perception_only else "lcc"
    output_path = output_dir / f"{path.stem}_{mode}.jpg"
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"failed to write image: {output_path}")
    record = record_from_result(
        0, 0.0, estimate, command, inference_ms, perception
    )
    write_csv(output_dir / f"{path.stem}_{mode}.csv", [record])
    return {
        "input": str(path),
        "output": str(output_path),
        "samples": 1,
        "valid_samples": None if perception_only else int(estimate.valid),
        "mean_confidence": None if perception_only else estimate.confidence,
        "mean_inference_ms": inference_ms,
        "mean_drivable_ratio": perception["drivable_ratio"],
        "mean_lane_ratio": perception["lane_ratio"],
        "actions": {command.action: 1},
        "mode": mode,
    }


def process_video(
    path,
    output_dir,
    detector,
    estimator,
    route_hint,
    sample_every,
    max_samples,
    perception_only,
    save_samples,
    perspective_mapper,
):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"failed to open video: {path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = max(1.0, source_fps / sample_every)
    mode = "perception" if perception_only else "lcc"
    output_path = output_dir / f"{path.stem}_{mode}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to create output video: {output_path}")

    controller = LaneCenteringController()
    records = []
    sample_dir = output_dir / f"{path.stem}_{mode}_frames"
    if save_samples:
        sample_dir.mkdir(parents=True, exist_ok=True)
    frame_index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % sample_every:
                continue
            annotated, estimate, command, inference_ms, perception = analyze_frame(
                detector,
                estimator,
                controller,
                frame,
                route_hint,
                sample_every / source_fps,
                perception_only,
                perspective_mapper,
            )
            writer.write(annotated)
            if save_samples:
                sample_path = sample_dir / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(sample_path), annotated):
                    raise RuntimeError(f"failed to write sample: {sample_path}")
            records.append(
                record_from_result(
                    frame_index,
                    frame_index / source_fps,
                    estimate,
                    command,
                    inference_ms,
                    perception,
                )
            )
            if max_samples and len(records) >= max_samples:
                break
    finally:
        capture.release()
        writer.release()

    if not records:
        raise RuntimeError(f"video contained no readable frames: {path}")
    write_csv(output_dir / f"{path.stem}_{mode}.csv", records)
    contact_sheet = None
    if save_samples:
        contact_sheet = output_dir / f"{path.stem}_{mode}_contact_sheet.jpg"
        write_contact_sheet(sample_dir, contact_sheet)

    action_counts = {}
    for record in records:
        action = record["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "input": str(path),
        "output": str(output_path),
        "samples": len(records),
        "valid_samples": (
            None
            if perception_only
            else sum(int(record["valid"]) for record in records)
        ),
        "mean_confidence": (
            None
            if perception_only
            else float(np.mean([r["confidence"] for r in records]))
        ),
        "mean_inference_ms": float(np.mean([r["inference_ms"] for r in records])),
        "mean_drivable_ratio": float(
            np.mean([r["drivable_ratio"] for r in records])
        ),
        "mean_lane_ratio": float(np.mean([r["lane_ratio"] for r in records])),
        "actions": action_counts,
        "mode": mode,
        "control_evaluation_disabled": bool(perception_only),
        "source_fps": source_fps,
        "output_fps": output_fps,
        "sample_every": sample_every,
        "sample_frames": str(sample_dir) if save_samples else None,
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_every < 1:
        raise ValueError("--sample-every must be at least 1")
    input_paths = [Path(value).expanduser().resolve() for value in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = make_detector(Path(args.weights).expanduser(), args.device, args.img_size)
    estimator = RoadCenterlineEstimator()
    perception_only = args.perception_only or args.camera_view == "external"
    perspective_mapper = None
    calibration_path = None
    if args.calibration:
        if args.camera_view != "onboard":
            raise ValueError("--calibration is only meaningful with --camera-view onboard")
        calibration_path = Path(args.calibration).expanduser().resolve()
        perspective_mapper = PerspectiveMapper.from_yaml(calibration_path)
    elif args.camera_view == "onboard":
        print(
            "WARNING: onboard LCC is running in uncalibrated image coordinates; "
            "do not transfer its gains to the real vehicle.",
            flush=True,
        )

    summaries = []
    for path in input_paths:
        print(f"Processing {path.name} ...", flush=True)
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            summary = process_image(
                path,
                output_dir,
                detector,
                estimator,
                args.route_hint,
                perception_only,
                perspective_mapper,
            )
        elif suffix in VIDEO_SUFFIXES:
            summary = process_video(
                path,
                output_dir,
                detector,
                estimator,
                args.route_hint,
                args.sample_every,
                args.max_samples,
                perception_only,
                args.save_samples,
                perspective_mapper,
            )
        else:
            raise ValueError(f"unsupported input type: {path}")
        summary["camera_view"] = args.camera_view
        summary["perspective_calibration"] = (
            str(calibration_path) if calibration_path else None
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
