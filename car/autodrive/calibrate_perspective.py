#!/usr/bin/env python3
"""Create a four-point bird's-eye calibration from an onboard frame."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


AUTODRIVE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = AUTODRIVE_DIR / "onboard_calibration.yaml"


def build_parser():
    parser = argparse.ArgumentParser(description="Calibrate onboard perspective")
    parser.add_argument("image", help="Fixed onboard-camera image")
    parser.add_argument(
        "--points",
        help="Pixel points: x1,y1;x2,y2;x3,y3;x4,y4 in TL,TR,BR,BL order",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--preview", help="Optional bird's-eye preview path")
    parser.add_argument(
        "--destination-margin",
        type=float,
        default=0.22,
        help="Left/right normalized margin in the bird's-eye output",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def parse_points(value):
    if not value:
        return None
    points = []
    for pair in value.split(";"):
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError("--points must contain four x,y pairs")
        points.append((float(parts[0]), float(parts[1])))
    if len(points) != 4:
        raise ValueError("--points must contain exactly four x,y pairs")
    return points


def collect_points(image):
    points = []
    window = "Click TL, TR, BR, BL; R=reset, Enter=save, Esc=cancel"

    def on_mouse(event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            display = image.copy()
            for index, point in enumerate(points):
                xy = (int(point[0]), int(point[1]))
                cv2.circle(display, xy, 7, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    str(index + 1),
                    (xy[0] + 8, xy[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10) and len(points) == 4:
                return points
            if key in (ord("r"), ord("R")):
                points.clear()
            if key == 27:
                raise RuntimeError("calibration cancelled")
    finally:
        cv2.destroyWindow(window)


def main():
    args = build_parser().parse_args()
    if not 0 <= args.destination_margin < 0.5:
        raise ValueError("--destination-margin must be in [0, 0.5)")
    image_path = Path(args.image).expanduser().resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"unable to read image: {image_path}")
    height, width = image.shape[:2]

    points = parse_points(args.points)
    if points is None:
        points = collect_points(image)
    source_pixels = np.asarray(points, dtype=np.float32)
    if (
        np.any(source_pixels[:, 0] < 0)
        or np.any(source_pixels[:, 0] >= width)
        or np.any(source_pixels[:, 1] < 0)
        or np.any(source_pixels[:, 1] >= height)
    ):
        raise ValueError("source points must lie inside the image")

    scale = np.array([max(1, width - 1), max(1, height - 1)], dtype=np.float32)
    source_normalized = source_pixels / scale
    margin = float(args.destination_margin)
    destination_normalized = np.asarray(
        [[margin, 0.0], [1.0 - margin, 0.0], [1.0 - margin, 1.0], [margin, 1.0]],
        dtype=np.float32,
    )
    destination_pixels = destination_normalized * scale

    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite without --force: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibrated": True,
        "source_image": str(image_path),
        "source_points": source_normalized.round(7).tolist(),
        "destination_points": destination_normalized.round(7).tolist(),
    }
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    matrix = cv2.getPerspectiveTransform(source_pixels, destination_pixels)
    preview = cv2.warpPerspective(image, matrix, (width, height))
    preview_path = (
        Path(args.preview).expanduser().resolve()
        if args.preview
        else output_path.with_name(f"{output_path.stem}_preview.jpg")
    )
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"unable to write preview: {preview_path}")
    print(f"Calibration: {output_path}")
    print(f"Preview: {preview_path}")
    print("Inspect the preview before enabling motors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
