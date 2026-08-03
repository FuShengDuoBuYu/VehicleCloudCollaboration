#!/usr/bin/env python3
"""Read-only Raspberry Pi readiness checks for the onboard driving runtime."""

import argparse
import importlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys

import yaml


AUTODRIVE_DIR = Path(__file__).resolve().parents[1]
CAR_DIR = AUTODRIVE_DIR.parent
REPO_ROOT = CAR_DIR.parent
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.perception.perspective import validate_calibration_camera_pose


LOCAL_CONFIG = AUTODRIVE_DIR / "config" / "onboard_runtime.yaml"
DEFAULT_CONFIG = LOCAL_CONFIG
MINIMUM_PYTHON = (3, 9)
MAXIMUM_PYTHON = (3, 12)


def build_parser():
    parser = argparse.ArgumentParser(description="Read-only onboard readiness check")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Open the configured camera and read one frame",
    )
    parser.add_argument("--json-output", help="Optional report path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero for warnings as well as failures",
    )
    return parser


def repo_path(value):
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def add(checks, name, status, detail):
    checks.append({"name": name, "status": status, "detail": str(detail)})


def check_imports(checks):
    modules = ["numpy", "cv2", "yaml", "smbus2"]
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "available")
            add(checks, f"python_import:{name}", "pass", version)
        except Exception as exc:
            add(checks, f"python_import:{name}", "fail", exc)


def check_camera(checks, camera_config):
    try:
        import cv2

        index = int(camera_config.get("index", 0))
        capture = cv2.VideoCapture(index)
        try:
            ok, frame = capture.read() if capture.isOpened() else (False, None)
        finally:
            capture.release()
        if ok and frame is not None:
            add(
                checks,
                "camera",
                "pass",
                f"index={index}, resolution={frame.shape[1]}x{frame.shape[0]}",
            )
        else:
            add(checks, "camera", "fail", f"unable to read camera index {index}")
    except Exception as exc:
        add(checks, "camera", "fail", exc)


def main():
    args = build_parser().parse_args()
    checks = []
    config_path = Path(args.config).expanduser().resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if config.get("version") != 1:
            raise ValueError("config version must be 1")
        add(checks, "runtime_config", "pass", config_path)
    except Exception as exc:
        config = {}
        add(checks, "runtime_config", "fail", exc)

    python_ok = MINIMUM_PYTHON <= sys.version_info[:2] < MAXIMUM_PYTHON
    add(
        checks,
        "python_version",
        "pass" if python_ok else "fail",
        f"{platform.python_version()}; supported: >=3.9,<3.12",
    )
    machine = platform.machine().lower()
    pi_arch = machine in {"aarch64", "arm64", "armv7l"}
    add(
        checks,
        "architecture",
        "pass" if pi_arch else "warn",
        f"{machine}; ARM is expected on the Raspberry Pi",
    )
    check_imports(checks)

    add(
        checks,
        "perception_mode",
        "pass",
        "yellow boundaries plus HSV/Lab road surface",
    )

    calibration = repo_path(
        config.get("perspective", {}).get("calibration")
    )
    if calibration is None:
        add(
            checks,
            "perspective_calibration",
            "warn",
            "not configured; dry-run only",
        )
    else:
        try:
            validate_calibration_camera_pose(
                calibration,
                config.get("camera", {}),
            )
            add(
                checks,
                "perspective_calibration",
                "pass",
                f"{calibration}; camera pose matches runtime",
            )
        except Exception as exc:
            add(checks, "perspective_calibration", "fail", exc)

    i2c_device = Path("/dev/i2c-1")
    if i2c_device.exists():
        readable = os.access(i2c_device, os.R_OK | os.W_OK)
        add(
            checks,
            "i2c_device",
            "pass" if readable else "fail",
            f"{i2c_device}, read/write={readable}",
        )
    else:
        add(
            checks,
            "i2c_device",
            "warn",
            "/dev/i2c-1 is absent; expected outside Raspberry Pi",
        )

    free_gib = shutil.disk_usage(REPO_ROOT).free / 1024**3
    add(
        checks,
        "free_disk",
        "pass" if free_gib >= 2.0 else "warn",
        f"{free_gib:.2f} GiB",
    )
    if args.camera:
        check_camera(checks, config.get("camera", {}))

    report = {
        "ok": not any(item["status"] == "fail" for item in checks),
        "strict_ok": all(item["status"] == "pass" for item in checks),
        "platform": {
            "machine": platform.machine(),
            "system": platform.platform(),
            "python": platform.python_version(),
        },
        "checks": checks,
    }
    for item in checks:
        print(
            f"{item['status'].upper():<5} {item['name']:<28} {item['detail']}"
        )
    print(
        f"\nResult: {'READY' if report['ok'] else 'NOT READY'}"
        + (" (warnings remain)" if report["ok"] and not report["strict_ok"] else "")
    )
    if args.json_output:
        output = Path(args.json_output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not report["ok"] or (args.strict and not report["strict_ok"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
