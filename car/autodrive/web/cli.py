#!/usr/bin/env python3
"""Start the dedicated onboard LCC web console."""

import argparse
from pathlib import Path
import sys

import yaml


AUTODRIVE_DIR = Path(__file__).resolve().parents[1]
CAR_DIR = AUTODRIVE_DIR.parent
REPO_ROOT = CAR_DIR.parent
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.web.server import (
    LCCProcessManager,
    LCCWebServer,
    MOTOR_CONFIRMATION,
)


def repo_path(value):
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_parser():
    parser = argparse.ArgumentParser(description="Start the onboard LCC web console")
    parser.add_argument(
        "--config",
        default=str(AUTODRIVE_DIR / "config" / "onboard_runtime.yaml"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--default-max-runtime-seconds",
        type=float,
        default=60.0,
        help="Default duration shown on the page; must be 1-300 seconds",
    )
    parser.add_argument(
        "--enable-motors",
        action="store_true",
        help="Arm the web Start button for physical motor output",
    )
    parser.add_argument("--confirm-motor-motion", default="")
    return parser


def main():
    args = build_parser().parse_args()
    if args.enable_motors and args.confirm_motor_motion != MOTOR_CONFIRMATION:
        raise ValueError(
            "--enable-motors requires --confirm-motor-motion "
            f"{MOTOR_CONFIRMATION}"
        )
    config_path = repo_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("version") != 1:
        raise ValueError("onboard runtime config version must be 1")
    output_dir = repo_path(
        config.get("runtime", {}).get("output_dir", "outputs/onboard_runtime")
    )

    manager = LCCProcessManager(
        repo_root=REPO_ROOT,
        python_executable=sys.executable,
        runner_path=AUTODRIVE_DIR / "run_onboard.py",
        config_path=config_path,
        output_dir=output_dir,
        motors_enabled=args.enable_motors,
        default_max_runtime_seconds=args.default_max_runtime_seconds,
    )
    server = LCCWebServer(manager, host=args.host, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        print("LCC web console interrupted; stopping vehicle.", flush=True)
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
