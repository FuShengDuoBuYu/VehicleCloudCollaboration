#!/usr/bin/env python3

import argparse

from vehicle_control.web import run_server


def build_parser():
    parser = argparse.ArgumentParser(description="Start the local vehicle control web service.")
    parser.add_argument("--host", default=None, help="Host address to bind, default uses settings.py")
    parser.add_argument("--port", type=int, default=None, help="Port to bind, default uses settings.py")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Preferred OpenCV camera index. Falls back to 0-3 if unavailable.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        run_server(host=args.host, port=args.port, camera_index=args.camera_index)
    except RuntimeError as exc:
        print(exc)
