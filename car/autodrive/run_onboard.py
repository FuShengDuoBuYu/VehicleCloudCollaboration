#!/usr/bin/env python3
"""Stable command-line entry point for the onboard LCC runtime."""

from pathlib import Path
import sys


CAR_DIR = Path(__file__).resolve().parents[1]
if str(CAR_DIR) not in sys.path:
    sys.path.insert(0, str(CAR_DIR))

from autodrive.runtime.onboard import main


if __name__ == "__main__":
    raise SystemExit(main())
