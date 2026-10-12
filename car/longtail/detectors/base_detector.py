"""Historical import compatibility; shared inference now lives in autodrive."""
from pathlib import Path
import sys

_car = str(Path(__file__).resolve().parents[2])
if _car not in sys.path:
    sys.path.insert(0, _car)

from autodrive.perception.model_base import BaseDetector
