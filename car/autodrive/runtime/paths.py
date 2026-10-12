"""Canonical source-tree locations; importing this module does not inspect devices."""
from pathlib import Path
AUTODRIVE_DIR = Path(__file__).resolve().parents[1]
CAR_DIR = AUTODRIVE_DIR.parent
REPO_ROOT = CAR_DIR.parent
DEFAULT_CONFIG = AUTODRIVE_DIR / 'config' / 'onboard_runtime.yaml'
MOTOR_CONFIRMATION = 'I_UNDERSTAND_MOTORS_WILL_MOVE'
