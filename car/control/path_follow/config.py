"""Base config for the sibling path_follow car app."""

from donkeycar.templates.cfg_path_follow import *
import os

CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')
