"""Project-specific overrides for path_follow app."""

import os

CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

# Camera backend aligned with your train_car setup (OpenCV).
CAMERA_TYPE = "CVCAM"
CAMERA_INDEX = 0
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3

# Match train_car drivetrain backend so we don't touch PCA9685 I2C.
DRIVE_TRAIN_TYPE = "ROSPBOT"
ROSPBOT_MAX_SPEED = 180
ROSPBOT_TURN_SCALE = 0.7
ROSPBOT_DEADZONE = 0.02

# 1) Small field visualization: make tiny movements more visible on map.
PATH_SCALE = 100.0

# 2) Lower manual driving max speed to ~1/3 while recording in User mode.
USER_MODE_THROTTLE_SCALE = 0.1

# 3) Increase recorded path point density.
PATH_MIN_DIST = 0.005

# Path follow behavior.
USE_CONSTANT_THROTTLE = True
PID_THROTTLE = 0.25
HAVE_ODOM = True
HAVE_ODOM_2 = False
ENCODER_TYPE = "MOCK"
MOCK_TICKS_PER_SECOND = 80

# 4) Auto stop when reaching the end of recorded path.
AUTO_STOP_AT_PATH_END = True
PATH_END_STOP_MARGIN_POINTS = 1

# Web-only control setup: map path-follow actions to web/w1..web/w5.
USE_JOYSTICK_AS_DEFAULT = False
SAVE_PATH_BTN = "web/w1"
LOAD_PATH_BTN = "web/w2"
RESET_ORIGIN_BTN = "web/w3"
ERASE_PATH_BTN = "web/w4"
TOGGLE_RECORDING_BTN = "web/w5"
WEB_BTN_LABEL_W1 = "保存路径"
WEB_BTN_LABEL_W2 = "加载路径"
WEB_BTN_LABEL_W3 = "重置原点"
WEB_BTN_LABEL_W4 = "清空路径"
WEB_BTN_LABEL_W5 = "录制开关"
