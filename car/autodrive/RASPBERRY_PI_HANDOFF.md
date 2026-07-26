# 树莓派无缝衔接手册

## 当前交接点

这里已经完成并通过无硬件验证：

- YOLOPv2 支持直接处理 OpenCV 帧，不再需要每帧写临时图片。
- 外部视角只输出分割；车载视角才允许生成 LCC 指令。
- 可行驶区域连通域、中心线、预瞄点、横向误差和航向误差。
- 差速 LCC、低置信度停车、PWM 安全映射和推理看门狗。
- 可选的 YOLOPv2 关键帧 + 光流 Mask 传播；默认关闭，等待车载数据验证。
- 四点透视标定、视频离线回放、实时 CSV/状态文件和最新叠加帧。
- 电机默认关闭；真实电机需要有效标定与两次显式确认。

到树莓派后不要重新实现这些模块。第一个未完成任务是：**固定车载相机并采集车载视角
数据，完成真实四点标定。**

## A. 系统与文件准备

目标环境：64 位 Raspberry Pi OS、Python 3.10 或 3.11。完整仓库还包含要求
Python `<3.12` 的 DonkeyCar，因此建议统一使用 Python 3.11。

```bash
getconf LONG_BIT
python3 --version
uname -m
```

期望输出包含 `64`、Python `3.11.x` 和 `aarch64`。

开启底盘使用的 ARM I2C，然后重启：

```bash
sudo raspi-config nonint do_i2c 0
sudo reboot
```

模型文件被 Git 忽略，必须单独确认下面的 150 MiB 文件已经复制到树莓派：

```text
car/longtail/models/yolopv2.pt
```

## B. 创建最小运行环境

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-yaml python3-smbus i2c-tools libopenblas-dev

python3 -m venv --system-site-packages .venv-pi
source .venv-pi/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-pi-autodrive.txt
```

PyTorch 官方只为 64 位 Arm 提供 Raspberry Pi 可用的 pip 包；如果 `torch` 没有匹配
版本，先检查系统是不是 `aarch64`，不要改用来历不明的 wheel。

复制运行配置：

```bash
cp car/autodrive/onboard_runtime.example.yaml car/autodrive/onboard_runtime.yaml
```

先运行完全只读的检查：

```bash
python car/autodrive/pi_self_check.py \
  --config car/autodrive/onboard_runtime.yaml \
  --load-model \
  --json-output outputs/onboard_runtime/pi_self_check.json
```

此时透视标定警告是正常的；模型、Python 和依赖不能出现 `FAIL`。

## C. 验证相机，不接电机

CSI 相机先执行：

```bash
rpicam-hello --list-cameras
```

USB 相机可以检查：

```bash
ls -l /dev/video*
```

然后运行仓库自检：

```bash
python car/autodrive/pi_self_check.py \
  --config car/autodrive/onboard_runtime.yaml \
  --camera
```

相机固定后不要再改变高度、俯仰角和横滚角。采集标定图片与视频：

```bash
python car/autodrive/capture_onboard.py \
  --camera-index 0 \
  --seconds 10
```

应生成：

```text
outputs/onboard_capture/onboard_calibration.mp4
outputs/onboard_capture/onboard_calibration_frame.jpg
```

## D. 四点透视标定

在有桌面显示的树莓派上运行：

```bash
python car/autodrive/calibrate_perspective.py \
  outputs/onboard_capture/onboard_calibration_frame.jpg \
  --output car/autodrive/onboard_calibration.yaml
```

依次点击地面道路梯形的左上、右上、右下、左下，按 Enter 保存。检查生成的
`onboard_calibration_preview.jpg`：道路应近似竖直、左右边界不应交叉。

编辑 `car/autodrive/onboard_runtime.yaml`：

```yaml
perspective:
  calibration: car/autodrive/onboard_calibration.yaml
```

再次运行自检，标定项必须为 `PASS`。

## E. 车载相机干运行

电机仍然不要上电或让轮胎接触地面：

```bash
python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --max-samples 100
```

检查以下文件：

```text
outputs/onboard_runtime/latest.jpg
outputs/onboard_runtime/status.json
outputs/onboard_runtime/onboard_log.csv
```

通过条件：

- 中心线沿当前道路延伸，预瞄点没有跳到其他分支。
- 直道转向接近 0，左右弯的正负方向相反。
- 道路丢失时动作立即变为 `stop`。
- `inference_ms` 大部分小于配置中的 `maximum_inference_time`。
- 看门狗没有在正常推理时频繁触发。

如果推理超时，应先降低 `model.img_size`、调整 Torch 线程或降低车速；不能简单增大
超时后直接开车。

基础结果正确后，可以只在干运行中测试时序传播：

```bash
python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --max-samples 200 \
  --temporal
```

比较 `onboard_log.csv` 中 `perception_source=yolopv2` 与 `optical-flow` 的中心线。出现
漂移、跨车道或错误分支时保持关闭。

## F. 架空检查轮子方向

车轮必须全部离地：

```bash
python car/autodrive/check_wheel_directions.py \
  --confirm-wheels-lifted WHEELS_ARE_LIFTED
```

把输出的 `suggested_left_sign` 和 `suggested_right_sign` 填入
`onboard_runtime.yaml`。再测量启动死区并修改 `minimum_moving_pwm`；初期不要把
`pwm_limit` 提高到 30 以上。

## G. 首次低速落地

至少两个人在场：一人看车，一人随时按 `Ctrl+C` 或切断电机电源。先只测试直道，
配置建议：

```yaml
lcc:
  base_speed: 0.20
wheels:
  pwm_limit: 18
safety:
  watchdog_timeout: 3.0
  maximum_inference_time: 2.5
```

确认标定、干运行、轮子方向和急停全部通过后，才允许：

```bash
python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE
```

测试顺序固定为：直道 → 单一左弯 → 单一右弯 → 连续弯道 → 路口显式路线选择。
不要一开始测试锥桶、云端决策或全场自主行驶。

## H. 下一次继续开发时需要带回的文件

完成树莓派干运行后，把下面文件复制回本仓库：

```text
car/autodrive/onboard_runtime.yaml
car/autodrive/onboard_calibration.yaml
outputs/onboard_capture/onboard_calibration.mp4
outputs/onboard_runtime/pi_self_check.json
outputs/onboard_runtime/onboard_log.csv
outputs/onboard_runtime/latest.jpg
outputs/onboard_runtime/wheel_check.json
```

然后从以下任务继续：

1. 用真实车载视频评估 YOLOPv2 的可行驶区域 IoU 和车道线 F1。
2. 根据日志标定横向/航向增益、PWM 死区和左右轮比例。
3. 用车载日志验证已有光流传播；若漂移仍大，再加入场地黄色边界快速跟踪。
4. 直道和弯道稳定后，将现有长尾检测/云端决策作为上层监督接入。
5. 最后做路口状态机、锥桶绕行、消融实验和论文指标统计。

如果需要让 Codex 接着处理，直接说：

> 我已经在树莓派完成到手册 E/F 步，这是自检、日志、最新帧和标定文件，请从
> `car/autodrive/RASPBERRY_PI_HANDOFF.md` 的 H 节继续。

## 官方环境参考

- PyTorch Raspberry Pi 教程：
  https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html
- Raspberry Pi 相机软件：
  https://www.raspberrypi.com/documentation/computers/camera_software.html
- Raspberry Pi I2C 配置：
  https://www.raspberrypi.com/documentation/configuration/computers/raspberry-pi.html
