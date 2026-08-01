# 树莓派无缝衔接手册

## 当前交接点

这里已经完成并通过无硬件验证：

- YOLOPv2 支持直接处理 OpenCV 帧，不再需要每帧写临时图片。
- 外部视角只输出分割；车载视角才允许生成 LCC 指令。
- 可行驶区域连通域、中心线、预瞄点、横向误差和航向误差。
- LCC、低置信度停车、PWM 安全映射、稳定帧恢复门和推理看门狗。
- 保留四轮独立直行 trim，并由同侧前后轮共同产生差动纠偏。
- 不依赖 YOLOPv2 的低延迟外圈控制通路。
- 外圈黄色双边界融合、内侧路口断口桥接和边界内控制走廊。
- 四点透视标定、视频离线回放、实时 CSV/状态文件和最新叠加帧。
- 电机默认关闭；真实电机需要有效标定与显式确认。

当前树莓派上的 LCC 软件链、USB 相机和 I2C 控制已经可用。电机映射已实测为
`0=前左、1=后左、2=前右、3=后右`，四轮 `16/16/20/20` 可稳定直行；相机 S1=25 度
为正前方。软件已修复三项整圈阻塞因素：逐帧 0.25 秒阻塞渐变、单帧错误导致进程级
永久锁止，以及过大的原图黄线安全区把正常弯道误判为压线。

当前 `onboard_calibration.yaml` 是 25 度之前的旧视角标定，没有相机姿态元数据，真实
电机模式会主动拒绝它。下一项任务是：**按 C、D 节用 S1=25 度重新采集和标定，完成
100 帧空跑后，再按 3 秒、10 秒、整圈逐级落地验证。**

## A. 系统与文件准备

当前验证环境是 64 位 Raspberry Pi OS、Conda `car`、Python 3.9.25、NumPy 2.0.2
和 CPU 版 PyTorch 2.8.0。LCC 路径支持 Python 3.9-3.11，不需要为运行本模块单独升级
Python 或降级 NumPy。

```bash
getconf LONG_BIT
python --version
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
uname -m
```

期望输出包含 `64`、Python `3.9.x` 至 `3.11.x` 和 `aarch64`。当前终端必须使用
`/home/pi/miniconda3/envs/car/bin/python`。

开启底盘使用的 ARM I2C，然后重启：

```bash
sudo raspi-config nonint do_i2c 0
sudo reboot
```

模型文件被 Git 忽略，当前运行配置直接使用外接盘上的真实权重：

```text
/media/pi/FSDBY/weights/yolopv2.pt
```

## B. 使用现有 car 环境

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
source /home/pi/miniconda3/etc/profile.d/conda.sh
conda activate car
which python
python -m pip check
```

依赖已经安装在 `car` 环境；先以 `python -m pip check` 和 `pi_self_check.py` 为准，
不要再创建并行的 Pi 虚拟环境或套用旧的 NumPy 版本约束。

当前机器配置是 `car/autodrive/onboard_runtime.yaml`，其中权重仍指向外接盘真实文件，
但外圈实时控制使用 `model.control_mode: surface-only`，不加载模型。透视配置指向
`car/autodrive/onboard_calibration.yaml`。相机位置、角度或分辨率改变后，必须重新执行
C、D 两节。

先运行完全只读的检查：

```bash
python car/autodrive/pi_self_check.py \
  --config car/autodrive/onboard_runtime.yaml \
  --json-output outputs/onboard_runtime/pi_self_check.json
```

完成 S1=25 度新标定后所有检查都应为 `PASS`。在此之前，
`perspective_calibration` 必须以缺少/不匹配 `camera_pose` 明确失败，这正是禁止误用
旧标定的保护。当前实时模式是 `surface-only`，不会加载 YOLOPv2 权重。

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

如果相机云台朝左或朝右，先确认云台周围没有线缆、手指或其他障碍，再用软件把水平
S1 舵机回到中位起点；该命令不会驱动车轮：

```bash
python car/autodrive/align_camera_gimbal.py \
  --confirm-camera-gimbal-clear CAMERA_GIMBAL_IS_CLEAR \
  --pan-angle 25
```

检查 `outputs/onboard_capture/gimbal_alignment.jpg`。当前车已实测 25 度为正前方；若
相机仍偏左或偏右，以 5-10 度小步调整，避免直接扫到机械限位。若光轴固定在不可转动支架
上，软件旋转/透视变换无法恢复视野外的前方道路，必须调整支架。

当前车的 `camera.gimbal.initialize_on_startup: true`，所以 `run_onboard.py` 使用实时
相机时会在取帧前自动发送配置的 S1 水平角度。该动作不驱动车轮，但即使在轮电机干运行
模式下也会移动云台；运行前同样要保证云台周围无障碍。视频回放不会访问云台。确认实际
当前运行配置已同步 S1=25 度。垂直角保持 `tilt_angle: null`。

相机固定后不要再改变高度、俯仰角和横滚角。采集标定图片与视频：

```bash
python car/autodrive/capture_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --confirm-camera-gimbal-clear CAMERA_GIMBAL_IS_CLEAR \
  --seconds 10
```

应生成：

```text
outputs/onboard_capture/onboard_calibration.mp4
outputs/onboard_capture/onboard_calibration_frame.jpg
```

该命令会先应用运行配置中的 S1=25 度，再用同一分辨率和图像变换采集，从源头避免
标定视角与运行视角不一致。

## D. 四点透视标定

在有桌面显示的树莓派上运行：

```bash
python car/autodrive/calibrate_perspective.py \
  outputs/onboard_capture/onboard_calibration_frame.jpg \
  --runtime-config car/autodrive/onboard_runtime.yaml \
  --output car/autodrive/onboard_calibration.yaml \
  --force
```

依次点击地面道路梯形的左上、右上、右下、左下，按 Enter 保存。新文件会写入相机
姿态元数据；25 度、分辨率或图像变换不匹配时，真实电机模式会拒绝启动。检查生成的
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
- `perception_source` 为 `surface-only`，`boundary_source` 为 `both`、
  `outer+width` 或短时 `history`。
- `inference_ms` 接近 0，`boundary_ms` 通常小于 15 ms。
- 看门狗没有在正常推理时频繁触发。

如果推理超时，应先降低 `model.img_size`、调整 Torch 线程或降低车速；不能简单增大
超时后直接开车。

需要保存逐帧诊断时增加 `--save-debug-frames`。当前 `surface-only` 模式不需要光流；
只有把 `model.control_mode` 切回 `yolopv2` 做对照实验时才使用 `--temporal`。

## F. 轮子映射与当前标定

可选的架空检查命令：

```bash
python car/autodrive/check_wheel_directions.py \
  --confirm-wheels-lifted WHEELS_ARE_LIFTED
```

当前已经结合旧代码和落地运动确认电机顺序为 `0=前左、1=后左、2=前右、3=后右`。
配置使用 `drive_mode: four-wheel-trim`，直行四轮基准为 `16/16/20/20`。
固定右转实验证明，仅调整前轮或让四轮保持同向、只制造左右 PWM 差时，转弯半径仍大于
赛道。仓库原始 `McLumk_Wheel_Sports.rotate_right()` 则明确使用 `[+,+,-,-]`。
落地验证 `16/16/-10/-10` 能明显右旋，但约 0.67 秒内车头转约 30 度而几乎不前进，
不适合作为 LCC 常规输出。因此常规模式保持四轮正转，并用
`front_steering_delta_pwm: 20`、`maximum_steering_delta_pwm: 10` 输出最强正向右弧
`26/26/10/10`。落地测试中该输出连续行驶约 1.47 秒，能够一边前进一边形成清晰右弧；
原生偏航模式只保留作诊断。

## G. 首次低速落地

至少两个人在场：一人看车，一人随时按 `Ctrl+C` 或切断电机电源。先只测试直道，
配置建议：

```yaml
lcc:
  base_speed: 0.12
wheels:
  drive_mode: four-wheel-trim
  pwm_limit: 35
safety:
  watchdog_timeout: 3.0
  maximum_inference_time: 2.5
  resume_valid_frames: 4
```

确认标定、干运行、轮子方向和急停全部通过后，才允许：

```bash
python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE \
  --max-runtime-seconds 3
```

先跑 3 秒，再改成 10 秒；两段日志正常后，回到外圈起点并用足以覆盖一圈的时间上限
（例如 `--max-runtime-seconds 60`，现场人员仍随时准备 `Ctrl+C`）。每段结束检查
`onboard_log.csv`、`latest.jpg` 和 `latest_birdeye.jpg`。黄线进入原图底部车体足迹、估计
超限或丢路会立即停车；只有随后连续 4 个有效帧才会重新起步，持续故障会一直保持停车。

完成命令行短测后，可以启动 LCC 专用实时网页：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_lcc_web.py \
  --config car/autodrive/onboard_runtime.yaml \
  --host 0.0.0.0 --port 8080 \
  --default-max-runtime-seconds 60 \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE
```

浏览器访问 `http://<车辆IP>:8080`。该页面只管理 LCC，不创建旧的直行/变道控制器；
待机不占用摄像头，启动后显示 LCC 标注帧、鸟瞰图和 `status.json` 实时状态。

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
2. 用 S1=25 度重新标定后，按 3 秒/10 秒验证边界模式，再采集完整一圈日志。
3. 根据整圈日志微调四轮差动纠偏量和黄线安全裕量，不改变四轮直行基准。
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
