# 双车型硬件适配层

`car/control/vehicle_control/` 把算法与两辆物理小车隔离。上层固定使用逻辑轮顺序
`前左、后左、前右、后右`，平台后端负责协议、原生限幅、电机顺序和符号转换。

```text
LCC normalized command
        |
SafeWheelDriver + profile wheel calibration
        |
FourWheelChassis (FL, RL, FR, RR)
        |
        +-- raspbot_pi5      -> I2C / Ctrl_Muto ([-255, 255])
        +-- rosmaster_jetson -> serial / set_motor ([-100, 100])
```

## 目录

- `vehicle_control/base.py`：四轮底盘和双轴云台公共契约、限幅、渐变和停车；
- `vehicle_control/platforms/raspbot.py`：Raspberry Pi 5 Raspbot I2C 后端；
- `vehicle_control/platforms/rosmaster.py`：Jetson Rosmaster X3 串口后端；
- `vehicle_control/factory.py`：按 profile 创建底盘/云台，并执行标定门禁；
- `vehicle_control/profile.py`：加载 profile，并把车型覆盖项深度合并到公共运行配置；
- `vehicle_control/profiles/*.yaml`：设备连接、轮序、符号、云台限制和车型标定；
- `vehicle_control/hardware.py`：保留旧 `RospbotChassis` 名称的兼容导出；
- `vehicle_control/camera.py`：线程化 OpenCV 相机取帧。

## 内置车型

| Profile | 底层协议 | 当前状态 |
| --- | --- | --- |
| `raspbot_pi5` | `/dev/i2c-1`，`Ctrl_Muto`/`Ctrl_Servo` | 沿用已有实车标定 |
| `rosmaster_jetson` | `/dev/myserial`，`set_motor`/`set_pwm_servo` | 相机可 dry-run；轮序、方向、云台和透视待标定 |

Jetson profile 的 `status.motion_calibrated` 和 `status.gimbal_calibrated` 默认是
`false`。正常运行即使传入 `--enable-motors` 也会拒绝输出。只有带现场确认字符串的
架空轮和云台工具允许在标定前创建硬件后端。

## 选择车型

命令行 dry-run：

```bash
python3 car/autodrive/run_onboard.py \
  --vehicle-profile rosmaster_jetson \
  --max-runtime-seconds 20
```

只读实时面板（HTTP服务已运行时直接访问，无需重复启动）：

```bash
./run.sh
```

Jetson 实验通过统一入口协调监控设备占用，默认电机禁用：

```bash
./run_vehicle_experiment.sh --max-runtime-seconds 30
```

`run.sh` 只读取状态文件；旧网页电机启停功能已删除。真实运动使用命令行入口并遵循当次
现场安排、确认参数及 profile 标定门禁，详见 [面板说明](../../docs/vehicle-dashboard.md)。

## Jetson 标定顺序

1. 保持 profile 的两个 calibrated 标志为 `false`，先运行自检和相机 dry-run。
2. 架空四轮后，使用 `check_wheel_directions.py --vehicle-profile
   rosmaster_jetson --confirm-wheels-lifted WHEELS_ARE_LIFTED` 逐轮确认轮序和方向。
3. 清空云台运动范围后，使用 `align_camera_gimbal.py --vehicle-profile
   rosmaster_jetson --confirm-camera-gimbal-clear CAMERA_GIMBAL_IS_CLEAR` 确认通道与角度。
4. 采集 Jetson 相机图像，生成独立透视标定文件，并写入 Jetson profile。
5. 架空轮、落地低速直线和弯道验证均通过后，才更新 wheel 参数并把相应状态改为
   `true`。

每次运行都会归档公共配置、解析后的最终配置和 vehicle profile，并在 CSV/status 中记录
车型 ID。公平对比应使用同一个 Git commit 和算法配置，只切换 `--vehicle-profile`。
