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
| `raspbot_pi5` | `/dev/i2c-1`，`Ctrl_Muto`/`Ctrl_Servo` | 历史 Pi 配置；不自动适用于当前 Jetson |
| `rosmaster_jetson` | `/dev/myserial`，`set_motor`/`set_pwm_servo` | 基础车型配置，完整标定标志仍为 false |
| `rosmaster_jetson_yolopv2_visual_feedback` | 同上 | 当前面板驾驶配置；视觉反馈及云辅助恢复 |

Jetson profile 的 `status.motion_calibrated` 和 `status.gimbal_calibrated` 默认是
`false`。正常运行即使传入 `--enable-motors` 也会拒绝输出。架空轮/云台工具及受单独限制的 field-trial 入口有各自确认与门禁；当前面板使用已配置的 field-trial 过程，不能把 `motion_calibrated=false` 误解为所有入口绝不运动。见[运行说明](../../docs/operations/runtime.md)。

## 选择车型

命令行 dry-run：

```bash
python3 car/autodrive/run_onboard.py \
  --vehicle-profile rosmaster_jetson \
  --max-runtime-seconds 20
```

只读实时面板（HTTP服务已运行时直接访问，无需重复启动）：

```bash
./vehicle dashboard
```

Jetson 实验通过统一入口协调监控设备占用，默认电机禁用：

```bash
./vehicle drive --max-runtime-seconds 30
```

`vehicle dashboard` 启动 HTTP 面板，本身不连接设备。当前页面通过独立 supervisor 接受会话开始/结束/心跳；开始可按 supervisor 模式调用硬件入口。页面不直接提交 PWM。遵循当次现场安排、确认参数及门禁，详见[运行说明](../../docs/operations/runtime.md)。

## 完整标定流程（仍需独立现场验收）

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
