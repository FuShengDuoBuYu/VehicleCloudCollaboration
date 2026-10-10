# 车端离线回放与转弯模拟

本工具在充电期间使用已保存的数据调试，不连接相机、串口或电机。实时驾驶仍使用当前 YOLOPv2 道路／车道／目标结果，车控负责路径提案、停止、IMU 相对角转向和恢复确认。

当前平台分为两种证据：真实录像与控制日志回放、合成观测驱动的控制状态检查。历史视频不会因候选指令改变下一帧；合成四角流程没有车辆位移模型。它们不能代替完整实车外圈验收，也不会从 PWM 虚构地图上的车辆位置。

## 查看已生成的实验包

直接在浏览器打开 `outputs/corner_fix/20261009/stationary-turn/viewer/index.html`。页面支持两路视频同步、逐帧查看、实际与候选 PWM 对照、IMU 时间图和合成四角流程。场地图仅展示布局；旧记录缺失的编码器或姿态保持未知。

若浏览器限制本地媒体，可在仓库根目录启动独立只读文件服务：

```bash
python3 -m http.server 8081 --bind 127.0.0.1 --directory outputs/corner_fix/20261009/stationary-turn/viewer
```

然后访问 `http://127.0.0.1:8081/`。此命令不更改已有服务。网页视频为保持原帧数和帧率的 H.264 派生副本，原始视频和无损模型输入仍保留并作为回放依据。

## 用已有输入重跑控制

下例只消费 step17 已归档的无损图像、掩码、序号、时效和 IMU，不运行模型，不调用云端，也不连接底盘。输出目录必须是新的目录。

```bash
outputs/field_yolopv2/venv/bin/python -B car/autodrive/tools/replay_outer_loop.py \
  --run outputs/corner_fix/20261009/increments/step17/runtime/runs/20261009_083038_547297_hardware \
  --profile outputs/corner_fix/20261009/candidate-320-stationary.yaml \
  --mode recorded \
  --output outputs/outer_loop_replays/manual-step17
```

运行合成状态检查：

```bash
outputs/field_yolopv2/venv/bin/python -B outputs/corner_fix/20261009/stationary-turn/simulation/simulate_four_corners.py
```

后一个命令更新派生检查结果，不覆盖原始实车记录。它验证同一个运行时对象连续四个右转角、新观测计数、暂停保留角度以及转后恢复；输入角度由脚本提供，不是物理仿真预测。

## 可调参数与边界

候选配置为 `outputs/corner_fix/20261009/candidate-320-stationary.yaml`，其他默认 profile 不自动启用原地转向。

| 参数 | 当前值 | 用途 |
| --- | ---: | --- |
| 四轮直行基准／输出上限 | 30 PWM | 用户已确认的运行基准；停止为 0 |
| `wheels.stationary_pivot_pwm` | 30 | 右转输出 `[30,30,-30,-30]`，需要真实方向证据 |
| `stationary_corner.front_near_ratio` | 0.74 | 当前横向前边界在图中的接近阈值，不是米制距离 |
| `front_max_ratio` | 0.85 | 超过范围保持停止 |
| `enter_observations` | 2 | 两个不同的新鲜近弯观测确认进入转向准备 |
| `settle_seconds` | 0.25 | 归零等待；不代表独立测得的物理停稳时间 |
| `target_yaw_rad` | 1.570796 | IMU 相对转角目标 90°，实际精度尚待复测 |
| `yaw_sign` | -1 | 原始右旋 IMU 标定方向 |
| `exit_observations`／普通恢复 | 3／5 | 出口确认三帧后，再用五个新鲜普通安全观测恢复运动 |

识别横向前边界但尚未看到右出口时，允许沿当前确实存在的直行通道接近；到近弯而出口未确认时停车。已经进入接近状态后，前边界丢失会保持停止，不退回旧弧线转向。原地转向依赖当前有效图像和 IMU，不能用历史掩码补出不可见道路。

数据目录清单见 `outputs/corner_fix/20261009/stationary-turn/catalog/`；软件快照、回放及验收范围见 `coordination/2026-10-10-offline-simulation-report.md`。
