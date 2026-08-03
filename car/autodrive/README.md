# 外圈 LCC

此包只维护当前树莓派实车使用的外圈车道居中控制。它直接处理车载相机中的黄色道路
边界和绿色岛区，不依赖 DonkeyCar、云端变道或旧网页。

## 目录结构

```text
autodrive/
├── camera/       # 云台姿态与图像方向变换
├── config/       # 当前实车配置、透视标定及示例
├── control/      # 中心线控制器、PWM 映射和安全门
├── perception/   # 外圈边界、透视映射和可视化
├── runtime/      # 相机到四轮输出的实时闭环
├── tools/        # 标定、采集、自检和架空轮测试
├── web/          # LCC 网页服务与前端
├── run_onboard.py
└── run_lcc_web.py
```

根目录的两个 Python 文件只是稳定命令行入口，具体实现放在对应子包中。

## 日常启动

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
./run.sh
```

浏览器访问 `http://<车辆IP>:8080`，勾选安全确认后点击“启动 LCC”。网页启动后才占用
相机；停止/急停、服务退出、运行时间到期或子进程异常都会执行四轮归零。

## 命令行与离线回放

直接实车运行：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE \
  --max-runtime-seconds 60
```

某轮出现问题后，使用归档原始视频逐帧干运行，不会驱动车轮或覆盖实车记录：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --video outputs/onboard_runtime/runs/<当次目录>/raw.mp4 \
  --sample-every 1 \
  --output-dir /tmp/lcc_replay \
  --no-run-archive
```

## 标定和诊断工具

- `tools/pi_self_check.py`：只读检查依赖、配置、相机和 I2C。
- `tools/align_camera_gimbal.py`：安全确认后调整云台。
- `tools/capture_onboard.py`：按运行配置采集标定图和视频。
- `tools/calibrate_perspective.py`：生成四点鸟瞰标定。
- `tools/check_wheel_directions.py`：车轮架空后的单轮方向检查。

例如重新采集和标定：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/tools/capture_onboard.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --confirm-camera-gimbal-clear CAMERA_GIMBAL_IS_CLEAR \
  --seconds 10

/home/pi/miniconda3/envs/car/bin/python car/autodrive/tools/calibrate_perspective.py \
  outputs/onboard_capture/onboard_calibration_frame.jpg \
  --runtime-config car/autodrive/config/onboard_runtime.yaml \
  --output car/autodrive/config/onboard_calibration.yaml \
  --force
```

相机机械位置、S1 角度、分辨率或图像旋转改变后必须重新标定。当前水平云台正前方为
`25°`。

## 输出和安全

每次网页或命令行实车运行都会在 `outputs/onboard_runtime/runs/` 保存原始、标注、鸟瞰
视频、逐帧 CSV、最终状态、配置与标定快照。`outputs/onboard_runtime/latest.jpg`、
`latest_birdeye.jpg`和`status.json`供网页实时显示。

录像、CSV 和网页快照均由后台线程写入，不会阻塞电机控制；存储持续跟不上时会优先保留
最新诊断帧，并在 `status.json` 的 `diagnostics` 和 CSV 的队列/丢帧列中记录。归档目录内
的 CSV 只包含实际写入视频的帧，因此时间戳始终与 MP4 一一对应。CSV 还记录采帧间隔、
感知、控制门、硬件下发和鸟瞰渲染耗时，用于直接区分相机、算法、I2C 与诊断 I/O 卡顿。

当前底盘映射为：

- `0=左前、1=左后、2=右前、3=右后`
- 直行 `16/16/20/20`
- 普通最大右弧 `26/26/10/10`
- 饱和急弯 `30/30/0/0`，任何车轮都不反转

单边界恢复最多 `8s`；弯顶后近场航向连续对齐且至少一条新鲜边界可用时，立即交还普通
LCC，不再等待双边界。连续 `1s`没有进展、黄线进入车底、边界真正消失、相机断流或
看门狗超时都会停车。停车后需要至少 4 帧来源一致、误差连续的有效感知才会重新起步。
