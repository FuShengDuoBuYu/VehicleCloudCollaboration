# 外圈 LCC

此目录只维护当前树莓派实车使用的外圈车道居中控制。感知直接处理车载相机画面中的
黄色道路边界和绿色岛区，不加载 DonkeyCar 或 YOLO 模型，也不调用云端变道接口。

## 运行链路

```text
车载相机
  -> 云台回到 S1=25°、统一图像方向
  -> 透视映射与外圈边界跟踪
  -> 道路中心线、横向误差和航向误差
  -> LCC 转向量
  -> 感知恢复门与看门狗
  -> 四轮 trim PWM
```

主要运行文件：

- `run_lcc_web.py`、`lcc_web.py`、`lcc_web.html`：唯一的实时网页入口
- `run_onboard.py`：实时 LCC 主循环；默认不启用电机
- `outer_loop.py`、`lane_centering.py`、`perspective.py`：感知与控制
- `drive_runtime.py`：四轮 PWM 映射、恢复门和看门狗
- `onboard_runtime.yaml`、`onboard_calibration.yaml`：当前实车配置与标定

标定和排障工具：

- `align_camera_gimbal.py`：安全确认后调整水平/垂直云台
- `capture_onboard.py`：按运行配置采集标定图和视频
- `calibrate_perspective.py`：四点透视标定
- `pi_self_check.py`：只读检查依赖、配置、相机和 I2C
- `check_wheel_directions.py`：车轮架空后的单轮方向检查

## 启动网页

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration

/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_lcc_web.py \
  --config car/autodrive/onboard_runtime.yaml \
  --host 0.0.0.0 --port 8080 \
  --default-max-runtime-seconds 60 \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE
```

访问 `http://<车辆IP>:8080`。页面启动 LCC 后才打开摄像头。停止/急停、服务退出、
运行时间到期或子进程异常都会执行四轮归零。

## 命令行运行

先进行不驱动车轮的检查：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/pi_self_check.py \
  --config car/autodrive/onboard_runtime.yaml

/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --max-samples 100
```

确认画面、鸟瞰中心线和停车逻辑正确后才允许真车运行：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE \
  --max-runtime-seconds 60
```

输出写入 `outputs/onboard_runtime/`：

- `latest.jpg`：相机叠加结果
- `latest_birdeye.jpg`：鸟瞰边界与中心线
- `status.json`：网页使用的最新状态
- `onboard_log.csv`：最近一次运行的逐帧误差、置信度和四轮 PWM
- `runs/<时间>_hardware/`：每次实车实验的独立归档，包含 `raw.mp4`、
  `annotated.mp4`、`birdeye.mp4`、`onboard_log.csv`、`status.json`、运行配置和
  透视标定快照

网页和命令行启动都会自动创建归档，不需要再加参数。只有需要逐张无视频压缩的图片时才加
`--save-debug-frames`，图片会放入当次归档的 `frames/`。归档不会自动删除，长期测试前应
检查磁盘空间。

某轮出现问题后，先用其中的 `raw.mp4` 做隔离干运行，不会驱动车轮，也不会覆盖实车记录：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --video outputs/onboard_runtime/runs/<当次目录>/raw.mp4 \
  --sample-every 1 \
  --output-dir /tmp/lcc_replay \
  --no-run-archive
```

修改算法后必须先比较回放 CSV 中的首次停车帧、动作和四轮 PWM，再进行下一轮实车测试。

## 相机与透视重新标定

当前实车的水平 S1=25°。云台周围确认无障碍后可单独回正：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/align_camera_gimbal.py \
  --confirm-camera-gimbal-clear CAMERA_GIMBAL_IS_CLEAR \
  --pan-angle 25
```

相机机械位置、S1 角度、分辨率或图像旋转变化后，重新执行：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/capture_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --confirm-camera-gimbal-clear CAMERA_GIMBAL_IS_CLEAR \
  --seconds 10

/home/pi/miniconda3/envs/car/bin/python car/autodrive/calibrate_perspective.py \
  outputs/onboard_capture/onboard_calibration_frame.jpg \
  --runtime-config car/autodrive/onboard_runtime.yaml \
  --output car/autodrive/onboard_calibration.yaml \
  --force
```

依次点击左上、右上、右下、左下四点并保存。真实电机模式会拒绝使用相机姿态不匹配的
标定文件。

## 当前底盘映射

- `0=左前、1=左后、2=右前、3=右后`
- 直行：`16/16/20/20`
- 普通最大正向右弧：`26/26/10/10`
- 饱和急弯右转：`30/30/0/0`；左转镜像为 `0/0/30/30`
- `drive_mode: four-wheel-trim`

相机断流、边界真正消失、黄线进入车体安全区、横向误差超限或看门狗超时会立即停车；
急弯中边界仍有效时，航向估计允许饱和到 `1.0` 并保持最大转向，不再把道路自身的
急转弯误判成失控。行驶中有效走廊使用 `0.25` 硬置信度底线；停车后仍须达到
`0.35` 并连续保持 4 帧才会恢复行驶。普通弧线仍保持四轮正转；只有饱和急弯才从
`26/26/10/10` 渐进到 `30/30/0/0`，让外侧轮继续推进、内侧轮停转。任何车轮都不反转，
因此不会使用旧版正反轮原地旋转。远端 `0.50` 预瞄只负责提前进入普通弧线；停内轮的
紧转程度由近端 `0.72` 预瞄单独控制，车辆对准出弯直道后会立即减弱。感知盲区续行也只
保留普通 `26/10` 弧线，不会盲目保持 `30/0`。已进入明确的单边界急弯后，
如果只有置信度或横向曲线外推短暂失真，控制器最多保持最近圆弧 `1.6s`。即使边界
横贯画面而无法拟合为 `x(y)`，只要鸟瞰图中仍有至少 `1%` 的黄色边界像素，就标记为
`visible-history` 并等待正常跟踪恢复；黄线进入车底、边界像素完全丢失、推理超时和
看门狗停车不会被该续行逻辑覆盖。

外圈边界处于 `both`、`outer+width` 或 `inner+width` 时，LCC 直接使用已经连续性校验过的
黄色边界曲线计算中心线。绿色路面掩膜仍用于排除岛区和验证走廊，但光照或地垫接缝造成的
零碎区域不再重复否决一条可信中心线。
