# VehicleCloudCollaboration

当前车端自动驾驶只使用外圈 LCC（Lane Centering Control）。系统从车载相机提取黄色
道路边界和绿色岛区，在鸟瞰坐标中估计道路中心线，再将横向/航向误差映射为四轮 PWM。
DonkeyCar、云端变道控制和旧网页均不参与当前运行。

运行链路：

```text
LCC 网页 -> run_onboard.py -> 外圈边界/中心线 -> LCC -> 安全门 -> Raspbot 四轮底盘
```

## 启动 LCC 网页

车辆放到安全起点、确认摄像头云台和车轮周围无障碍后运行：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration

/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_lcc_web.py \
  --config car/autodrive/onboard_runtime.yaml \
  --host 0.0.0.0 \
  --port 8080 \
  --default-max-runtime-seconds 60 \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE
```

浏览器访问 `http://<车辆IP>:8080`。页面只提供 LCC 启动、停止/急停、实时相机图、
鸟瞰图和运行状态。服务待机时不占用摄像头；关闭服务或按下急停会停止四轮输出。

## 直接运行

不使用网页时可直接启动闭环：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.yaml \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE \
  --max-runtime-seconds 60
```

去掉 `--enable-motors` 和确认参数即为电机干运行。`Ctrl+C`、运行时间到期、相机断流、
道路丢失或误差超限都会触发停车。

## 当前实车标定

- 相机水平舵机 S1 正前方角度：`25°`
- 电机编号：`0=左前、1=左后、2=右前、3=右后`
- 直行基准 PWM：`16/16/20/20`
- 最大正向右弧 PWM：`26/26/10/10`
- 运行配置：`car/autodrive/onboard_runtime.yaml`
- 透视标定：`car/autodrive/onboard_calibration.yaml`

相机位置、角度、分辨率或画面旋转发生变化后，必须重新采集图片并完成透视标定。

## 目录

- `car/autodrive/`：LCC、网页、安全控制及标定工具
- `car/control/vehicle_control/`：相机和 Raspbot 底盘适配
- `car/control/utils/Raspbot_Lib.py`：I2C 底层接口
- `car/test/`：当前 LCC、相机和硬件映射测试

详细操作见 [car/autodrive/README.md](car/autodrive/README.md)。

## 软件验证

以下命令不驱动车轮：

```bash
/home/pi/miniconda3/envs/car/bin/python -m unittest \
  car.test.test_lane_centering \
  car.test.test_camera_gimbal \
  car.test.test_camera_transform \
  car.test.test_hardware_mapping \
  car.test.test_lcc_web
```

本项目仅供受控场地研究使用。真车运行时必须有人在车辆旁准备急停或切断电机电源。
