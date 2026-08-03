# LCC 硬件适配层

`car/control/` 现在只保留外圈 LCC 仍在使用的硬件适配：

- `vehicle_control/camera.py`：线程化 OpenCV 相机取帧；
- `vehicle_control/hardware.py`：按 `前左、后左、前右、后右` 顺序输出四轮 PWM；
- `vehicle_control/settings.py`：相机默认分辨率和帧率；
- `utils/Raspbot_Lib.py`：Raspbot I2C 电机与云台底层接口。

这里不再实现 `forward`、`lane-left`、`lane-right` 等预编排动作，也不再提供独立网页。
自动驾驶、状态机、安全门和网页启动入口都位于 `car/autodrive/`：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
./run.sh
```

不要同时启动其他会打开同一摄像头或写同一 I2C 电机控制器的程序。
