# 车辆控制说明

`car/control/` 目录现在已经重构为一套独立的本地车控层，不再以 `path_follow` 作为主控制方案。

## 当前目录职责

当前主要子目录与入口如下：

- `vehicle_control/`：新的车控核心逻辑、相机取流与网页控制服务
- `run_vehicle_control.py`：网页控制入口
- `train_car/`：TODO: 后续替换自动驾驶端到端小模型
- `utils/`：Raspbot 底层驱动与辅助工具
- `donkeycar/`：DonkeyCar 源码与模板


## 四个基础动作

当前车控层提供 4 个基础动作：

- `forward`：车辆正常直行
- `stop`：车辆停止
- `lane-left`：车辆左变道
- `lane-right`：车辆右变道

这些动作已经内置了当前验证过的一组参数，其中右变道与回正参数已经按实车测试结果固化。

## Python 调用方式

推荐直接通过 `VehicleController` 调用：

```python
from vehicle_control.controller import VehicleController

controller = VehicleController()

controller.drive_forward()
controller.lane_left()
controller.lane_right()
controller.stop()
```

如果你的上层逻辑更适合统一分发动作名，也可以使用字符串接口：

```python
controller.execute("forward")
controller.execute("lane-left")
controller.execute("lane-right")
controller.execute("stop")
```

这更适合后续把长尾检测结果直接映射成车辆动作。

## 推荐接入方式

建议采用如下调用链：

1. 检测模块输出动作决策
2. 上层逻辑将结果映射为 `forward`、`stop`、`lane-left` 或 `lane-right`
3. 通过 `VehicleController.execute(...)` 执行对应底层动作

例如：

```python
action = "lane-right"
controller.execute(action)
```

## 网页控制页面

启动方式：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration/car/control
python run_vehicle_control.py
```

启动后在浏览器访问：

```text
http://<车辆IP>:8080
```

页面提供以下能力：

- 展示摄像头实时流画面
- 展示当前车辆控制状态
- 提供四个基础控制按钮

## HTTP API

当前网页服务同时提供 HTTP 接口，便于外部程序直接调用：

- `GET /api/state`：获取当前车辆状态
- `POST /api/control?action=forward`：执行正常行驶
- `POST /api/control?action=stop`：执行停止
- `POST /api/control?action=lane-left`：执行左变道
- `POST /api/control?action=lane-right`：执行右变道

## 相机流说明

当前网页中的视频流来自本地摄像头，默认使用 OpenCV 的 `VideoCapture(0)`。

如果后续需要调整摄像头设备号、分辨率或帧率，可以修改：

- [car/control/vehicle_control/settings.py](/home/pi/Desktop/VehicleCloudCollaboration/car/control/vehicle_control/settings.py)

## 当前建议

如果你的目标已经变成“长尾检测结果直接控制车辆动作”，那么建议：

1. `vehicle_control/` 作为后续唯一主控制层
2. 旧的 `path_follow` 不再作为主方案保留
3. 不再需要的旧测试脚本可以逐步清理
