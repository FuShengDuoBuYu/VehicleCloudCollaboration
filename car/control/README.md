# control

`car/control/` 是车端本地控制模块，负责底盘动作、摄像头取流、车辆状态和网页控制服务。

## 目录结构

```text
control/
├── run_vehicle_control.py
├── vehicle_control/
│   ├── controller.py
│   ├── camera.py
│   ├── hardware.py
│   ├── settings.py
│   ├── web.py
│   └── static/index.html
├── utils/
├── train_car/
└── donkeycar/
```

## 车辆动作

`VehicleController.execute(action)` 支持四个动作：

- `forward`：直行
- `stop`：停止
- `lane-left`：左变道
- `lane-right`：右变道

Python 调用示例：

```python
from vehicle_control.controller import VehicleController

controller = VehicleController()
controller.execute("forward")
controller.execute("lane-left")
controller.execute("lane-right")
controller.execute("stop")
```

## 摄像头

`CameraStream` 使用 OpenCV 读取本地摄像头，提供两个主要接口：

- `get_jpeg_bytes()`：给网页视频流使用
- `get_frame()`：给闭环检测逻辑使用

默认摄像头参数在 `vehicle_control/settings.py`：

- `camera_index = 0`
- `width = 640`
- `height = 480`
- `fps = 20`
- `jpeg_quality = 80`

## 网页控制台

启动网页控制服务：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration/car/control
python run_vehicle_control.py
```

指定端口和摄像头：

```bash
python run_vehicle_control.py --port 8081 --camera-index 0
```

浏览器访问：

```text
http://<车辆IP>:8080
```

页面能力：

- 显示摄像头实时画面
- 显示当前动作和状态
- 触发闭环 `start`、`pause`
- 触发 `forward`、`stop`、`lane-left`、`lane-right`

## HTTP API

- `GET /api/state`
- `GET /api/closed-loop`
- `POST /api/closed-loop?action=start`
- `POST /api/closed-loop?action=pause`
- `POST /api/control?action=forward`
- `POST /api/control?action=stop`
- `POST /api/control?action=lane-left`
- `POST /api/control?action=lane-right`
- `GET /stream.mjpg`

## 闭环调用

`../run_closed_loop.py` 会直接创建并共享以下对象：

- `VehicleController`
- `CameraStream`
- `VehicleControlServer`

这样闭环检测和网页控制台使用同一个摄像头流与同一个车辆控制器。
