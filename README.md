# VehicleCloudCollaboration

这是一个车云协同实验仓库，车端闭环由摄像头、长尾检测、云端 mock 决策、车辆控制和网页控制台组成。

![整体架构图](arch.svg)

## 目录结构

```text
VehicleCloudCollaboration/
├── car/
│   ├── run_closed_loop.py
│   ├── cloud_client/
│   ├── longtail/
│   └── control/
└── cloud/
```

## 车端闭环

中期演示入口：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration/car
python run_closed_loop.py
```

默认流程：

1. `CameraStream` 读取本地摄像头画面。
2. `vehicle_control` 网页控制台展示摄像头画面、车辆状态和闭环开始按钮。
3. 点击网页里的“开始闭环”后，车辆才会进入自动行驶。
4. `LongTailClassifier` 输出长尾分数。
5. 分数达到阈值时调用 `cloud_client` 的 mock `/chat` 接口。
6. mock 决策返回。
7. 车端将 `left` 映射为 `lane-left` 并通过 `VehicleController` 执行。

常用参数：

```bash
python run_closed_loop.py
python run_closed_loop.py --threshold 0.6
python run_closed_loop.py --cloud-mode none
python run_closed_loop.py --web-port 8081
python run_closed_loop.py --no-web
python run_closed_loop.py --start-immediately
python run_closed_loop.py --no-stop-for-detection
```

当前默认变道参数参考见 `car/control/vehicle_control/lane_change_reference.yaml`。

网页控制台默认地址：

```text
http://<车辆IP>:8080
```

## 车端模块

### car/longtail

长尾检测模块，核心类是 `LongTailClassifier`。它读取 `config.yaml`，组合多个检测器，对单张图片输出：

- `is_long_tail`
- `score`
- `threshold`
- `individual_scores`
- `inference_time`
- `fps`

模块说明见 [car/longtail/README.md](/home/pi/Desktop/VehicleCloudCollaboration/car/longtail/README.md)。

### car/cloud_client

车端云服务客户端模块。当前包含 mock 客户端 `CloudMockClient`，用于调用通用 LLM mock `/chat` 服务并生成车辆动作决策。

模块说明见 [car/cloud_client/README.md](/home/pi/Desktop/VehicleCloudCollaboration/car/cloud_client/README.md)。

### car/control

车辆控制模块，包含底层底盘封装、动作控制、摄像头流和网页控制服务。

支持动作：

- `forward`
- `stop`
- `lane-left`
- `lane-right`

模块说明见 [car/control/README.md](/home/pi/Desktop/VehicleCloudCollaboration/car/control/README.md)。

## 云端目录

`cloud/` 放置云端推理与服务端代码。车端中期闭环默认使用 `car/cloud_client/mock_client.py` 调用 mock 服务。

云端说明见 [cloud/README.md](/home/pi/Desktop/VehicleCloudCollaboration/cloud/README.md)。

## 运行检查

只检查入口参数和语法，不启动电机：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
python -m py_compile car/run_closed_loop.py car/cloud_client/mock_client.py
python car/run_closed_loop.py --help
```

## 闭环数据测试

不启动车辆硬件的完整数据闭环测试：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
python car/test/closed_loop_test.py
```

测试说明见 [car/test/README.md](/home/pi/Desktop/VehicleCloudCollaboration/car/test/README.md)。

## 许可证

本项目仅供研究使用。
