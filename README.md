# VehicleCloudCollaboration

这是一个面向车云协同实验的仓库，当前包含三条核心能力链路：

- `cloud/`：云端推理与服务端能力
- `car/longtail/`：车端长尾场景检测与评估
- `car/control/`：车端本地控制、相机流与网页控制界面

![整体架构图](arch.svg)

## 仓库结构

```text
VehicleCloudCollaboration/
├── cloud/
└── car/
    ├── longtail/
    └── control/
```

## cloud

`cloud/` 目录主要用于承载云端服务、模型调用与多端协同逻辑。

如果你需要启动云端推理服务，请优先查看：

- [cloud/README.md](/home/pi/Desktop/VehicleCloudCollaboration/cloud/README.md)

## car/longtail

`car/longtail/` 负责长尾场景检测与评估，保留了原有的检测器、分类器、数据集和测试脚本。

主要内容包括：

- `main.py`：主程序入口
- `classifier.py`：融合分类器
- `benchmark.py`：性能测试
- `evaluate.py`：评估脚本
- `config.yaml`：配置文件
- `detectors/`：各类检测器实现
- `dataset/`：数据集与样例数据

运行示例：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration/car/longtail
pip install -r requirements.txt
python main.py
```

## car/control

`car/control/` 目录现在已经从旧的路径跟随实验，重构为一套独立的本地车控层。

当前主要组成如下：

- `vehicle_control/`：新的车控核心逻辑
- `run_vehicle_control.py`：网页控制服务启动入口
- `train_car/`：原有训练车与 DonkeyCar 驱动整合代码
- `utils/`：Raspbot 底层驱动与工具脚本
- `donkeycar/`：DonkeyCar 源码与模板

新的车控层已经提供 4 个基础动作接口：

- `forward`：正常直行
- `stop`：停止
- `lane-left`：左变道
- `lane-right`：右变道

启动车控页面：

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration/car/control
python run_vehicle_control.py
```

启动后可在浏览器访问：

```text
http://<车辆IP>:8080
```

页面会展示：

- 摄像头实时画面
- 当前车辆控制状态
- 四个基础控制按钮

详细说明请查看：

- [car/control/README.md](/home/pi/Desktop/VehicleCloudCollaboration/car/control/README.md)

## 推荐调用链路

当前更推荐的整体工作流是：

1. `car/longtail/` 负责识别长尾场景
2. 上层逻辑输出动作决策
3. `car/control/vehicle_control/` 负责执行对应的车辆动作
4. `cloud/` 负责需要上云的推理与协同能力


## 性能优化建议

### 1. CPU环境
- 使用轻量级模型（已默认配置）
- 减少检测器数量
- 降低置信度阈值以加快推理

## 常见问题

### Q: 如何添加新的检测器？

1. 在 `detectors/` 目录创建新文件
2. 继承 `BaseDetector` 类
3. 实现 `detect(image_path) -> float` 方法
4. 在 `detectors/__init__.py` 中导入
5. 在 `classifier.py` 的 `detector_map` 中注册
6. 在 `config.yaml` 中配置

### Q: 如何调整权重？

修改 `config.yaml` 中的权重值，系统会自动归一化。建议：
- 语义检测器（CLIP）权重较高
- 特定目标检测器（YOLO-World）次之
- 通用检测器权重较低

### Q: 为什么FPS很低？

这是正常的，因为系统组合了多个深度学习模型。优化方法：
- 使用GPU加速
- 减少检测器数量
- 使用更轻量的模型
- 对视频进行跳帧处理

## 开发路线图

- [ ] 添加GPU加速支持
- [ ] 实现异步检测提高吞吐量
- [ ] 添加更多检测器（语义分割、深度估计等）
- [ ] 支持视频流实时处理
- [ ] 添加Web界面
- [ ] 模型量化和加速

## 许可证

本项目仅供研究使用。

## 联系方式

如有问题，请提交 Issue。
