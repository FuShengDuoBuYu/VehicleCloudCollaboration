# control 模块说明

本模块负责底层车控能力与 DonkeyCar 运行栈，提供车辆驱动、训练、校准与硬件接口代码。

## 模块职责

- 提供 DonkeyCar 主体框架与运行脚本
- 提供 mycar 工程入口（drive/train/calibrate）
- 提供底层电机控制与可视化辅助工具

## 目录结构

- donkeycar/：DonkeyCar 框架源码
- mycar/：车辆工程目录（manage.py, train.py, calibrate.py, 配置）
- utils/：底层车控与辅助工具

## 运行入口

主要入口在 mycar/manage.py。

支持命令：

- drive：启动驾驶流程
- train：训练模型

示例：

cd mycar
python manage.py drive --myconfig=myconfig.py

训练示例：

cd mycar
python manage.py train --tubs=data --model=models/mypilot.h5

校准入口：

cd mycar
python calibrate.py

## 逻辑关系

1. mycar/manage.py 负责构建车辆运行流水线
2. donkeycar/ 提供各类 parts（相机、控制器、执行器、记录、训练）
3. utils/ 提供底层轮控函数，供 longtail 模块或自定义脚本调用

## 与 longtail 模块关系

longtail/main.py 在摄像头模式下会导入 control/utils 里的：

- McLumk_Wheel_Sports.py
- visualizer.py

因此 control/utils 目录需保持不变或同步更新导入路径。

## 依赖与环境

- Python 版本与 DonkeyCar 版本需匹配
- 常见依赖包含 docopt、opencv、tensorflow/torch（取决于你的训练与推理路径）
- 若提示 ModuleNotFoundError: docopt，请先安装：

python -m pip install docopt

## 常见问题

- manage.py 无法导入 donkeycar：确认当前目录与 PYTHONPATH
- 硬件无响应：检查串口、PWM、驱动板连接与权限
- 训练命令失败：检查 tubs 路径、模型输出目录与依赖版本
