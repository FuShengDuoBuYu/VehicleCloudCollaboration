# longtail 模块说明

本模块负责长尾场景检测与评估，包含单图检测、目录批处理、摄像头实时检测三种运行方式。

## 模块职责

- 基于多检测器融合输出长尾风险分数
- 在摄像头模式下周期性取帧并判断是否触发停车
- 提供离线评估与性能基准脚本

## 主要文件

- main.py：统一入口（单图 / 目录 / 摄像头）
- classifier.py：多检测器融合分类器
- config.yaml：阈值与检测器参数配置
- benchmark.py：性能基准测试
- evaluate.py：数据集评估
- detectors/：具体检测器实现
- dataset/：样例数据
- requirements.txt：本模块依赖

## 逻辑流程

1. 读取 config.yaml
2. 初始化 LongTailClassifier（加载 detectors）
3. 按输入模式运行：
- 单图：调用 classifier.predict
- 目录：循环图片并统计汇总
- 摄像头：定时保存 latest_frame.jpg 并推理
4. 当摄像头模式判定 is_long_tail=True 时调用停车；否则继续前进

## 运行方式

在本目录执行。

安装依赖：

python -m pip install -r requirements.txt

单图检测：

python main.py --image path/to/image.jpg --verbose

目录批处理：

python main.py --directory path/to/images --output results.csv

摄像头模式：

python main.py --camera-index 0 --camera-interval 2 --camera-width 640 --camera-height 480 --speed 5 --visualize

## 关键配置

- threshold：长尾阈值
- detectors：检测器类型、权重、模型路径
- camera-* 参数：可由命令行覆盖

## 与 control 模块关系

main.py 会通过相对路径加载 ../control/utils 下的底层车控与可视化工具。

- McLumk_Wheel_Sports.py：前进/停车控制
- visualizer.py：实时监控面板

## 常见问题

- 缺少 yaml：安装 pyyaml
- 缺少 cv2：安装 opencv-python
- 模型加载失败：检查 config.yaml 中模型路径
- 摄像头打不开：检查 camera-index 与设备权限
