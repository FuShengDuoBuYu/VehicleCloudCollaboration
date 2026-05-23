# longtail

`car/longtail/` 是车端长尾场景检测模块。闭环入口 `../run_closed_loop.py` 会加载这里的分类器与配置文件。

## 主要文件

- `classifier.py`：多检测器融合分类器，提供 `LongTailClassifier`
- `config.yaml`：检测器类型、权重、阈值与模型参数
- `detectors/`：检测器实现
- `benchmark.py`：单张图片多轮推理性能测试
- `evaluate.py`：数据集评估脚本
- `dataset/`：样例数据集
- `requirements.txt`：依赖列表

## 分类器接口

```python
import yaml
from classifier import LongTailClassifier

with open("config.yaml", "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

classifier = LongTailClassifier(config)
result = classifier.predict("path/to/image.jpg")
print(result["is_long_tail"], result["score"])
```

`predict(image_path)` 返回：

- `is_long_tail`：是否达到长尾阈值
- `score`：融合后的长尾分数
- `threshold`：当前阈值
- `individual_scores`：各检测器输出
- `inference_time`：总推理耗时
- `fps`：按本次耗时估算的 FPS

## 配置

`config.yaml` 中的核心字段：

- `threshold`：长尾触发阈值
- `detectors`：检测器列表
- `type`：检测器类型，例如 `clip`、`yoloworld`、`yolov8`、`yolopv2`
- `weight`：融合权重
- `config`：单个检测器的模型路径和参数

## 性能测试

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration/car/longtail
python benchmark.py --image dataset/Long-tail/000001_1616005254200.jpg --runs 5
```

## 数据集评估

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration/car/longtail
python evaluate.py --dataset dataset --output evaluation_results
```

评估脚本会根据目录名中的关键词识别正负样本，例如：

- 正样本关键词：`long-tail`、`corner-case`、`accident`、`construction`、`positive`
- 负样本关键词：`non-long-tail`、`normal`、`negative`、`clear`

## 闭环调用

`../run_closed_loop.py` 会将 `car/longtail` 加入 Python 路径，并按以下方式使用：

1. 读取 `config.yaml`
2. 初始化 `LongTailClassifier`
3. 周期性保存摄像头帧
4. 调用 `classifier.predict(frame_path)`
5. 根据 `is_long_tail` 决定是否请求云端 mock 决策
