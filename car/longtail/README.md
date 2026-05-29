# longtail

`car/longtail/` 是车端长尾场景检测模块。闭环入口 `../run_closed_loop.py` 会从环境变量生成检测器配置。

## 主要文件

- `classifier.py`：多检测器融合分类器，提供 `LongTailClassifier`
- `env_config.py`：从环境变量构造检测器配置
- `../../.env_example`：环境变量示例
- `detectors/`：检测器实现
- `benchmark.py`：单张图片多轮推理性能测试
- `evaluate.py`：数据集评估脚本
- `dataset/`：样例数据集

依赖统一维护在仓库根目录的 `requirements.txt`。

## 分类器接口

```python
from classifier import LongTailClassifier
from env_config import load_longtail_config_from_env

config = load_longtail_config_from_env()
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

## 环境变量配置

核心环境变量：

- `CAR_LONGTAIL_THRESHOLD`：长尾触发阈值
- `CAR_LONGTAIL_DETECTORS`：检测器列表，例如 `clip,yolov8,yolopv2`
- `CAR_LONGTAIL_*_WEIGHT`：检测器融合权重
- `CAR_LONGTAIL_*_MODEL` / `CAR_LONGTAIL_*_WEIGHTS`：本地模型路径
- `CAR_LONGTAIL_*_REMOTE_MODEL` / `CAR_LONGTAIL_*_REMOTE_URL`：本地模型缺失时的下载来源
- `CAR_LONGTAIL_YOLOV8_IMG_SIZE` / `CAR_LONGTAIL_YOLOPV2_IMG_SIZE`：推理输入尺寸，树莓派优先使用 320
- `CAR_LONGTAIL_YOLOPV2_FAST_MASK`：跳过 YOLOPv2 大尺寸 mask 上采样，降低单帧延迟
- `CAR_LONGTAIL_TORCH_NUM_THREADS` / `CAR_LONGTAIL_OPENCV_NUM_THREADS`：运行线程数，`0` 表示使用框架默认值

完整示例见仓库根目录 `.env_example`。运行时优先加载根目录 `.env`；如果 `.env` 不存在，则加载 `.env_example`。如果本地模型路径不存在且 `CAR_LONGTAIL_AUTO_DOWNLOAD=true`，程序会自动使用对应远端模型下载或让底层模型库拉取。

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

1. 从环境变量构造检测器配置
2. 初始化 `LongTailClassifier`
3. 周期性保存摄像头帧
4. 调用 `classifier.predict(frame_path)`
5. 根据 `is_long_tail` 决定是否请求云端 LLM 决策

树莓派运行时可以用 `--interval 3` 控制每 3 秒检测一次。`run_closed_loop.py` 默认会先执行一次 synthetic warmup，让三个模型的首次推理开销发生在闭环开始前；如需关闭可传 `--no-warmup-detector`。

## TODO

- YOLOPv2 目前使用单帧可行驶区域和车道线 mask 的几何特征评分。后续增加前后帧时序状态，基于相邻帧的可行驶区域面积、底部中心区域、道路宽度、道路中心偏移和车道线密度突变来提高长尾判断稳定性。
