# YOLOPv2-LCC 离线自动驾驶

这一模块把 YOLOPv2 的可行驶区域分割结果转换为可测试的连续控制量，不依赖摄像头、
电机或 Raspbot 驱动。

处理链路：

```text
图片/视频 -> YOLOPv2 masks -> 道路连通域 -> 左右边界与中心线
          -> 横向误差/航向误差 -> LCC -> 左右轮归一化速度
```

## 离线回放

在仓库根目录运行：

```bash
.conda/bin/python car/autodrive/offline_replay.py \
  954c623c0ad6774905270ecca99aecbc.mp4 \
  d95a49f05b4fa169a864f74de6693665.mp4 \
  --sample-every 6 \
  --camera-view external \
  --save-samples
```

仓库根目录的两段 MP4 是场地外部视角，必须使用 `--camera-view external`。外部视角下
画面中心不等于车体中心，不能从中产生有物理意义的 LCC 指令。将来采集的车载视角录像
使用 `--camera-view onboard`，即可离线回放中心线和控制量。

不同的车载相机高度、俯仰角和视场角不能共用同一组图像坐标控制增益。复制
`onboard_calibration.example.yaml`，在真实车载帧上填写道路梯形的四个归一化点并把
`calibrated` 改为 `true`，然后增加：

```bash
--camera-view onboard --calibration car/autodrive/onboard_calibration.yaml
```

程序会先把 YOLOPv2 mask 转为鸟瞰坐标计算误差，再把中心线投回原图用于检查。

## 树莓派车载入口

车载入口默认是干运行，只保存控制建议，不实例化 I2C 底盘：

```bash
python car/autodrive/run_onboard.py \
  --config car/autodrive/onboard_runtime.example.yaml \
  --max-samples 20
```

主要辅助工具：

- `pi_self_check.py`：只读检查 Python、依赖、模型、相机、标定和 I2C
- `capture_onboard.py`：采集固定车载相机的标定图和视频
- `calibrate_perspective.py`：点击四点生成鸟瞰标定和预览
- `check_wheel_directions.py`：架空车轮后的低 PWM 方向检查
- `run_onboard.py`：干运行或显式确认后的真实底盘闭环

`run_onboard.py --temporal` 会在 YOLOPv2 关键帧之间用稠密光流传播两个 Mask，提高
控制更新频率。该功能已经通过合成平移测试，但默认关闭，必须先在真实车载视频上检查
漂移和置信度衰减。

完整操作顺序见 [RASPBERRY_PI_HANDOFF.md](RASPBERRY_PI_HANDOFF.md)。

默认结果写入 `outputs/autodrive_offline/`：

- `*_lcc.mp4`：分割、道路中心线、预瞄点和控制量叠加视频
- `*_lcc.csv`：逐帧误差、置信度、转向量和左右轮建议速度
- `*_perception_frames/`：外部视角视频中实际送入模型的逐帧识别图
- `*_perception_contact_sheet.jpg`：前、中、后时刻抽帧识别结果汇总图
- `summary.json`：每段视频的有效率、耗时和动作分布

控制输出是 `[-1, 1]` 的归一化建议量；CSV 中的 PWM 只是按 100 为上限换算的
待标定建议值，离线运行不会访问车辆硬件。

## 当前边界

- 分叉路口可以用 `--route-hint left|center|right` 指定优先分支，但正式上车前仍需
  结合路线状态机验证。
- PWM 正负方向、相机单应性、车体左右轮差异和控制增益必须在实车上标定。
- YOLOPv2 的可行驶区域不等同于障碍物自由空间；锥桶等障碍仍由长尾检测与安全监督层处理。
