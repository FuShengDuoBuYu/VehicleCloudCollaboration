# 外圈 LCC

此包只维护当前树莓派实车使用的外圈车道居中控制。它直接处理车载相机中的黄色道路
边界和绿色岛区，不依赖 DonkeyCar、云端变道或旧网页。

当前实车配置还会在独立后台线程运行 YOLOPv2，并把其可行驶区域与黄色边界 LCC 的
最终走廊做保守融合。YOLOPv2 只能验证或缩小当前走廊，不能把走廊扩大到绿色岛区；
结果过旧、缺失或与当前走廊重叠不足时，默认回退到传统 LCC。这样同一份原始语义 mask
可继续供长尾突变检测使用，而后台语义推理不会阻塞控制循环。LCC 专用配置会冻结并融合
TorchScript 推理图，只计算实际使用的可行驶区域分支；权重和 320 输入分辨率均不改变。

## 目录结构

```text
autodrive/
├── camera/       # 云台姿态与图像方向变换
├── config/       # 当前实车配置、透视标定及示例
├── control/      # 中心线控制器、PWM 映射和安全门
├── perception/   # 外圈边界、透视映射和可视化
├── runtime/      # 相机到四轮输出的实时闭环
├── tools/        # 标定、采集、自检和架空轮测试
├── web/          # LCC 网页服务与前端
├── run_onboard.py
└── run_lcc_web.py
```

根目录的两个 Python 文件只是稳定命令行入口，具体实现放在对应子包中。

## 日常启动

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
./run.sh
```

浏览器访问 `http://<车辆IP>:8080`，勾选安全确认后点击“启动 LCC”。网页启动后才占用
相机；停止/急停、服务退出、运行时间到期或子进程异常都会执行四轮归零。

网页的实时链路面板直接展示黄色边界 LCC、YOLOPv2 当前/目标精度、融合或安全回退、
运动门控与四轮执行状态。下方详情同时展示边界观测行数、融合重叠率、语义结果年龄、
FP32/INT8 完成计数、控制循环频率、看门狗和后台队列；这些值均来自当前 `status.json`，
并显示状态文件与画面的更新时间。页面底部可展开查看网页实际收到的完整状态 JSON。

## 命令行与离线回放

直接实车运行：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE \
  --max-runtime-seconds 60
```

某轮出现问题后，使用归档原始视频逐帧干运行，不会驱动车轮或覆盖实车记录：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --video outputs/onboard_runtime/runs/<当次目录>/raw.mp4 \
  --sample-every 1 \
  --output-dir /tmp/lcc_replay \
  --no-run-archive
```

融合配置位于 `perception.yolopv2`。当前调试图中蓝色轮廓是原始 YOLOPv2 可行驶区域，
绿色是控制器实际使用的融合走廊。CSV 中的 `semantic_fusion_source` 表示当帧状态：

- `fused-intersection`：YOLOPv2 与当前 LCC 走廊正在做保守交集。
- `fallback-low-overlap`：两者重叠不足，继续使用当前 LCC。
- `fallback-stale`：YOLOPv2 结果超过最大年龄，继续使用当前 LCC。

`required_for_motion` 默认保持 `false`。只有在多段正常实车视频均验证稳定后才可改为
`true`；严格模式下，YOLOPv2 缺失、过期或低重叠都会请求停车。

`optimize_for_inference: true` 与 `drivable_only: true` 是无需微调的车端加速项。后者会跳过
LCC 当前不用的目标检测和车道线输出；如果其他调用方需要这两类输出，应在该调用方的
YOLOPv2 配置中保持 `drivable_only: false`。

车端还支持 `backend: onnxruntime` 的 FP32 可行驶区域专用模型。当前树莓派 5 建议配置
`onnx_intra_op_threads: 1`，给 LCC 和系统保留最大 CPU 余量。纯 INT8 模式只用于离线消融；
实测严格直道上 INT8 平均 IoU 为 `0.9975`、约加速 `3.07x`，但个别连续转弯帧会产生
假可行驶区域。因此 `adaptive_precision` 只在黄色边界 LCC 连续确认稳定直道后使用
INT8；弯道、弱边界或较大转向立即请求 FP32，切换期间未完成的 INT8 结果不参与融合。
完整实验条件和结果见 `YOLOPV2_ONNX_EXPERIMENT.md`。

## 标定和诊断工具

- `tools/pi_self_check.py`：只读检查依赖、配置、相机和 I2C。
- `tools/align_camera_gimbal.py`：安全确认后调整云台。
- `tools/capture_onboard.py`：按运行配置采集标定图和视频。
- `tools/calibrate_perspective.py`：生成四点鸟瞰标定。
- `tools/check_wheel_directions.py`：车轮架空后的单轮方向检查。

例如重新采集和标定：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/tools/capture_onboard.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --confirm-camera-gimbal-clear CAMERA_GIMBAL_IS_CLEAR \
  --seconds 10

/home/pi/miniconda3/envs/car/bin/python car/autodrive/tools/calibrate_perspective.py \
  outputs/onboard_capture/onboard_calibration_frame.jpg \
  --runtime-config car/autodrive/config/onboard_runtime.yaml \
  --output car/autodrive/config/onboard_calibration.yaml \
  --force
```

相机机械位置、S1 角度、分辨率或图像旋转改变后必须重新标定。当前水平云台正前方为
`25°`。当前标定下最新直道双边界拟合宽度为 `0.554–0.577`，单边界回退宽度配置为
`0.70`。双边界可用时控制器始终采用两条真实曲线的中点，不用固定宽度覆盖右边界。

## 输出和安全

每次网页或命令行实车运行都会在 `outputs/onboard_runtime/runs/` 保存原始、标注、鸟瞰
视频、逐帧 CSV、最终状态、配置与标定快照。`outputs/onboard_runtime/latest.jpg`、
`latest_birdeye.jpg`和`status.json`供网页实时显示。

录像、CSV 和网页快照均由后台线程写入，不会阻塞电机控制；存储持续跟不上时会优先保留
最新诊断帧，并在 `status.json` 的 `diagnostics` 和 CSV 的队列/丢帧列中记录。归档目录内
的 CSV 只包含实际写入视频的帧，因此时间戳始终与 MP4 一一对应。CSV 还记录采帧间隔、
感知、控制门、硬件下发和鸟瞰渲染耗时，用于直接区分相机、算法、I2C 与诊断 I/O 卡顿。
`boundary_left_rows` 和 `boundary_right_rows` 分别记录每帧实际拟合到的左右黄线行数；
`outer+width` 表示右侧内边界当前不可用、系统正用左外边界与固定宽度恢复中心，并不表示
原始相机画面中一定完全没有右侧黄线。标注图和鸟瞰图中，青色实线是真实拟合边界，
品红色虚线是固定宽度推算边界。

已经进入有界急转后，最短急转期内新出现的一帧 `both` 只作为候选恢复，不会因横向误差
立即取消急转。最短期结束后，真实双边界恢复与近场航向对齐分别连续确认两帧；两类证据
不会混合计数。车底黄线仍在任何阶段立即停车。CSV 的
`corner_apex_both_valid_count` 记录当前连续双边界确认帧数。

当前底盘映射为：

- `0=左前、1=左后、2=右前、3=右后`
- 直行 `16/16/20/20`
- 普通最大右弧 `26/26/10/10`
- 饱和急弯 `30/30/0/0`，任何车轮都不反转

单边界恢复最多 `8s`；弯顶后近场航向连续对齐且至少一条新鲜边界可用时，立即交还普通
LCC，不再等待双边界。连续 `1s`没有进展、黄线进入车底、边界真正消失、相机断流或
看门狗超时都会停车。停车后需要至少 4 帧来源一致、误差连续的有效感知才会重新起步。
