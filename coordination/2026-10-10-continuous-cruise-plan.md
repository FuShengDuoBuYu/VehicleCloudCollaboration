# 实时直道巡航与弯道短步修复计划

用户已明确接受本聊天提出的“直道持续有效时连续低速修正，弯道短步转动或前拱”，并报告本轮结束。采用本聊天直接实现，保留已有工作，不提交、不主动启动运动。沿用 vehicle-cloud-thesis、systematic-debugging、brainstorming 和 TDD 流程；按用户要求只做针对性检查。

**Goal:** 取消直道固定停走，使每个新的有效 YOLO 结果更新轻微差速；弯前和弯中仍按当前近场道路决定短步。

**Architecture:** `VisualFeedbackController` 增加当前帧支持的巡航许可，只有新的语义序号能续期，独立驱动截止时间始终存在；缓存、过期、撤销和取消不能延长运动。`StationaryCornerRuntime` 用近场方向、横向偏差、当前前边界位置及原控制器轻微转向判定巡航；退出巡航先停再重新观察。旧 profile 与旧归档配置保持短步默认。

**Tech Stack:** Python/OpenCV/NumPy，原有只读录制输入回放，不连接相机、模型、UART或底盘。

**Evidence:** 最新 session `a3dde8b5ae9a44cea9681aca2d0f3f5a` 已结束并停止确认；最后150帧74次当前证据否决、34次应用前过期，仍有18次运动建议。保存到 `outputs/visual_feedback/20261010/continuous-cruise-fix/`。四个真实输入 CPU profile 显示绘图每帧约43ms、曲线路径约38ms；先减去控制前的绘图及等价的数组分配开销，不放宽0.3s语义/0.25s相机时效。

## Task 1: 巡航许可与当前方向修正

- [x] 在 `car/test/test_visual_feedback.py` 先复现连续新观察仍被0.15s强制停车；覆盖缓存不续期、失效停车、巡航转短步停车、轻微左右修正保留及前边界接近切短步。
- [x] 修改 `control/visual_feedback.py` 的 update 增加 `continuous=False`；巡航只有新序号续期，截止时间不超过当前消费帧采集时间+0.3s及本次授权时间+0.25s。
- [x] 修改 `control/stationary_corner.py` 增加默认false的 `continuous_cruise`；`runtime/onboard.py` 按当前近方向/横向误差和可见前边界判定，保留原建议的轻微差速。
- [x] 仅实车visual profile启用连续巡航、短步停顿设0.25s；最高PWM30、每个转向短步最多0.15s/5°不变。

## Task 2: 缩短从结果到执行的延迟

- [x] 在控制前绘图延后用例和真实输入几何基准上验证原始问题。
- [x] `runtime/onboard.py` analyze 增加可选 deferred renderer，默认调用者行为不变；实车先检查并执行命令再画同一消费帧。
- [x] `control/lane_centering.py` 用预先构建的索引矩阵代替每行多个临时数组；所有原掩码排除、逐像素过渡检查与代价保持等价。
- [x] 核对少量实际录制输入与保存的旧源码，保持路径和估计值相同。

## Task 3: 定向回放和交付

- [x] 运行新增和相关控制测试，禁止全项目设备测试；回放本轮原始输入的直道、弯前、末尾窗口，记录保存时序与候选许可，区别历史时序重放与新的处理耗时测量。
- [x] 保存前后源码/增量patch/hash、输入hash、分析结果，更新 PROJECT_INDEX 和本轮报告。
- [x] 复核独立截止时间、取消锁存、缓存不得续期以及弯道状态切换；下一次用户点击启动由新进程加载，不发送启动请求。

## Review focus

缓存序号不能续期；已过截止时间的指令不能恢复；巡航与弯道切换先停车；当前帧未知/未来/回退不能运动；IMU缺失不能pivot。针对性现有/新增CPU测试覆盖这些条件。回放不能证明候选新轨迹、真实停稳或完整过弯。
