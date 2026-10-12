# 车端自动驾驶

当前总览见[架构说明](../../docs/architecture/current.md)，操作见[运行说明](../../docs/operations/runtime.md)，验证见[开发指南](../../docs/operations/development.md)。

当前 Jetson 主链：YOLOPv2 道路/车道/检测 + 同帧颜色/路径证据 + LCC/视觉短步 + 异步云恢复 + 车端最终门禁。活动 profile 为 `rosmaster_jetson_yolopv2_visual_feedback`，旧 Pi/ONNX 模式仅为兼容和历史实验。

runtime各模块分别负责 sources、perception_pipeline、corner_runtime、components、configuration、archive、publication、reporting 和 provenance。onboard只保留编排及历史公共入口兼容。control 的 drive_runtime 为导出门面，具体实现分到 wheel_driver、motion_gate、corner_continuation 和 watchdog。

原始实验数据保留原位，统一入口见[实验目录](../../experiments/README.md)。已知实车问题与撤回记录见[项目索引](../../PROJECT_INDEX.md)。
