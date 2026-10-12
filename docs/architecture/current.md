# 当前实现与扩展边界

本页是当前源码架构的唯一概述；历史设计见[历史索引](../history/README.md)。基线为 6647934，2026-10-12 结构重构保持当前 visual_feedback profile 参数与控制算法，未改变已知实车结果。

## 操作入口

`vehicle` → `tools/vehicle_cli.py` 统一命令路由、环境和默认profile；需要独占设备的命令通过 `tools/process_lifecycle.py` 释放/恢复监控并等待子进程停车。面板、采集器及驾驶仍是独立进程。三个旧脚本仅转发已部署路径，服务模板已指向新入口，尚未重新部署。协作规则唯一入口为根AGENTS，两项技能正文唯一位于 `.agents/skills/`。

## 数据流与模块

1. `runtime/sources.py` 读取相机或归档视频，保留捕获时间与序号。
2. `perception/yolopv2_torch.py` 提供共用推理，`yolopv2_fusion.py` 管理异步推理；语义/颜色/路径模块组合当前帧证据。
3. `runtime/perception_pipeline.py` 计算本地提案；`control/lane_centering.py` 定义 LCC，`runtime/corner_runtime.py` 组合视觉短步与本地门禁。
4. `control/cloud_recovery.py` 构造候选并确认持续停车；`recovery_worker.py` 管理一个在途请求、响应与云证据；网络和合同位于 `car/cloud_client/`。
5. `control/decision_fusion.py` 的 `FusionPolicy` / `DecisionFusionAdapter` 统一处理本地、云辅助、hold。当前为规则策略，不是学习融合。
6. `control/wheel_driver.py` 映射有界输出；`motion_gate.py`、`watchdog.py`、`motion_guard.py` 与运行时共同约束时效、许可和停止。硬件后端位于 `car/control/vehicle_control/platforms/`。
7. `runtime/archive.py`、`sensor_recording.py` 与云 journal 记录原始证据；`runtime/reporting.py` 维护日志字段与状态结构；`publication.py` 只发布诊断。
8. `runtime/onboard.py` 编排初始化、控制循环与资源清理；`components.py` 负责装配、`configuration.py` 负责参数验证、`paths.py` 提供源码位置；`provenance.py` 在新运行初始化时冻结源码与版本，不在驾驶期间扫描仓库。

本地正常控制不等待云。云支路是“持续停车 → 当前原图/候选 → 异步语义建议 → 更新观测复核 → 单步 → 再观察”。仅观察请求不能授权运动。云等待不污染用于判断本地恢复的独立证据。

## 云端边界

本仓库实现车端客户端、异步 worker、合同校验和本地融合。当前云端是外部托管的 Qwen Realtime 多模态服务：接收受限候选与稀疏图像，返回候选建议/不确定性及理由；仓库没有自建云推理集群或训练服务。网络经过客户端的超时与协议校验，原始模型回答不直接送到底盘。论文图应把外部服务画在云端边界内，把复核、动作许可、停止与执行画在车端。完整绘图内容见[架构交接](../../coordination/2026-10-12-current-architecture-drawing-brief.md)。

## 稳定接口

| 边界 | 合同 |
|---|---|
| 感知与决策 | 匹配原图/掩码/检测、sequence、captured_at；不得把不同帧混用 |
| 本地提案 | `DifferentialDriveCommand` 与本地 deadline |
| 云输入/输出 | `road-recovery-v1` 与本地 event/candidate/frame 封装 |
| 融合 | `FusionInput`、`CloudAdvice`、`FusionDecision`；source 为 local/cloud_recovery/hold |
| 执行 | 本地许可有截止；最终拒绝必须记录为实际零输出 |
| 记录 | 原 CSV/JSONL 字段保持；新增 `provenance.json` + `source.zip` |

旧 `drive_runtime.py` 和 `onboard.py` 的公共符号保留兼容导出，供旧工具和回放使用。新代码直接导入职责模块。旧 `longtail.detectors` 提供同一 detector 类的兼容导入，当前感知不再反向依赖历史分类包。

## 扩展位置

- 新车型：增加 profile 和后端，通过工厂注入；不得在感知模块直接创建驱动。
- 新感知实现：遵守匹配帧/时间语义，支持回放注入；不得改变旧掩码或伪造捕获时钟。
- 新云模型：实现 provider/contract；先做本机假服务与归档输入评测，再安排真实云测试。
- 新融合策略：实现 `FusionPolicy` 并复用适配器校验和最终门禁，不能自设 PWM 或延长旧许可。
- 新分析：读取实验注册表，输出到新派生目录；注明输入哈希、代码、参数和证据等级。

## 当前边界

雷达/深度用于原始记录、展示和记录健康门禁，尚未输入局部几何避障或 SLAM；IMU 参与短转角度/时效约束，编码器保留用于复盘。旧 Pi 配置、历史弯道续行和关闭的 cloud_arbitration 不属于当前 visual_feedback 主链，但仍由兼容回归保护。

后续独立环境语义、yaw 漂移补偿、连续回中、越界锁定曾被撤回；不可把其历史文档当当前实现。当前仍保留第三弯停车、出弯偏左等已知行为。结构重构的验证是行为等价与证据完整，不宣称完成实车性能改进。
