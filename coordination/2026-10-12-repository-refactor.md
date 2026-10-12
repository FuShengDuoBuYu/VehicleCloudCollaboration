# 仓库重构交付记录

2026-10-12。源码基线：`664793438d9d2cf51dae603e8d6fcc45902f6fb1`。按用户要求 **不 commit、push、merge**；保留未提交差异供 review。隔离工作树为 `/tmp/vcc-repository-refactor`，验证后按文件前像哈希同步到 `/home/jetson/VehicleCloudCollaboration`，已有未提交资料保留。此记录是工程重构交付，不是部署或实车验收。

## 已完成的结构调整

| 范围 | 结果 |
|---|---|
| 运行编排 | onboard 从 2796 行拆至 934 行；采集、归档、发布、配置、装配、感知流水线、局部控制集成、日志/状态分别归属独立模块 |
| 控制层 | 轮速驱动、运动门禁、看门狗、历史续行分别独立；drive_runtime 留 13 行兼容入口，保留旧导入与回放 |
| 感知层 | 共用 YOLO detector 移入 autodrive/perception；longtail 历史路径兼容，当前运行不再依赖研究包实现 |
| 证据 | experiments 提供 session/run/传感器/云事件注册及数据集内容与标注映射；新运行保存源码ZIP、commit/dirty、源码/配置/primary和INT8模型哈希及允许的环境版本 |
| 文档 | 当前架构、运行副作用、开发入口统一；长项目索引、阶段设计、失效闭环脚本/报告、旧依赖原文归档；19 条迁移记录包含内容哈希 |
| 依赖与检查 | 运行、研究、可选WebSocket测试依赖分开，保存Jetson已安装版本；check.sh 统一无硬件回归与静态检查 |

当前入口：[README](../README.md)、[系统架构](../docs/architecture/current.md)、[运行说明](../docs/operations/runtime.md)、[实验注册](../experiments/README.md)。交给绘图 agent 的文档为[架构内容说明](2026-10-12-current-architecture-drawing-brief.md)，已更新模块位置，不产生PDF。

## 修复的不一致

- 文档曾混用初始“未接入云”、单帧观察与当前多帧恢复，现已区分独立观察合同与驾驶恢复合同，以及当前实现和撤回方案。
- 当前面板支持会话启停，底盘说明中的旧“仅只读、无网页启停”描述已纠正；正常标定门禁与受限 field-trial 分开说明。
- 修复旧测试对基线行为的过时假设：斜前沿不算有效入弯观测但保留诊断；没有云journal的本地颜色fixture显式关闭云恢复；跨测试引用使用完整包路径。
- 离线 recorded 回放曾读取分析机器的耗时，CPU竞争导致 4/867 帧差异。现使用归档的精确许可或旧毫秒数；缺失/舍入临界值标未知并保持停车。live 感知计时、控制算法及profile阈值未改。
- 独立审查发现并修复自适应INT8模型身份遗漏，以及外盘symlink可绕过catalog输出保护。详见[审查记录](refactor-2026-10-12/review.md)。

## 数据保全

原始视频、无损NPZ、CSV、传感器、云记录、失败运行、源代码快照、模型、现用venv与数据集均未删除或搬移。19条历史迁移通过内容哈希核对；旧Markdown路径留短跳转。失效闭环代码与报告退出当前测试目录，但原文仍在 research/legacy。

注册表包含 **26 个面板session、74 个历史run路径、21 组关联传感器、1600 个数据集文件/800 份唯一内容**。未发现同内容不同二分类标签冲突。798组重复内容的路径/哈希与初次盘点一致；重复文件保留，source与label别名分别登记，避免物理去重造成来源丢失。

3组run ID在原目录与打包目录重复，注册表要求以完整路径区分，不把副本当独立实验。旧run缺失源码来源时保留未知，`integrity=not_verified` 不冒称原始证据完整。未对约54GB既有输出逐字节重算全部哈希，也未核验异机备份。

新增验证派生资料独立保存于 `outputs/refactor_validation/20261012/`，未覆盖原实验；[文件哈希清单](refactor-2026-10-12/validation-artifacts.json)记录前后回放、原主机计时差异及日志。

## 验证与实际限制

- 主目录 `bash check.sh --torch --loopback`：**576项通过，49.068秒，退出码0**，包含正常运动/超预算/慢主机回放回归。解释器为现用 `outputs/field_yolopv2/venv/bin/python`；完整输出见[最终测试日志](refactor-2026-10-12/final-main-tests.log)。隔离目录上一轮575项通过，之后补充1项运动回归。
- 静态检查：157个Python文件、9个shell入口；19份维护文档本地链接及19条历史迁移哈希通过；git diff --check通过。
- 冻结原run：`outputs/autodrive_sessions/05a3c25bc5784be48c9e22fd39d6073b/runtime/runs/20261011_062723_573638_hardware`。867帧完整控制结果相同，820份NPZ及原CSV、配置哈希一致，未知处理预算0行。基线用归档预算注入harness，新回放原生使用归档预算；只排除随机事件UUID和新增预算字段。动作、PWM、融合和门禁状态均参与比较。见[逐帧等价结果](refactor-2026-10-12/replay-equivalence.json)及同目录两份复核脚本。重跑基线需从6647934单独检出源码，不使用已重构主目录冒充基线。
- 新工程回归覆盖损坏元数据、跨目录symlink、保护目录禁写、数据标签别名、未提交源码ZIP、模型身份、旧预算未知/精度边界与慢主机条件。
- `test_cloud_scene_ws` 需要当前未安装的 websockets 假服务器依赖，明确未执行；其余云合同/Realtime协议假对象测试已包含。没有安装依赖、访问真实云或驱动硬件，没有重启服务。

本轮保留09bf控制行为及已知第三弯停车、出弯偏左、IMU可信度等现场问题。结果证明结构与离线行为，不证明整圈通过或真实车辆停止距离；这些仍需独立算法分析及当次现场验证。

## 交付状态

119项文件写入/归档删除已通过同步前像检查，集成后全部字节与隔离验证工作树一致；映射见[integration.json](refactor-2026-10-12/integration.json)。随后仅补齐本交付记录、项目索引及计划完成状态，新增最终验证记录。源码、数据和架构的最终文件身份见[final-state.json](refactor-2026-10-12/final-state.json)，源码来源快照见[source-manifest.json](refactor-2026-10-12/source-manifest.json)。

直接 `runtime/onboard.py --help` 和历史 `longtail/detectors` 导入兼容检查通过，未构造模型或设备。最终 HEAD 仍为6647934，暂存区为空。所有改动留在主工作区等待用户review；隔离工作树保留，无自动提交或服务重启。
