# 车端项目入口

更新：2026-10-12（Asia/Shanghai）。论文题目：**基于云–端协同的大模型增强智能驾驶长尾场景感知与决策方法研究**。

## 当前事实

- 平台为 Jetson / Rosmaster；源码基线 `664793438d9d2cf51dae603e8d6fcc45902f6fb1`。本轮重构尚未提交，用户要求先 review；不是部署/实车验收。
- 面板入口选择 `rosmaster_jetson_yolopv2_visual_feedback`：异步 YOLOPv2 + 局部视觉控制 + cloud_recovery，旧 cloud_arbitration 关闭。
- 云合同为 road-recovery-v1：云选受限候选，车端规则融合、当前帧复核和最后门禁决定执行，云不设PWM。
- 当前控制版本保留09bf运行行为。后续独立环境语义、回中和漂移补偿等修复曾撤回；不能把历史记录作为当前集成事实。
- 雷达/深度用于记录/展示与记录健康门禁，未集成SLAM/几何避障；时序掩码突变触发不是当前主链。
- 第三弯停车、出弯偏左及IMU可信度等问题仍须独立算法/现场任务核验。结构重构不等于修复这些实车结果。

2026-10-12上一阶段工程核验（当时的检查入口）：主目录 `bash check.sh --torch --loopback` 576项通过；冻结05a3运行的867帧控制结果在归档计时下重构前后等价。原数据保留，19条历史迁移通过哈希核对；详细命令、run ID和限制见交付记录。WebSocket假服务依赖未安装，该可选组未执行。

2026-10-12入口收敛：统一使用 `./vehicle`；根AGENTS是唯一协作约定，两项技能实体统一至 `.agents/skills/`。三份旧脚本暂为已部署/已加载进程保留兼容转发，源码服务模板已迁移但未重启服务。主目录最终593项回归通过；首次云恢复合成回放出现间歇性断言失败，三次单测与全量重跑未复现，根因待查。验证记录见[入口收敛报告](coordination/2026-10-12-entrypoint-consolidation.md)。

## 权威入口

| 内容 | 位置 |
|---|---|
| 当前模块与扩展合同 | [系统架构](docs/architecture/current.md) |
| 启动/设备/服务副作用 | [运行说明](docs/operations/runtime.md) |
| 无硬件回归 | [开发指南](docs/operations/development.md) |
| 原始与派生数据目录 | [实验注册](experiments/README.md) |
| 历次事实完整原文 | [历史项目索引](docs/history/project-index-through-2026-10-12.md) |
| 全仓审查依据 | [审查报告](coordination/2026-10-12-repository-review.md) |
| 绘图交接 | [架构内容文档](coordination/2026-10-12-current-architecture-drawing-brief.md) |
| 本轮重构与验证 | [交付记录](coordination/2026-10-12-repository-refactor.md) |

## 证据与保全

E1：源码/配置/run/原始数据/分析方法可共同追溯；E2：代码或开发验证；E3：历史实现/演示；E4：计划或缺原始依据的描述；X：已证伪/失效。目录存在不自动达到E1，回放不产生新实车轨迹。

原始outputs、数据集、权重、环境、失败记录和源码快照保留原位，未清理。初次审查有26个面板session、22个run目录和21组对应传感器记录；注册表还列出其它历史runs。独立异机备份尚未核验。新run初始化增加版本manifest与源码ZIP，旧run不回填虚构版本。

完整论文、开题/中期报告及正式实验计划主要在电脑 `C:\Users\fushe\Desktop\thesis`。车端不假定它们全部同步。规则、硬件授权和跨聊天发送遵循根AGENTS.md。

## 下一步与负责人

本轮由用户review未提交差异；车端按review修订，再由用户决定提交和现场验证安排。论文分析角色通过实验注册表关联证据并更新正文/图注。没有自动向其他聊天发送任务。
