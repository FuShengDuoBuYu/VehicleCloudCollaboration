# 仓库审查与保留证据的重构建议

日期：2026-10-12（Asia/Shanghai）。状态：审查与方案，尚未执行迁移、删除或产品重构。

## 目标、范围与基线

用户希望整理整个仓库，保留可能用于毕业论文的实验记录和中间产物，移除与论文无关或已失效的内容，并形成可与导师讨论、可用于论文的当前车云架构图。

- 仓库：`/home/jetson/VehicleCloudCollaboration`，分支 `main`，HEAD `664793438d9d2cf51dae603e8d6fcc45902f6fb1`。
- 审查开始时 `git status --short` 为空，共 1874 个跟踪文件。当前提交后的进程加载状态未经检查；本文说明当前磁盘代码及历史证据，不宣称重新完成部署验收。
- 覆盖：全仓目录/文件元数据、数据集内容哈希、启动脚本和服务定义、当前 profile、感知—控制—云恢复—驱动调用链、记录/回放接口、历史交接和主要设计。
- 不包括：逐帧重算所有实验、逐条审计全部日志、全源码正确性证明、全量原始数据哈希核验、独立备份验收、线上云服务核验、硬件运行或测试套件执行。
- 电脑论文正文、开题/中期报告原件及完整实验计划不在本仓库。这里只能核对车端证据，不能宣布所有文件已完成论文价值鉴定。

## 主要发现及影响

### 1. 当前入口说明与实现冲突，优先修正文档

`README.md:93–100` 仍把云端描述为 `qwen3.8-max` 且尚未接入实时链路。实际 `car/autodrive/runtime/onboard.py:1910–1923` 在恢复模式构建 `qwen-realtime` 客户端，指定 `qwen3.8-omni-flash-realtime`、`road-recovery-v1`，并建立 `RecoveryCoordinator`。

README 关于仅经典黄色边界 LCC、Jetson 仅相机干运行、网页启停已移除等描述也不是当前面板入口的完整说明。操作者入口实际通过 `drive_sessions.py:40` 使用 `rosmaster_jetson_yolopv2_visual_feedback`。应建立唯一的 `docs/architecture/current.md`，README 只保留入口和指向当前事实的链接。

### 2. 最新文件日期不等于当前生效方法

2026-10-11 恢复报告明确恢复 09bf 实车版本，撤回了随后新增的独立环境语义、持续回中、yaw 漂移补偿、全量云事件展示、越界锁定等修改。当前代码仍是候选选择合同，没有 `road-environment-v1`；不能将 f306/063829 后续修复画成当前模块。

当前保留旧版偏左、第三弯停车等已知问题。09bf“第二个弯通过、第三弯卡住”属于历史运行结论；本轮没有新实车成功结果。回退备份中的源码和 patch 是解释实验版本的必要证据，不能因“中间文件”删除。

### 3. 论文目标与当前触发机制需明确区分

活动 profile 的 `cloud_recovery.enabled=true`、`cloud_arbitration.enabled=false`。当前主链是持续停车确认、原图/上下文/候选上传、云语义选择、当前帧复核、有界短步和重新观察。

`cloud_arbitration.py` 确实存在掩码时序变化触发代码，但该分支在活动 profile 关闭，且 `cloud_configuration()` 明确禁止两模式同时开启。恢复模块的 mask-change 判断主要用于请求场景变化后的失效检查；不能据此声称论文“可行驶区域时序突变触发”已经成为当前主链并通过正式实验。旧分支应保留为可追溯研究基线，是否重新整合属于下一项算法决策，不夹带在目录整理中。

### 4. 大部分磁盘占用是可关联的研究证据

`du -h -d 1 .` 约 55 GB，其中 `outputs/` 约 54 GB；下面使用文件逻辑字节折算 GiB，和磁盘占用口径略有差别。

| 路径 | 逻辑容量约 GiB | 处理建议 |
|---|---:|---|
| `outputs/vehicle_dashboard/` | 24.088 | 保留。主要为传感器记录，21 组 recordings 全部能按 run ID 对应驾驶会话 |
| `outputs/autodrive_sessions/` | 19.421 | 保留。26 个 session 记录、22 个 run 目录，包含启动失败/短运行，不按成功与否过滤 |
| `outputs/corner_fix/` | 4.959 | 保留原始运行、源码快照、标定、分析脚本、派生对照；按实验与版本建索引 |
| `outputs/hardware_acceptance/` | 0.837 | 保留底盘/深度/雷达验收及失败尝试 |
| `outputs/diagnostics/` | 0.792 | 保留复盘、撤回实现、差异补丁与复算材料；区分原始和派生 |
| `outputs/outer_loop_trial/` | 1.017 | 保留实车试验和评估包 |
| `outputs/field_yolopv2/` | 0.755 | 混合运行环境、安装包与实验；不能整体移走或清理 |
| 其余 outputs 子目录 | 约 1.115 | 按 run 和来源补全分类；不能按大小或文件名判断价值 |

传感器与会话的对应是目录身份核对，不是每条原始日志完整性验收。尚未验证异机独立备份，Git 并不保护被忽略的 `outputs/`、权重和环境。

### 5. 数据重复保存了来源/标注关系，不等于无关数据

`car/longtail/dataset/` 有 1600 个文件；对全部文件做 SHA-256 后有 800 个不同内容、798 个重复组、800 份额外副本。主要是 `BDD100K/img`、`CODA/images` 与人工 `Long-tail`/`Non-long-tail` 目录之间复制。上述统计包含非图片文件，不能写成 800 个独立场景或论文样本数。人工两类之间未发现字节完全相同的跨标签冲突；没有做近重复/语义冲突检查。

建议保留一个内容原件，另建 `samples.csv`：内容哈希、所有原路径、来源数据集/原 ID、人工标签、标签版本、分组/划分、备注。每个路径的来源与标注关系分别保留，不因去重折叠标签。正式划分按来源场景/事件分组，避免同图或相邻帧跨集合。迁移前原目录保持原样；若既往评估脚本使用原目录结构，保留版本化样本清单和兼容读取入口。

### 6. 运行代码和历史研究目录存在真实依赖

`yolopv2_fusion.py:153` 直接导入 `longtail.detectors.yolopv2_detector.YOLOPv2Detector`。因此整个 `car/longtail/` 不能当旧架构删除。先抽出共用 YOLO 推理实现，再迁移单帧 CLIP/YOLOv8 分类实验。

`onboard.py` 有 2796 行，同时处理输入、归档、发布、感知、云协调、执行和清理；`drive_runtime.py` 有 1445 行。适合沿既有职责拆分，但不应同时改控制算法/阈值。`car/test/closed_loop_test.py:26` 导入已不存在的 `vehicle_control.controller`，`car/longtail/README.md` 还引用缺失的 `run_closed_loop.py`，属于应退出当前入口的历史链路。其历史报告仍保留。

### 7. 输出目录里包含当前运行环境，依赖说明也需重整

多个启动脚本直接使用 `outputs/field_yolopv2/venv/bin/python`，且配置中的标定证据指向 `outputs/corner_fix/.../summary.json`。移动目录会影响启动、回放、校准依据和服务挂载路径。

仓库 `requirements.txt` 要求 `torch>=2.1.0`；现有环境的安装元数据是 `2.0.0+nv23.5`。目前不能用通用 requirements 复建同一 Jetson 环境。应分别冻结 Jetson 运行、离线评测、开发验证依赖及系统库/镜像/权重版本。先记录，不自动升级或重建虚拟环境；虚拟环境也不能直接搬目录后假设可用。

## 保留、归档与删除规则

| 类别 | 具体对象 | 决策 |
|---|---|---|
| 不可替代证据 | 原视频、无损模型消费帧/掩码、原始传感器、UART、CSV、云原请求/回复、事件日志、标签、标定、配置/源码快照 | 原样保留，建立关联与校验；失败运行同样保留 |
| 可重算但有论文价值 | 回放、分析脚本、统计表、派生图、反事实对照、首次失败结果 | 保存输入/代码/参数/输出关系；是否保留大体积派生副本逐项决定 |
| 过时设计/计划 | 已完成 bringup 计划、旧 Pi/ONNX 架构、撤回的 f306/063829 设计 | 移出当前说明，转为历史记录并标注 superseded/reverted；实验解释所依赖部分不删除 |
| 历史实现 | 旧左右变道 mock 客户端、失效闭环测试、单帧分类实验 | 先查当前调用和历史复现依赖，转入 legacy/research；已有兼容测试也同步归属 |
| 可清理候选 | `__pycache__`、确认可再生的缓存、已有可信副本的下载包、无引用且无研究信息的重复说明 | 形成准确路径和理由清单后处理；下载包可能是 Jetson 复建所需，不能先删 |
| 不确定用途 | 无来源图像、未关联的日志、无法确认是否正文引用的材料 | 标记待核验并保留，不以“看起来无关”删除 |

本轮没有发现足以直接认定“整批无关、可立即删除”的实验数据目录。过时的架构可从活动文档移除；其历史实验依据仍需要保留。Git 历史不替代忽略文件的备份，不重写 Git 历史来缩小体积。

## 推荐方案：先整理证据，再逐步拆分实现

三种选择：只整理目录的风险小但无法解决巨型运行模块；一次性重写架构容易改变实验行为、损失版本对照；推荐分阶段迁移，先建立可追溯目录，再做可验证的职责拆分。

建议的逻辑结构（目标，不是本轮已经迁移）：

```text
README.md / PROJECT_INDEX.md       当前入口与简短状态
car/autodrive/                     保留现有包路径起步，逐步拆职责
car/cloud_client/                  车端云访问与合同，不能标成自建云服务端
car/control/                      硬件接口与平台配置
research/                         历史分类、量化、基线实验代码
tests/{unit,replay,hardware}/      明确无硬件验证与现场实验边界
docs/architecture/                current、接口、架构图及版本对应
docs/operations/                  当前运行/部署/标定操作
docs/history/                     旧架构与已撤回设计，带替代关系
experiments/catalog/              跟踪的小型索引、样本表与证据 manifest
experiments/protocols/             固定实验协议、人工标注规则
experiments/analysis/              可复算的分析代码
coordination/                     日期化任务报告与交接
outputs/                          原始证据先原位保存，通过 catalog 组织
```

不建议第一轮移动 54 GB 数据。为现有文件建立统一“逻辑视图”即可消除大部分混乱，不破坏快照中的绝对路径。运行时状态/套接字与历史证据最终分开；新实验未来写入统一 run 布局，旧实验继续由 registry 解析。若后续做物理迁移，使用显式旧新路径映射，并修改回放解析器；仅加软链可能被当前 `resolve()` 路径约束拒绝。

### 阶段 A：冻结与目录化

记录 commit、工作树差异、活动 profile、环境/模型/标定版本。为每次实验建立 `session_id → run_id → sensor recordings → cloud events → source snapshot → review/derived` 关系。索引含实际时区、原文件名、boot/单调时钟域，不把本机美东命名直接当北京时间重命名。无法确认的字段显式标记未知。

对最终保全范围计算原始文件哈希，确认独立备份的具体主机/介质和容量后按重要性备份；本轮没有把同盘副本当成备份。优先不可再现的运行、标注、配置、patch 和模型版本。代码管理、数据保全、环境复建分别验收。

### 阶段 B：统一当前说明、隔离历史

更新 README 和 `docs/architecture/current.md`。把 PROJECT_INDEX 的历史长记录转为带日期的历史入口，保留迁移映射；当前页只写当前 commit/profile、证据边界和未完成事项。过时架构移出活动说明，混有实验事实的报告整体归档。旧链接、图注、manifest 路径必须检查。

### 阶段 C：重构运行职责，保持行为

先从 `onboard.py` 抽出归档/发布/配置装配，再抽取观测证据构建和最终执行，主循环只负责调度。稳定接口为：匹配帧的 Observation、LocalProposal、CloudAdvice、FusionDecision、AppliedCommand。保持现有 schema、时钟语义、队列上限、单步 deadline 和最终否决顺序。

保留当前规则融合策略，不能因为命名“融合”就升级成学习融合或 PWM 加权。把共用 YOLO detector 从历史分类目录迁出后，才处理旧实验包。最后整理旧测试和依赖分组。暂不引入微服务、消息中间件或额外数据库。

### 阶段 D：等价验收与论文图定稿

选定同一批冻结观测、配置、历史云回复，在重构前后执行无网络/无执行器回放，比较本地提案、融合来源、事件生命周期、deadline、最终 PWM/停止原因、记录数量。文件改名等纯路径变化单独列出。历史故障可保留为回归用例，但不能在结构重构中默默修复或撤销算法。

校验所有原始证据哈希不变、旧 run 可定位、历史清单可解析、当前入口可找到所需环境和模型。新的实车试验由用户另行安排；重构回放通过不等于过弯/居中性能改善。

## 当前架构图与源码对应

初版图文件位于 `docs/figures/2026-10-12/`，保留 PNG、SVG 和绘图源。用户随后不采用初版视觉设计并要求不提供 PDF；PDF 已移除，架构内容另整理为 `coordination/2026-10-12-current-architecture-drawing-brief.md`，交由其他 agent 重新绘制。架构只描述当前启用 profile 的代码结构；不是物理成功率或安全性证明。

| 图中部分 | 当前依据 |
|---|---|
| 操作者启停与会话管理 | `car/autodrive/web/drive_sessions.py`、`deploy/vehicle-autodrive-panel.service` |
| RGB → 异步 YOLOPv2 → 同帧观测/局部路径 | `perception/yolopv2_fusion.py`、`yolopv2_semantic.py`、`semantic_outer_route.py`、`track_colors.py` |
| LCC/视觉小步控制 | `control/lane_centering.py`、`visual_feedback.py`、`runtime/onboard.py:871` |
| 持续停车与候选构建 | `control/cloud_recovery.py:53`、`:276`；1 秒和 3 个新观测为配置默认，不是已标定的长尾定义 |
| 原图稀疏历史、异步上传与留证 | `control/recovery_worker.py`、`cloud_client/realtime.py`；1–3 帧，单个在途请求 |
| 云端 | 外部百炼多模态模型服务；本仓库保有客户端，不包含可核验的自建云服务端源码 |
| 合同与规则融合 | `cloud_client/recovery_contract.py`、`control/decision_fusion.py`；返回解释、风险及候选/hold，不含 PWM |
| 最终复核与执行 | `runtime/onboard.py:2277`、`control/drive_runtime.py`、`motion_guard.py`、`control/vehicle_control/platforms/rosmaster*.py` |
| IMU/编码器 | UART 遥测；IMU 参与短转时效/角度约束，编码器记录用于回放/复盘，不宣称完整里程计导航 |
| 雷达/深度 | `runtime/sensor_recording.py`、`web/sensor_publishers.py`；采集与记录健康门禁，不是当前几何避障/SLAM 输入 |
| 数据闭环 | `RunArchive`、传感器 journal、cloud journal、`replay_session.py` 和 `recorded_recovery.py` |

建议详细图用于第 3 章或导师讨论。摘要版只保留“本地感知控制—异常停车—云语义建议—车端复核—有界执行—证据归档”主线；若论文格式不允许摘要插图，可放绪论/总体设计或汇报首页。学校模板需由论文端核对。

图中不加入未启用的掩码突变触发、自建云 API 后台、SLAM、多传感器导航、已撤回的独立环境语义或学习式融合。若统筹选择将它们作为目标方法，另画明确标注的“拟研究架构”，不覆盖本图。

## 本轮验证与后续负责人

- 只读依据：`git status --short`、`git rev-parse HEAD`、`git ls-files`、`rg` 源码/引用检索、`wc -l`、`du -h -d 2 outputs`；Python 标准库读取文件元数据、session/run 目录及数据集 SHA-256；读取现有 torch dist-info 版本，不导入模型。
- 本轮新增审查/目录统计/图稿并追加索引；没有删除、移动原文件、改业务逻辑、安装依赖、启动设备、请求云推理、重启服务、提交或推送。
- 用户/论文统筹：审阅本方案，确定采用“当前实现”还是另行扩展目标方法；提供论文端开题/中期/实验协议引用核对，确定异机备份目的地。
- 本车端实现：方案确定后先做阶段 A/B，交付逐项保留/归档/清理清单与数据注册表，再分步拆运行代码并给出等价验证。
- 论文分析角色：在注册表稳定后核对样本、运行独立性和版本，形成论文图注及实验章节引用。当前没有自动向其他聊天发送消息。
