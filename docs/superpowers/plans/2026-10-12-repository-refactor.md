# Repository Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task.

**Goal:** 在保留实验和行为的条件下统一项目入口、拆分运行职责、建立可追溯实验注册与可验证开发流程。
**Architecture:** 保留car包和启动兼容边界；从巨型模块抽出已有职责，显式组织当前/历史文档，元数据工具独立于设备和推理。
**Tech Stack:** Python 3.8, unittest, existing numpy/OpenCV/PyYAML; no installation.
**Spec:** ../specs/2026-10-12-repository-refactor.md

## Global Constraints
- 不commit、push、merge；按用户要求保留未提交review。
- 不修改原始数据、模型、密钥、控制阈值；不运行硬件或真实云；不安装。
- 先隔离验证，集成前检查主目录并发改动，保留上轮产物。

## Review Focus
- 模块提取后共享时钟/旧导入与测试注入点保持有效。
- 源码dirty哈希和配置/模型身份准确，不采集秘密。
- 目录扫描面对损坏JSON、缺失关联、符号链接和活跃run时不捏造完整性。
- 无硬件测试入口不得discover现场驱动脚本。
- 历史文档和原始证据路径保持可解析。

## Task 1: Baseline and validation entry
- [x] 静态检查test_*.py副作用，记录无硬件基线；现场入口不运行。
- [x] 增加tools/check.py统一静态/单元检查与明确的可选loopback组，不装依赖。

## Task 2: Runtime and control separation
- [x] 提取runtime sources/archive/publication/corner_runtime/perception/configuration等职责，保留onboard兼容导入与编排。
- [x] 拆drive_runtime为轮速映射/驱动、运动门禁、历史续行、watchdog；保留旧导入。
- [x] 抽出当前YOLO detector，历史路径兼容；测试老入口和新模块。
- [x] 运行控制、归档、模型注入、完整无执行器入口与回放测试。

## Task 3: Evidence catalog and provenance
- [x] 先写缺失/坏JSON/路径逃逸/secret排除/源码身份测试。
- [x] 实现experiments/catalog.py只读扫描+显式输出，生成runs/datasets索引与问题清单。
- [x] 实现runtime/provenance.py，并在RunArchive初始化时保存新run来源；不重写旧run。
- [x] 用原数据只读生成catalog并核对计数和原路径。

## Task 4: Repository and dependency organization
- [x] 归档历史失效闭环入口及报告，保留现用兼容客户端与Pi配置。
- [x] 分离运行、研究、开发依赖及Jetson环境事实，不pip安装。
- [x] 统一README、当前架构、操作与测试说明；缩短PROJECT_INDEX并原样保存旧索引，生成路径迁移清单。

## Task 5: Full verification and review
- [x] 运行无硬件基线/新增测试、静态检查、导入边界、文档链接、记录目录一致性。
- [x] 用冻结历史run做无推理/无云/无执行器的重构前后回放对照；限制写入新派生目录。
- [x] 新上下文最终审查并修复可复现问题；报告风险与证据边界。

## Task 6: Reviewable delivery
- [x] 对主工作区逐文件并发检查后同步未提交改动，不merge/commit。
- [x] 更新交付报告：文件变化、移除/归档理由、测试结果、证据路径、未完成的实车验证。

最终状态：全部工程任务完成，576项主目录测试通过、867帧冻结回放等价。用户review待进行；未commit/push/merge。执行证据见 coordination/2026-10-12-repository-refactor.md。
