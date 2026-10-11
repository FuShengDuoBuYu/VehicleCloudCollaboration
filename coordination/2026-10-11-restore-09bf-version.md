# 2026-10-11 按用户要求恢复 09bf 实车版本

用户引用“第二个弯通过、第三个弯卡住、出弯贴左线”的历史反馈，并明确回复“恢复当时的版本，不要当前代码”。对应北京时间18:45:54–18:50:00，session `09bf862a43cb477f9247a2d97212d720`，run `20261011_064559_596659_hardware`。目标是这次实车实际使用的版本，即05a3反馈修复完成之后、09bf后续修复之前。

## 恢复依据与结果

- 基础提交仍为 `f211ff6c00a770b4c46f5c82b3cae92859f585c8`；没有reset、提交、拉取或切换分支。
- 在临时目录以该提交的car/docs构建文件树，应用`outputs/diagnostics/20261011-09bf-feedback-fix/before.patch`，排除项目索引。补齐05a3归档的未跟踪回放/测试文件及此前c8测试。
- 与`20261011-05a3-feedback-fix/source-manifest.json`中10个历史哈希全部一致，同时与`20261011-09bf-feedback-fix/before/`中8个文件逐字节一致。
- 共恢复26个不同的源码/测试/面板/文档文件；5个后增模块或测试移出运行代码：yaw_observer、path_obstacles、test_third_corner_feedback、test_environment_recenter、test_boundary_departure。它们和被覆盖文件均先备份，不清理原始数据。
- 运行代码恢复当时的控制、云恢复合同和面板。后续独立环境语义、连续回中、漂移补偿、全量云事件面板、越界锁定等新增修改不在当前运行版本中。旧版的偏左、第三弯停止等已知行为也未在此次回退中另行修正。
- `.env`、密钥、依赖、模型、视频和传感器记录保持原样。项目索引和历史复盘保留，不将过去的修复记录伪装成当前已生效状态。

## 验证和加载

- 恢复前只读实时面板确认最新会话`503e3d5c5a6f4a94a5168a8e0a8fbf89`已finished/running=false/stopped_verified=true。
- 14项相关纯CPU离线检查通过，0.187秒；没有模型推理、云请求、设备访问或电机运动测试。
- 按此前用户“仅重启面板”的授权，重启`vehicle-dashboard.service`。实际HTTP返回的dashboard.js与历史哈希`7e4ce1a5904517854cf562b122d4164b81dc500f2dd3eefd37d52ec4853377d8`一致，驾驶仍finished/running=false/stopped_verified=true。没有重启传感器/驾驶服务或替用户点击Start。
- 下一次用户点击开始会由新驾驶子进程读取恢复后的代码；本轮没有新实车运动结果。

完整证据：`outputs/diagnostics/20261011-restore-09bf/`，含`restore-plan.json`、`restored.json`、`target/`、`current-backup/`、`working-tree-before.patch`、`checks.log`、`dashboard-verification.json`。后续由用户安排是否试跑，车端以新的run记录对照此恢复版本。
