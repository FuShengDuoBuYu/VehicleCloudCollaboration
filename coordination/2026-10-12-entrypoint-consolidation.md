# 启动与协作入口收敛

2026-10-12（Asia/Shanghai）。用户在上一轮review中要求修复run脚本散落、skills/AGENTS多路径；继续不commit。基线是6647934及上一轮全部未提交重构，不覆盖已有工作。

## 设计与边界

- 唯一日常入口 `./vehicle`：dashboard、panel、drive、monitor、replay、trial、wheel-check、check、catalog。无参数/帮助不启动服务或硬件。
- `tools/vehicle_cli.py` 解析命令、选择环境和构造进程；`tools/process_lifecycle.py` 统一设备监控释放/恢复与信号等待。当前控制算法、模型、数据路径保持。
- drive显式选择当前visual_feedback，默认电机禁用；trial必须明确运动确认，wheel-check沿用现有抬轮确认。replay默认recorded，避免隐式GPU推理。
- 根目录只暂留三个无业务逻辑的旧脚本：已安装systemd引用dashboard/panel；已加载的旧supervisor仍可能调用experiment。其余历史run脚本与check.sh退出活动目录；源码版本可从Git及本任务前像还原。
- 仓库内只有一个AGENTS.md。技能实体统一到 `.agents/skills/<name>/`，移除根skills和发现软链接；技能引用根规则及项目事实入口，不复制过时架构结论。
- 全局Codex插件、其它仓库、既有实验输出不在清理范围。仅更新仓库服务模板，不部署/重启、不启动硬件或真实云，不安装依赖。

## 实施与验收

- [x] 记录已有修改前像，隔离实现；检查旧服务及源码调用路径。
- [x] 先补CLI路由、默认无副作用、监控释放失败、恢复失败、信号只转发一次及原参数保留的测试。
- [x] 实现统一CLI和公共生命周期，更新supervisor、服务模板、兼容入口及源码溯源范围。
- [x] 合并规则和技能实体，核验元数据与引用；历史来源以原始哈希记录，不保留活动重复副本。
- [x] 更新所有维护中的文档，静态检查防止入口/规则结构再次分叉。
- [x] 无设备全量回归、独立review、按前像核对后同步主目录；留未提交review。

原有服务路径依据：只读检查 `/etc/systemd/system/vehicle-*.service` 的ExecStart。没有执行systemctl修改命令。

## 审查与验证

- 基线HEAD：`664793438d9d2cf51dae603e8d6fcc45902f6fb1`；所有改动未提交、未暂存。本轮在隔离目录实现，按初始文件哈希核对后同步，避免覆盖用户并发修改。
- 启动专项26项通过（`python3 -B -m unittest tests.test_vehicle_cli car.test.test_dashboard_launcher car.test.test_wheel_check_launcher`），覆盖路由、参数边界、确认门禁、设备监控释放/恢复失败、信号等待和旧路径转发。
- 独立审查发现并修复：轮校准交互提示被日志重定向隐藏；中断试验且驾驶进程退出非零时仍触发回放。两项先复现失败，再修复；复审3项针对测试通过。记录见 `entrypoint-2026-10-12/review-regressions-{red,green}.log`。
- `tools/repository_layout.py` 检查唯一入口/规则/技能布局及旧规则归档哈希，纳入 `./vehicle check --static`；两项技能通过skill-creator结构校验。
- 试验数据、日志、视频、模型、环境保持原位。退出活动目录的入口以及本轮更改前像存于 `entrypoint-2026-10-12/preimages.zip`，对应身份和最终文件变更列表见 `sync-manifest.json`。原始规则另有可读历史文本及 `rule-migrations.json`。
- 最终主目录执行 `CAR_TEST_REPLAY_EVIDENCE_DIR=/tmp/vcc-recovery-full-final ./vehicle check --torch --loopback`：593项通过（54.457秒）；日志 `entrypoint-2026-10-12/main-final-rerun.log`。真实硬件、云端、服务部署与运动均未验证；可选WebSocket假服务测试依赖未安装，未运行。
- 保留不稳定项：首次全量云恢复回放出现1个未匹配事件；三次单测与最终全量未复现，根因尚未确定，没有宣称修复。仅提前保存测试合成证据，未放宽断言，详见 `entrypoint-2026-10-12/review.md`。最终合成run与回放封存为 `synthetic-recovery-validation.zip`，不属于实车实验数据。
- 同步后1927项未涉及的基线身份一致，前像ZIP中的37个原文件哈希一致；HEAD未改变，Git暂存区为空。最终文件身份见 `entrypoint-2026-10-12/final-state.json`。
