# 开发与验证

先读 `PROJECT_INDEX.md`，再定位 `docs/architecture/current.md`。每次改动检查工作区，保留实验和他人改动；本次重构按用户要求不自行 commit。

## 本地验证

```bash
./vehicle check --static
./vehicle check --torch
./vehicle check --torch --loopback
# 仅安装了 requirements/test-websocket.txt 的已有环境才运行：
./vehicle check --torch --loopback --websocket-loopback
```

`vehicle check` 使用现有环境或 `CAR_PYTHON`，不自动安装。`tools/check.py` 固定读取当前测试目录的 `test_*.py`；排除研究、输出目录和硬件工具。默认不运行绑定端口的测试。WebSocket 假服务器是可选开发依赖，当前 Jetson 环境未安装；不能把未执行项称为通过。

模拟设备回归覆盖控制、云合同、超时/事件、短步截止、最后应用否决、归档、回放和启动器。启动器测试复制脚本到临时目录并替换服务命令；不要把测试中的命令移出假环境执行。`--torch` 使用 CPU 张量验证输入，不加载真实权重。

新工程测试放 `tests/`；现有跨测试 fixture/import 保持在 `car/test/`，统一从仓库根用检查入口执行。禁止为了目录整齐而破坏旧回放和导入。

## 结构与提交审查

- 业务模块不导入 `runtime/onboard.py`；历史回放/兼容入口可使用导出符号，新实现直接引用职责模块。
- 网络 worker 无 driver 引用；模型或云输出不能直接驱动执行器。
- 新策略通过现有合同扩展，保留 deadline、时钟、帧关联、最终否决与记录。
- 实验分析和产品逻辑分离；固定输入回放不产生新轨迹。
- 原数据迁移必须有校验与映射；当前数据采用原位注册。
- review 前运行检查、核对文档相对链接和迁移清单，报告哪些检查未运行。

`--static` 验证 Python/shell 语法、维护中文档的本地链接及历史迁移内容哈希；不验证不可变历史原文里的旧上下文链接，也不联网检查外链。

本轮基线已有两个过时fixture：斜边不再是有效入弯观测，但仍保留位置诊断；本地颜色回放fixture没有云journal，须明确关闭云恢复。这次只修正fixture，不放松生产代码门禁。
