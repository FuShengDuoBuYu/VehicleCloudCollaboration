# 当前车端回归

统一从仓库根执行 `./vehicle check`；可选 CPU 张量与本机 HTTP 集成分别使用 `--torch`、`--loopback`，WebSocket 假服务需 `--websocket-loopback` 及其开发依赖。详见[开发指南](../../docs/operations/development.md)。

本目录 `test_*.py` 使用假设备、合成输入或临时归档。测试文件名含 hardware 并不表示实车试验。测试之外的现场工具不得被自动 discover。

历史失效左右变道闭环脚本和JSON已原样归档到 [research/legacy](../../research/legacy/README.md)。新证据目录/来源回归在 `tests/`；不混入旧实验报告作为“测试通过”依据。
