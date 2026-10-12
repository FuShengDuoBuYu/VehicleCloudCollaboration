# 记录与离线回放

当前模块与控制边界见[系统架构](architecture/current.md)。早期外圈入口及当时结果已[原文归档](history/2026-10-09-outer-loop-record-replay.md)，不代表当前面板配置。运行入口见[运行说明](operations/runtime.md)。

## 冻结输入回放

从[实验注册表](../experiments/README.md)取得一个完整 run 的原路径，指定其冻结配置和一个新的派生输出目录：

```bash
./vehicle replay \
  --run /绝对路径/到/运行目录 --profile recorded --mode recorded \
  --output /绝对路径/到/新的回放目录
```

`recorded` 使用归档 NPZ 中的模型输出、消费序号和时间，执行当前后处理及控制；不会重新推理、请求真实云或创建底盘。存在云恢复 journal 时读取历史请求/结果并匹配事件，缺失证据不能声称重现云恢复。它不重现 CPU、执行器或归档耗时，也不产生新实车轨迹。

处理预算使用归档的完整精度许可字段，旧记录退回归档毫秒数；缺失或舍入临界值标为未知并限制为停车。分析机器当前负载不再产生新的预算否决，`summary.json` 记录时间来源和未知行数。

工具要求终态停车、连续日志、末样本和帧数一致；不完整记录保留作为故障证据，不能为通过回放而补写原始字段。最终应用否决只在原日志明确记录时使用，旧日志缺失时保持未知。

`--profile <候选profile>` 可比较当前候选配置，但须与 `--profile recorded` 区分。`--mode auto` 在缺 NPZ 时回退到视频推理；需要确保绝不加载模型时显式使用 `--mode recorded`。`--mode video` 使用有损视频重新执行模型，不能等价重现原异步输入或过期时序。

## 原始与派生记录

- `raw.mp4` 是控制循环采集流，`annotated.mp4` 对应实际消费的模型图像；两者时间线不同。
- `semantic_inputs/<sequence>.npz` 保留模型原图、道路/车道掩码、检测及时间，通过 CSV 序号关联。
- `onboard_log.csv`、`status.json`、冻结 YAML 和云 journal 描述输入、提案、最终应用与终止。
- 新运行另有 `provenance.json` / `source.zip`，记录代码、配置和模型身份。旧运行缺失这些文件不回填。
- 回放生成 `commands.json`、`summary.json`、视频及回放配置；输出必须使用新目录。没有逐段预期注释时 `expected_behavior_verified=null`，不能称为实车通过。

整场会话分析还可用 `car/autodrive/tools/replay_session.py`，其可用参数以 `--help` 为准。分析输出与原始记录分开，论文引用注明源 run 路径、分析代码和输入哈希。
