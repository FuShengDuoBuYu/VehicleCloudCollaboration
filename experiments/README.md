# 实验注册与证据保全

原始数据保留原位，注册表提供统一入口。不要因为目录叫 outputs、diagnostics 或 corner_fix 就删除；失败运行、撤回实现和中间分析可能解释论文结论。

```bash
./vehicle catalog --root /path/to/VehicleCloudCollaboration \
  --output /path/to/new-catalog
```

输出必须是新目录，且不能在原始 outputs 或数据集内。工具不加载模型、设备、云认证或原始大数组；只读运行元数据，数据集以内容哈希建立别名/标签关系。损坏JSON、重复run ID和被拒绝的符号链接列入 issues，不把缺失证据视为完整。

- `runs.json`：session → run → 原始传感器记录 → cloud event；旧run缺失源码来源时明确 unknown，不能填成扫描时的HEAD。
- `datasets.json`：一个内容哈希关联所有原路径、来源组和人工标签。重复文件不删除；split 未确认时为 null。
- `integrity=not_verified`：目录存在/身份关联不证明全部视频或journal完整；正式论文引用仍须逐run核验原始数据和代码/配置。
- 同名run可能是副本；以路径限定身份，并报告重复，不能合并计为两次独立实验。

新运行初始化在原有配置快照之外保存 `provenance.json` 与 `source.zip`。manifest含commit、dirty状态、源码哈希、配置快照哈希、按 primary/int8 角色区分的启用模型哈希和允许的依赖版本，不含环境变量、key或代理。源文件在生成期间变化会报错，不能启动带歧义版本的驾驶。

历史录制时间按原文件名保留，时区/boot未知则不猜测；目录时间不自动当北京时间。原始视频/无损观测/journal只读，分析输出使用新目录。Git不备份被忽略的outputs，异机备份状态必须单独核验。

本轮初始目录位于 `catalog/2026-10-12/`，包含车端实存的全部runs目录和面板会话，不等于全部都是正式实验。历史报告与派生分析仍在 `coordination/` 和原outputs路径，目录化不会修改它们。
