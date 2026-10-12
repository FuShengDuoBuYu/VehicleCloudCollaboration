# 车云架构图

状态更新（2026-10-12）：用户不采用本目录初稿的视觉设计，改由其他 agent 绘制。请使用[当前架构绘图交接文档](../../../coordination/2026-10-12-current-architecture-drawing-brief.md)作为内容依据。本目录保留初稿来源记录；PDF 已按用户要求移除，生成脚本不再输出 PDF。

基线：2026-10-12，`664793438d9d2cf51dae603e8d6fcc45902f6fb1`，活动入口配置 `rosmaster_jetson_yolopv2_visual_feedback`。图稿描述磁盘当前代码，不验证既有后台进程已加载此提交。

- `current-architecture.{png,svg}`：总体架构初稿，视觉方案未获用户认可。
- `method-overview.{png,svg}`：简化方法概览初稿。
- `render_architecture.py`：初稿绘图源。PNG 供预览，SVG 为矢量版；SVG 字体转轮廓，避免跨主机缺少中文字体。

生成命令：

```bash
MPLCONFIGDIR=/tmp/vcc-architecture-mpl python3 -B docs/figures/2026-10-12/render_architecture.py
```

使用现有 matplotlib 与 Noto CJK 字体，不导入车辆模块、不访问设备或网络。

建议图注：当前车云协同系统由车端本地感知与控制闭环、异步云辅助恢复及实验记录链组成。车端基于同帧视觉证据构建局部提案，在持续停车时提交原始图像、上下文和受限候选；外部多模态服务返回结构化语义建议，车端结合新观测复核并经安全约束执行。雷达与深度数据用于采集、记录和健康检查，尚未用于本图中的局部几何规划。箭头表示信息或控制关系，不表示所有模块独立部署。

图注补充配置：当前代码指定云 provider 为 `qwen-realtime`、模型标识 `qwen3.8-omni-flash-realtime`、合同 `road-recovery-v1`。这只是仓库中的模型配置与调用协议，不是本轮对第三方服务可用性或内部架构的验证。候选为空/硬条件不满足的诊断请求不能授予运动许可。

完整代码对应、原始证据边界及未启用功能见 [审查报告](../../../coordination/2026-10-12-repository-review.md)。简图中的状态检查允许正常本地提案通过，只有持续停车才发起云请求；并非每个控制周期必须停车或访问云端。

论文端定稿时可移除版本日期等审阅信息，但必须在图源/图注旁保留对应代码版本。当前实现与拟研究的时序触发方法应分别绘图；本图不能用于声称全系统实车实验已通过。
