# Jetson YOLOPv2 场地配置实施记录

> 执行方式：本聊天继续实施；复用已有隔离工作树，开发修改不与其它聊天并行写同一文件。

用户任务：先获取最新车云协同代码，配置YOLOPv2 GPU环境并实测，必要时调整LCC，使行驶路径由YOLOPv2感知决定。用户随后明确云端暂缓，等待其选定供应商/模型。

## 当前设计

保留已修复的Rosmaster底层与安全门禁。新增显式`perception.mode=yolopv2`，只使用新鲜YOLOPv2可行驶区域和车道线构建行驶通道，不调用黄色边界/路面颜色规则；现有中心线控制器负责将几何转换为有界车轮提案。空掩码、过期、推理错误、缺少车底通道锚点和不可信全图掩码均停车；禁用旧弯道历史续行。模型不是端到端油门/转向网络，检测框和安全距离能力按实际权重输出单独验收。已有经典模式保持兼容，新的Jetson场地profile显式选择GPU模型。

## 步骤与验收

- [x] 保存55项本地dirty文件，fetch并快进到6bf8693；README局部stash后自动三方合并，备份保留。
- [x] GPU依赖：JetPack 5.1.1对应CUDA11.4/cuDNN8.6运行库、NVIDIA PyTorch2.0.0+nv23.05、隔离venv，官方YOLOPv2 V0.0.1权重。验证真实CUDA张量运算与模型前向，不以安装成功作为GPU通过。
- [x] 先写并复现失败用例：YOLO模式不依赖经典颜色，过期/故障/空通道停车，模型初始化失败不打开底盘；然后实现最小语义通道及配置入口。
- [x] 解除YOLOPv2导入对CLIP/YOLOv8依赖的耦合，增加适配GPU的精度和可选保留原始图像比例；保持旧调用默认行为。
- [x] 固定相机场地采样、电机禁用YOLO完整推理与运行日志，保存掩码/叠加图/延迟/设备/权重哈希；模型对场地无效时如实停车并说明需要数据与微调。
- [x] 独立审查本轮增量、相关回归、主工作区哈希前置核验后合回；交付配置入口、部署/实测报告和下一负责人。

## 实车前置与边界

本轮已明确授权环境安装与代码配置；云端调用暂缓。上一次架空动作授权不自动代表本次新场地地面行驶已安排。轮位/正方向、透视/安装姿态和停止保护仍需完成，两个calibrated门禁不因GPU可用而解除。此阶段先完成静止的真实感知和虚拟控制验证，地面低速试跑需当次具体场地安排。

## 证据

证据根：`/home/jetson/VehicleCloudCollaboration/outputs/field_yolopv2/20261008/`。
隔离工作树：`/home/jetson/.codex/worktrees/car-hardware-acceptance/VehicleCloudCollaboration`，分支`codex/jetson-yolopv2-field`。
官方来源：NVIDIA PyTorch for Jetson安装指南（v511/nv23.05 wheel），YOLOPv2作者GitHub V0.0.1 release。云端`.env`为600权限且Git忽略，记录只存是否配置，不保存密钥。

## 完成本轮配置后的状态

实施/配置/无执行器验证已交付，详见 `coordination/2026-10-08-jetson-yolopv2-field-report.md`。完成上述配置步骤不表示场地自动驾驶已验收；真实运动、场地模型质量、相机/轮位标定仍按报告继续。云端等待用户选择。
