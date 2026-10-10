# 当前近路接近、短转与可见余量：实施记录

2026-10-10，车端 `/home/jetson/VehicleCloudCollaboration`。用户已批准方案；本轮只修改代码、做定向CPU检查和冻结数据短回放，未启动电机/相机/串口/GPU、未安装、重启服务或提交。HEAD仍 `fce828b89b084eda293e83b99c8861f522858d8e`，已有大量dirty/未跟踪文件保留；源码版本需结合本轮快照，不能仅由HEAD复原。

## 修改后的行为

- 当前visual profile启用 `stationary_corner.visual_corner_guard: true`。前横向边界还远时，selector从当前近直路选前进目标；runtime只在当前中心前方道路及可见余量支持时短步前进。前边界达到已有near判据、真实右出口支持、近方向需要右转且IMU有效，才开始小pivot。边界位置来自每次匹配YOLO输入，不使用固定多走厘米数或90°结束目标。
- 开始pivot后保留弯道阶段；近方向暂时看直不会立即巡航。软件停止时刻之后采集的两个不同新观测，均需近路对齐、横向前边界消失、前方道路/余量有效，再解除弯道阶段。缺道路、时间倒退、最终执行否决会清空确认，但保留弯道阶段。
- 新 `visual_clearance.py` 在匹配当前road与yellow掩膜上检查近段路径每条连线，以及中心前进带；当前余量为图像宽度的.06（320宽时20px）。缺颜色证据或缺road不能授权，不补道路；白色允许规则及黄色排除保持。新增CSV/实时状态字段记录阶段、出口确认数、路径最小余量、要求余量、前进支持和拒绝原因。
- live与录制回放共用 `semantic_outer_route.route_options(config)`，避免派生配置不一致。启动校验要求该功能配套启用visual反馈及匹配颜色处理。
- input384/FP16/完整YOLO头、.6s模型/控制/应用时效、恢复2新观测、巡航双截止、短步.15s/5°/等待.25s、相机.25s/IMU.2s、取消与记录门禁保持。会话总时限仍0。

## 检查与审查

三个关键行为及稀疏路径漏检先观察失败再修复。独立只读审查发现两项Important，均已CPU复现失败并修复：全局capture退序仍计出口、最终application veto仍保留出口计数/本周期解除状态。新增测试覆盖退序输入、第一确认后veto、第二确认解除后veto；无设备连接。

最终定向检查107项通过（12.379s），命令：

```text
outputs/field_yolopv2/venv/bin/python -m unittest car.test.test_visual_corner_guard car.test.test_visual_feedback car.test.test_semantic_bend_path car.test.test_track_colors car.test.test_outer_loop_trial car.test.test_yolopv2_primary car.test.test_control_render_timing car.test.test_outer_loop_replay -q
```

实际panel历史参数执行 `onboard.main`，读取当前profile/标定证据，在CameraSource构造前拦截，前置校验通过；未打开设备。脚本 `outputs/visual_feedback/20261010/startup-policy-fix/preflight.py`。`git diff --check`通过；本轮没有全仓广泛测试。

## c92a冻结输入配对短回放

原始session `c92a8904c03c4f76822ef305e532a649`，run `outputs/autodrive_sessions/c92a8904c03c4f76822ef305e532a649/runtime/runs/20261010_054444_110581_hardware`。465/465归档，上一轮用户End且停止确认。诊断及照片关联见 `coordination/2026-10-10-c92a-corner-feedback.md`；原始NPZ/视频/日志未修改。

每段冷启动，用原始掩膜/源图/目标头和捕获、控制、执行时钟；按同一.6s时效重新计算许可，保留未知历史执行否决；虚拟PWM≤30，corridor为原始road子集且不含排除像素。共175行：

| 样本区间（末端不含） | 旧策略 | 新策略 |
| --- | --- | --- |
| 80–130直道，50行 | 43运动/7停 | 37运动/13停 |
| 170–205首次pivot，35行 | 5pivot/8巡航/22停 | 9短前进/26停 |
| 205–235转后，30行 | 4pivot/5巡航/21停 | 8短前进/22停 |
| 300–330road减少，30行 | 9运动/21停 | 2短前进/28停 |
| 435–465最终停滞，30行 | 30停 | 30停 |

sample190旧pivot-right→新forward；该输入上前边界ratio.575，尚未达到near .74，新路径可见余量约47.75px。直道多6停：110/111/116/117余量不足，112/113恢复等待；不能称直道无回归。最终原始road缺失仍停，不能称本轮修复了已经停到的最终姿态。

冻结回放不能预测新动作之后的新图像、物理轮迹或新实时延迟。20px是可见图像代理，未做车身/轮下标定，不能保证不压线。软件 `stopped_at` 是控制器停止时刻，非底盘零输出完成/实际停稳测量。现场图片并未与某帧精确同步。

## 交付与下一轮

证据目录 `outputs/visual_feedback/20261010/corner-approach-fix/`：before/after、`incremental.patch`、`source-sha256.json`、`candidate-config.yaml`、`replay.py`、`paired-records.json`、`replay-summary.json`、`verification.json`、审查修复记录和只读 `panel-state.json`。源输入/CSV哈希随回放保存。

2026-10-10 10:11–10:12 UTC只读socket确认available=true、finished/running=false、用户End、stopped_verified=true、exit0、hardware、总时限0。下一次用户在页面点击Start会由新进程读取本版，无需本轮服务重启；助手未发送Start/End/运动指令。下一负责人仍本车端聊天：以新run分析实际接近距离、短转、余量停顿和轮迹，不宣称已成功过弯。
