# 2026-10-11 05a3 出弯回中、云请求和面板修正

用户在05a3复盘后明确要求开始修改。沿用vehicle-cloud-thesis工作流，保留既有dirty/未跟踪文件，没有提交或拉取。HEAD `f211ff6c00a770b4c46f5c82b3cae92859f585c8`。本轮未启动电机、相机、模型推理或新云请求；仅离线CPU消费原记录。用户此前仅重启面板的授权用于本次HTTP服务加载，不重启传感器/驾驶程序。

## 输入和修改

源session `05a3c25bc5784be48c9e22fd39d6073b`，源run `20261011_062723_573638_hardware`，完整路径 `outputs/autodrive_sessions/05a3c25bc5784be48c9e22fd39d6073b/runtime/runs/20261011_062723_573638_hardware/`。首弯成功、随后偏左/压左线、最终人工End是用户实车反馈；本次修改尚无新实车轨迹。原始CSV/NPZ/视频/云响应均未改写。

1. `semantic_outer_route.py` 在横向前沿跨度检验通过后才设置front_observed。斜侧边缘被拒绝后不再错误进入approach，已有LCC向右纠偏得以持续。没有引入按时间或源录像sample号转弯，没有修改已有效的小右转/前拱流程、轮速增益或最低PWM。
2. `cloud_recovery.py` 从当前匹配图像重建的合法地面生成scene_support。原始road掩膜突变时，只有当前候选ID/action/assessment一致、余量变化≤2px、方向变化≤.025、地面1-IoU≤.1、缩小RGB平均通道差≤8且最大通道差>20的像素≤5%、新鲜IMU航向差≤2°，才继续保留请求。任何真实安全否决仍先停车；回复到达仍须新帧复核候选，限制≤.15秒/5°并经过最终driver截止时间。没有无限重试、历史掩膜续行或云PWM直通。
3. 当前有效event_id与last_event_id分离；veto/timeout不再残留pending。保留最近原因和短步曾获准标志；dashboard按当前或最近事件读取白名单归档，显示收到但未采用，或短步获准后停止。历史回复只供显示，融合器仍丢弃已取消事件，不能恢复授权。step_authorized表示提案获准，实际执行以applications/PWM为准。

## 验证与证据

派生目录 `outputs/diagnostics/20261011-05a3-feedback-fix/`。before/和before.patch保存本轮前状态；final-source/、source-manifest.json保存最终相关源码和哈希。原先诊断回放位于 `outputs/diagnostics/20261011-05a3-run-review/replay/`，867/867动作和PWM复现，作为对照。

最终命令使用 `PYTHONPATH=car OMP_NUM_THREADS=1 outputs/field_yolopv2/venv/bin/python`，OpenCV固定单线程。135项相关CPU检查通过18.167秒，见tests-final.log；未运行硬件测试。首轮与录像回放并行时，模拟入口的25周期回放出现24/25匹配，其余134项通过；单独复核该项通过，随后135项全通过，未删减断言。该次并行环境差异的具体原因未单独验证，不把它描述成已定位的实车故障。JS在V8仅编译检查通过。

第一次整段迭代回放replay/已经验证：前442行首弯动作/PWM完全不变；14个错误等轮速sample全部恢复向右差速（30/21或30/22）。初版云候选要求完全相等，实际窗口暴露heading .775→.76875的像素量化变化仍误取消，因此进一步加入上述小范围容差，最终整段结果以replay-final/和comparison-final.json为准。

独立原云窗口 `replay_cloud_windows.py`/cloud-windows.json 使用原请求帧、mask与候选签名、PNG像素一致性断言，并在原控制消费时刻注入真实归档回复；不外发、不构造云建议。候选和hard_safe与原journal逐行核对，再经过DecisionFusionAdapter及电机禁用SafeWheelDriver：

- 请求687→回复717：原来689被raw mask取消；现在保持到713，当时录制的stationary_external_allowed=false，仍取消且全程零PWM。没有放过真实外部门禁。
- 请求761→回复792：原来762即取消；现在792产生一次虚拟 `[30,30,-30,-30]`，793最终driver已因截止时间归零。
- 请求792→回复820：原来793即取消；现在820产生一次虚拟 `[30,30,-30,-30]`，821最终driver归零。

这三个窗口各自从原请求开始隔离复核，不是把三段拼成新的闭环物理轨迹。固定录像不会因新算法向右回中而改变后续视角；整段回放若产生新请求，缺少精确对应真实回复则必须标unmatched并停车，不能套用邻近回复。

最终完整867周期结果：634停、104前进、107差速右转、22原地右转；对照为678停、118前进、50差速右转、21原地右转。共58行PWM改变，前442行完全一致，原14条纠偏覆盖全部消除。云端实际虚拟PWM出现在531和792，原记录只有531；源事件精确匹配3、原回复消费3、新请求未匹配1、网络0。新恢复动作改变后续请求时机，未匹配请求无真实对应回复，因此离线保持停车，不能称4个云事件全部在新整段轨迹复现。`verify_final.py`检查首弯、14条纠偏、所有已记录最终应用否决、PWM上限30、独立窗口一步后归零，以及最终源码哈希，全部通过；结果comparison-final.json。软件修正证据与新实车过弯结论分开。

## 加载和下一步

已在只读HTTP确认drive_control.finished/running=false/stopped_verified=true后，仅执行 `sudo systemctl restart vehicle-dashboard.service`。重启后/api/state与/dashboard.js可用，服务JS哈希与本地一致，仍是原05a3已结束会话；证据deployment.json。没有替用户点击Start。驾驶控制在用户下一次点击开始时由新子进程加载修改。

下一负责人：用户现场安排下一次低速运行，重点看首弯后能否回中、左边界距离是否恢复，以及云回复后的短步是否有效；本车端聊天随后读取新run视频、IMU/编码器、候选和实际PWM，产出与本轮对照的轨迹复盘。当前结果支持软件缺陷修正，不保证下一次完整绕圈或物理不压线。
