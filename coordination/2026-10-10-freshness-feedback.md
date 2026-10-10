# 端到端时效放宽与恢复等待修复

> 后续纠正：用户实际Start暴露field_trial.py启动校验仍要求5帧/最大.3s；本报告此前关于完整启动链已无残留/就绪的结论不足，已撤销。控制回放和54项检查结果仍有效，启动校验遗漏由 `coordination/2026-10-10-startup-policy-feedback.md` 补齐，并以真实入口在相机边界拦截的验证取代仅配置检查。

- 日期：2026-10-10。用户明确要求“过期不用这么严格，请帮我解决问题”。开发前只读确认 session `f62d846624244a7487726351a9dcbec7` 已用户结束且 stopped_verified=true；本次无相机/串口/底盘/GPU实验、服务重启、安装、提交或助手运动。
- 当前 HEAD `fce828b89b084eda293e83b99c8861f522858d8e`，保留全部既有 dirty。原始 run `outputs/autodrive_sessions/f62d846624244a7487726351a9dcbec7/runtime/runs/20261010_052958_291001_hardware`；归档2359/2359、dropped0/error=null，原始记录不修改。

## 原因与实现

最新实测200行的163行有路径，但115行执行过期，83行未通过恢复。推理中位约106ms并不等于完整闭环时效：读入源年龄中位.243s、路径分析约61ms，最终执行年龄中位.306s。0.3s否决清零五帧恢复导致几乎不移动。

仅修改 visual_feedback profile：模型 max_result_age_seconds .3→.6、stationary_corner.semantic_max_age_seconds显式.6、恢复不同新观测5→2。合并配置、控制器、最终执行及源捕获deadline共用.6s；不改全局或旧模式默认。不是消除延迟或权重改善，仍根据当前YOLO选择路线。

巡航许可仍 `min(当前新观测更新+.25s,源捕获+.6s)`，缓存不续期；弯道0.15s/5°、停稳等待.25s并要求等待后采集的图像；相机.25s、IMU.2s、独立截止和操作取消锁存保留。缺失/错误道路、黄色危险仍停；超过.6s仍停并重建恢复证据。白线/黄线颜色规则保持。

## 验证与数据对照

- 两项新增纯CPU消费行为检查在旧配置下预期失败：.4s路面被拒绝/第二个不同新观测无法前进。配置修改后通过，包含重复帧不凑数/不续deadline、真过期停止、停后采集再恢复、当前硬危险立即停。
- 定向命令：`outputs/field_yolopv2/venv/bin/python -m unittest car.test.test_visual_feedback car.test.test_yolopv2_primary car.test.test_control_render_timing -q`，46通过；`... -m unittest car.test.test_track_colors -q`，8通过。没有运行设备或全套冗长测试。
- 同一run的短窗口300行，保存源图片、road/lane/detections、源/执行时间、相机/IMU/external permission，使用真实感知几何/稳定器/控制器重放；只按两套配置重新计算时效否决和最终deadline，未知原始否决保持。直道200–239：40stop→12stop/28forward；停滞690–749：60stop→14stop/42forward/4turn-right；实时诊断1164–1363：198stop/2运动→17stop/159forward/24turn-right。后者与原记录动作计数一致，候选明显减少等待。
- 窗口从冷状态启动，固定原始输入不能预测不同运动后的新画面，不是新轨迹、物理通过率或真实新时延；候选200/200路径有效主要是放宽了时效，不是分割精度变化。虚拟PWM≤30、通道仍为原road子集且不含排除像素。
- 独立只读 reviewer `freshness_review` 未发现重要问题，确认面板后置profile参数覆盖旧启动默认、执行链没有残留.3s否决；cloud .3s目前disabled。未由审查者接触硬件。

## 产物与加载

`outputs/visual_feedback/20261010/freshness-fix/` 保存 before/after、incremental.patch、manifest/raw/source哈希、replay.py、paired-records.json、replay-summary.json、candidate-config.yaml及panel-state.json。仅两文件产品/测试变化，可按增量审阅，不以HEAD代替dirty源码。

最后只读socket仍finished/停止确认/hardware/总时限0，配置加载核对model=.6/control=.6/resume=2/input384。每次面板Start新建进程读当前profile，无须重启服务；由用户点击下一次开始进行实车观察。实际连续行驶、过弯与画面频率仍由下一run检验。
