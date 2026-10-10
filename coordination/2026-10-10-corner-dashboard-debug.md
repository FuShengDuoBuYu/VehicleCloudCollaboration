# 2026-10-10 弯道停滞与面板画面修复

用户实车反馈：直道动作符合预期，弯道转一小步后卡住；静止时识别边缘跳动，页面刷新慢、启动黑屏/过期。按用户要求主要分析实车数据，仅做短回放和必要检查。

## 原始证据和原因

- HEAD `fce828b89b084eda293e83b99c8861f522858d8e`，已有 dirty 工作树保留，没有提交或推送。完整工作树状态见 `outputs/visual_feedback/20261010/corner-dashboard-fix/git-status-after.txt`。
- 用户操作的主运行 `outputs/autodrive_sessions/3b88faac3c514901ac20cf052c05f4b6/runtime/runs/20261010_031040_396428_hardware`，1702 行，最后运动 sample1209。全停重启运行 `outputs/autodrive_sessions/7cdc03d5592345ca8c203d9831769b18/runtime/runs/20261010_031318_288804_hardware`，583 行全部 stop；两次均由用户结束，退出0、停止确认、cleanup_errors空。
- 原始 CSV 与 lossless NPZ 不作修改。读取 selected samples1208、1210、1214、1220、1300、1699：转动后内圈弯曲标线进入右下方，旧全图行分叉条件拒绝 front-corner 目标；备用路线又要求预览左边界 x>0，持续报告 `YOLO outer left edge missing at preview`。当时传感器记录、external motion permission 正常，并非电机启动失败。
- 各运行末尾35个 NPZ 的相邻 road-minus-lane 掩膜 IoU 中位约0.9953/0.9954，相邻原始相机图像平均绝对差中位4.7755/4.6596（8bit图像）。这是当前不同采集帧的比较，不能据此归因于GPU随机性。详细统计和命令脚本保存在输出目录。
- 实测主运行 YOLO 推理耗时10/50/90分位59.7791/63.489/81.6246ms；重启运行60.5624/71.882/94.2696ms。采集、异步推理、控制及网页发布有独立时序；不保证固定100ms新结果。
- 页面旧状态/图像轮询约500ms，运行时发布配置5Hz；LivePublisher直接覆盖JPEG有并发读取不完整图像的风险。启动实验时 launcher暂停相机监控并重新建立模型/相机，首个 CUDA 结果可因预热过期（主运行首行semantic age5.0989s），旧页面误以为一般运行异常。
- supervisor.start 用update沿用上一轮 archive/completed_at/termination 等，已在新一轮session.json中观察到旧运行元数据。

## 本轮修改

- 仅视觉反馈模式：右出口由目标邻近当前道路支持确认，远处内圈分叉不再否决所有候选；逐像素检查车前到右侧目标射线不可跨过当前 road/lane 排除区域。仍需当前 ego-connected道路、有效完整局部路径、新鲜模型/相机/IMU及记录门禁。旧角度模式不改；保留现有短步时长、角度上限和停顿，未写入固定90度转弯，也没有移除失效停车。
- JPEG临时文件写完后原子替换；浏览器先离屏解码再换图，损坏/超时保留最后完整图。图像请求目标100ms，状态轮询目标100ms，深度独立较慢刷新。视觉反馈profile和静止监控均发布目标10Hz。
- 每次session重置状态；开始阶段等新鲜语义结果后才显示running；页面明确标注相机切换/模型预热，保留上一帧但不把旧帧当实时控制证据。活动session运行目录优先，补齐视觉模式的front/right_exit/hard_safe原始日志字段。

## 验证与加载

- 44项弯道/视觉控制/会话针对性检查通过，6项监控profile检查通过；修改的Python语法检查通过。没有跑全部回归或新电机测试。
- `replay-windows.py` 仅CPU读取记录帧、掩膜、时间/序号、IMU和最终执行否决，无GPU、设备或云端。每个窗口新建controller，所以是局部原因对照，并非整轮逐时刻完全复现。
- straight500:540：有效40/40，before/after动作完全一致（34stop、6forward）。corner1200:1280：路径有效14/80→79/80，pivot-right2→13。全停重启480:540：有效0/60→60/60，pivot-right0→8。新增指令是离线假设，不能当成真实位移或通过弯道。
- Firefox模拟网页验证：实际脚本可解析，Start/End/heartbeat/离页停止正常，图像请求间隔中位101ms，坏JPEG保留上一完整图。原子JPEG并发60写/91读，解码失败0。独立只读审查未发现重要或阻断缺陷。
- 沿用用户本次实车迭代的加载授权，HTTP先确认running=false、stopped_verified=true后重载按钮manager、dashboard和静止感知monitor。未点击Start、未发送运动指令。重载后active PID：panel402365、dashboard402477、monitor402478；sensors340847和serial365261保持运行。
- 实际加载和只读HTTP/图像检查见 `deployment.json`、`live-readonly-check.json`。下一步是用户点击Start进行实车验证，助手根据新session分析；本轮没有新物理过弯成功证据。

## 精确产物

`outputs/visual_feedback/20261010/corner-dashboard-fix/` 保存 before/after 源码、SHA256 manifests、仅本轮增量patch、原始输入选段hash、统计、逐帧before/after命令、派生掩膜/回放图、浏览器证据、并发JPEG结果与部署记录。原始运行数据保持原样。

## 重启后的按钮恢复与绿色道路复查

- 后续用户询问开始按钮变灰。只读核对发现新boot `e3235f44-aad1-42ba-8cca-a379caa099fe` 与上一轮 `f48a60e0-9a6d-4a9a-b29d-3c4262820e2b` 不同，uptime约195s，临时用户unit LoadState=not-found，旧socket仍存在；HTTP controls unavailable。此前的已加载状态不能当作重启后的运行事实。
- 按当前用户实车启动安排重新执行原 systemd-run --user 临时supervisor加载命令，仅恢复按钮等待点击，没有Start POST或电机指令。恢复后HTTP状态和传感器/图像证据见 outputs/visual_feedback/20261010/corner-dashboard-fix/reboot-panel-restored.json。仍是本次临时授权，未自动扩展为开机永久实车模式。
- 同次复查 monitor CSV末1000行全部fusion=yolopv2、estimate_reason=ok，连续25次HTTP样本均模型道路有效。因此这段观察未复现用户报告的绿色完全消失；不能据此称跳变已经解决，或断言当前消失由结果过期导致。绿色显示的是当前选出的road-minus-lane有效走廊，而蓝色轮廓是原始YOLO道路；任一失效条件触发时走廊会清零，这是代码行为。准确短采样与JPEG绿色覆盖统计见 green-monitor-check-*、green-monitor-log-summary.json 和 reboot-panel-restored.json。
