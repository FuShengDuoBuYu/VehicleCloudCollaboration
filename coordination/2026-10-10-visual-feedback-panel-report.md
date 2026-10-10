# 实时视觉小步控制、面板启停及离线回放

日期：2026-10-10。车端 `/home/jetson/VehicleCloudCollaboration`，HEAD `fce828b89b084eda293e83b99c8861f522858d8e`、main。既有修改及未跟踪文件全部保留，无提交、推送、安装或由助手启动的新增底盘试运行；末尾用户明确授权后已重载两个服务。使用既有 `outputs/field_yolopv2/venv/bin/python`。

## 用户需求与实际状态

用户明确要求：转弯应每次少转，根据实时YOLO决定下一步前进或再旋转；页面手动开始/结束，每次记录传感器；回放主要是后续离线复现、模拟和修bug，减少每次都跑车。最新要求优先于历史固定90°及助手看图监督流程。

本轮完成代码、软件验证、历史实车数据的派生回放。新逻辑随后按用户“现在立刻加载，我要先实车跑一次看看效果”的当次授权部署。启停程序已启用实车模式并待机，未由助手点击开始。首弯、完整外圈与物理停住仍未验证。用户另要求减少详尽测试，后续以逐run数据分析和针对性回放/实车效果为主。

## 根因及改动

只读复核旧 `stationary_corner.py`：旋转累积IMU相对角达到固定90°才退出，视觉主要起否决作用；10°、20°、30°时即使出口已对齐仍继续 pivot。历史 `live04` 记录在2026-10-10既有运行中实际完成约92.8°的软件偏航累积；此前续测报告仅写live01且仍标执行中，不能当作完整历史账本。本轮没有改写旧原始数据或伪造该报告后续结果。

- 新 `VisualFeedbackController` 没有整弯目标角/入弯距离；当前YOLO局部路径决定下一步，当前意图改变或证据失败立即撤销。保持至少5个不同新观测的启动门禁。
- 每步最多0.15秒，pivot还有5°相对偏航限制；零输出等待0.7秒后，必须取得等待期结束后采集的新帧，再决定下一步。它们是软件上限，不是固定转弯动作；惯性和实际车身停稳未保证。独立 `MotionGuard` 查时限和会话停止请求，driver会话取消锁存，迟来的非零提案不能解除。
- 新 profile `car/control/vehicle_control/profiles/rosmaster_jetson_yolopv2_visual_feedback.yaml`：320 CUDA FP16完整三个YOLO头，现有30 PWM上限，右转偏航符号−1；不使用旧固定道路宽度裁剪，右出口目标由当前道路像素选择，关闭盲转/经典备用/云请求。`motion_calibrated=false`保留，不把候选写成已实测标定。
- 新页面按钮向独立本机Unix-socket supervisor提交固定 start/stop/heartbeat。默认预演模式，启用实车需现场终端显式参数。HTTP只接受明确POST意图；跨Origin和PUT/DELETE拒绝；前端不提交PWM/配置/任意命令。现有HTTP服务不提高权限。结束通知与归档封口分开；只有程序终止且记录了停止成功才显示软件停止确认。
- 每次会话保存命令、console、运行结束状态和完整配置。共享串口记录RX/TX原始字节，逐周期保存完整遥测；已有ROS采集者按run ID存全量收到的LaserScan与原始深度字节/元数据。异步有界归档，错误/丢失保持并撤销运动许可；新实车profile必须等当前run雷达/深度及UART原始记录就绪，未更新采集服务时不会放行。
- 记录停止后发布inactive及closed；不再误显示仍采集中。当前雷达/深度用于采集和记录健康检查，不宣称已经实现传感器避障。

## 回放工具与证据

`car/autodrive/tools/replay_session.py` 默认使用当次完整配置快照；显式candidate profile用于比较。旧工具原先固定 `left` 路线提示，已复现并修复为读取配置。无模型、相机、串口、ROS订阅、云端或实体底盘；控制器使用归档模型输入/输出、消费序号、年龄、逐row IMU及末端否决。缺失/未封口/丢失数据不默认为完整。

输出包含逐row候选指令、与历史动作的差异、按控制时刻之前样本对齐的全量传感器索引、原文件hash、当前回放代码hash和完整性。没有的历史字段保持未知。派生输出不能覆盖原目录；原始文件只读。运行时仍需要ROS原始目录与archive一起保存/复制。

实际命令：

```bash
outputs/field_yolopv2/venv/bin/python -B car/autodrive/tools/replay_session.py \
  --run outputs/corner_fix/20261010/live04/runtime/runs/20261010_004747_946146_hardware \
  --profile rosmaster_jetson_yolopv2_visual_feedback --allow-incomplete-sensors \
  --output outputs/visual_feedback/20261010/live04-verified-replay
```

461个控制周期、395个独立无损输入；候选445 stop、8 forward、8 pivot-right，最大虚拟PWM30，93行动作不同。主要停车原因：当前视觉门禁314、等待停稳后新帧73、历史末端时效否决48。未提供独立行为标注，因此不是93项bug修复或成功率。历史缺完整雷达/深度归档，manifest明确 `all_raw_streams_complete=false`、`historical_partial_replay=true`。

准确输入CSV为归档副本，SHA-256 `8a5b39618000f2e44abeec03c46e6ea016ee69f034da504ce5aec4f8e19a92d5`；根目录原CSV是另一个文件，SHA-256 `a484b2580af978f840f55e34b0e1eb8c0641e6c813b2631f4ab607702a44fbfa`，不能混为同一hash。全量原始与代码hash见回放 `manifest.json`。

记录回放只能验证在同一组已观察输入下控制如何改变。改动动作后车辆实际姿态与新画面不会从旧录像中生成；尚无已标定车辆动力学、米制轨迹和相机外参，不称为完整物理仿真。

## 验证

- 先失败后通过：新视觉短步/过期缓存/意图改变/IMU/独立停止、传感器无损保存、会话启停/租约、冻结配置/完整传感器索引/越界拒绝、整段无设备回放。代码复核新增用例也复现PUT误接受、ROS结束状态仍active及replay路线提示不一致，再修复通过。
- 27组相关CPU/模拟硬件/回环HTTP回归：341项通过，命令和结果记录 `outputs/visual_feedback/20261010/final-verification.json`。最初system Python缺torch，未安装依赖，改用现有venv后通过；该环境错误不写成产品故障。
- Firefox实浏览器只访问临时本机模拟面板：按钮开始/结束、持续心跳、离开页面后租约停止、dashboard/replay脚本解析、手机宽度无溢出通过。运行child仅写假状态并sleep；不访问真实8080面板，不创建底盘。证据 `browser-final/browser-validation.json`、`panel-synthetic.png`，脚本 `browser_check.py`。
- 当前代码21文件静态语法、3个shell入口语法和当前候选的两项配置门禁进行只读校验；详细命令见final-verification。软件通过不等于实车闭环完成。

任务源码快照/校验表/增量patch保存在 `outputs/visual_feedback/20261010/final-source/`；最初13文件备份在 `before/`。增量patch只覆盖有本阶段准确before副本的文件，不能代替所有pre-existing dirty变化；其它任务文件提供最终源码hash和快照。未在本轮开始前备份的既有未跟踪文件，不把Git HEAD当成其准确before。

## YOLOPv2与SLAM

官方YOLOPv2是道路/车道/目标的感知网络，训练/评测来自BDD100K，不是自动输出方向盘/PWM的完整驾驶系统：[YOLOPv2官方仓库](https://github.com/CAIC-AD/YOLOPv2)。本项目实际使用它做视觉输入，再由本地控制和门禁执行。结合现有架构判断：封闭场地沿当前道路行驶可先完成视觉闭环，不要求先加SLAM；若要“到地图上某个位置”、全局路线或位姿轨迹，应另接定位/地图。Nav2对此提供SLAM或已有地图定位两种链路：[Nav2 Mapping and Localization](https://docs.nav2.org/rolling/configuration_and_development/first_time_robot_setup_guide/sensors/mapping_localization/)。这是架构选择，不是已完成导航。

本场地黄线/绿岛及近车道路漏分仍是已观察问题；SLAM不会直接修复语义误分。当前视觉候选宁可停住，尚不能保证连续完成整圈；模型适配与实车惯性检查仍需独立证据。

## 下一责任与部署窗口

本车端聊天继续负责按运行数据调整；用户现在可在已加载页面启动一次短段/单弯，观察车身实际转动与停止，并点击结束保存。按AGENTS.md“未经本次实验授权……不重启服务”，本轮没有把历史实跑授权自动扩展；随后收到当次明确部署/实跑安排，已加载且仅待按钮点击。

现场下一项不是重新跑完整圈：先检验新小步是否让车身确实少量转动/前进、停止后惯性如何、当前YOLO是否能决定继续转或前进，按次保存原始数据，失败先离线回归。代码、模型、配置冻结后再做正式实车验收。本报告未发给其它聊天。

## 最终独立复核与部署

独立复核发现并修复：驱动因截止/会话取消返回零时，必须写最终veto及原因，不能让日志和回放保留允许前进；UART须有0.5秒内接收且采集线程健康，不用旧累计计数放行。3项针对用例先失败后通过，独立复核3项通过。随后补充停止等待以实际否决时刻为起点，并让回放消费该时刻；32项相关纯CPU回归通过。341项全回归是在该最后时间锚修正前通过，追加完整回归因自动审批超时未执行；按用户要求不再扩展，保留这些验证层次。

用户“请你现在立刻加载，我要先实车跑一次看看效果”授权后执行 `sudo -n systemctl restart vehicle-dashboard.service vehicle-sensors.service`，退出0；随后 `systemd-run --user --unit=vehicle-autodrive-panel --collect --property=WorkingDirectory=/home/jetson/VehicleCloudCollaboration --property=TimeoutStopSec=20 --property=KillMode=mixed /usr/bin/bash /home/jetson/VehicleCloudCollaboration/run_autodrive_panel.sh --authorize-motion I_UNDERSTAND_MOTORS_WILL_MOVE`。此临时用户服务不配置开机自动启用实车；当前hardware/idle/running=false。没有发送Start POST或电机命令。

只读验收GET首页/API state通过，开始/结束DOM已加载，7类传感器online，电压11.4V/估算70%。新网页PID340846、传感器服务PID340847；相机监控PID288486和被动串口PID1167前后不变。页面 `http://10.20.30.132:8080/`，用户刷新后自行开始/结束。所有离线回放已结束；未留下测试或后台分析与本次实跑并行。部署证据在final-verification.json。
