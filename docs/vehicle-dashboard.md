# Jetson 小车实时面板

复用现有 LCC 状态与图像发布。综合页 `/`，状态接口 `GET /api/state`，健康接口 `GET /api/health`。同一局域网访问 `http://小车IP:8080/`，浏览器自动刷新，不依赖公网资源。DHCP地址改变后用 `hostname -I` 查询。

2026-10-08 按用户要求移除 LCC 详细诊断页及旧 Web 启停代码；旧 `/diagnostics` 地址返回404。2026-10-10 按新要求添加会话式“开始/结束自动驾驶”，通过独立本机启停程序接入，当前行为见[运行说明](operations/runtime.md)，离线分析见[记录与回放](outer-loop-record-replay.md)。2026-10-10已按用户当次安排重载网页/传感器并启动启停程序，该句记录的是当日部署状态，本轮未重启或验收服务。当前统一入口为 `./vehicle dashboard`，HTTP启动本身不启动采集或驾驶。已有开机服务运行时直接访问页面，无需重复启动。手动运行可用 `--host`、`--port`、`--runtime-dir`、`--sensor-dir` 参数；`CAR_PYTHON` 可指定Python，`DASHBOARD_HOST`/`DASHBOARD_PORT` 可指定默认监听地址。

## 开机服务

| 服务 | 作用 |
|---|---|
| vehicle-dashboard | HTTP不连接硬件；启停程序可用时接受三个明确的会话POST意图，其他修改拒绝 |
| vehicle-perception-monitor | RGB + YOLOPv2 CUDA FP16 + 虚拟控制；强制关闭电机、云台初始化、云端请求和视频归档。每30分钟退出重启，限制工作日志规模 |
| vehicle-serial-monitor | 独占底盘串口被动接收电压、IMU、姿态、编码器，不发送任何串口帧 |
| vehicle-sensors | 已有ROS Foxy镜像的雷达/深度节点；容器不挂载底盘串口，不启动导航/运动节点 |
| vehicle-autodrive-panel | 2026-10-10按用户要求持久部署；开机只启动启停程序进入实车待机，用户点击开始才运行硬件 |

安装脚本 `deploy/install-vehicle-dashboard.sh` 校验并安装四个新服务，只接受不存在或内容完全相同的已有unit。安装启用与启动分开。设备空闲时：

```bash
sudo systemctl start vehicle-dashboard vehicle-perception-monitor vehicle-serial-monitor vehicle-sensors
systemctl is-active vehicle-dashboard vehicle-perception-monitor vehicle-serial-monitor vehicle-sensors
journalctl -u vehicle-dashboard -u vehicle-perception-monitor -u vehicle-serial-monitor -u vehicle-sensors --since '10 minutes ago'
```

开机运行的是监控；屏幕上的PWM/转向是提案，不表示车轮运动。新增按钮需要独立启停程序，默认预演；实车启用按当次安排。结束按钮提交停车请求，现场仍可使用当前运行程序与物理急停。

用户于2026-10-10进一步要求恢复实车按钮并开机自动启动，已新增持久系统级 `vehicle-autodrive-panel.service`（源码 `deploy/vehicle-autodrive-panel.service`），以jetson用户运行并显式启用后续按钮的实车模式。该服务开机、进程重启均从idle开始，不恢复旧驾驶会话、不打开硬件。手动终端入口仍默认预演。该unit单独部署，不由上面的四服务安装脚本自动启用。检查用 `systemctl is-enabled vehicle-autodrive-panel`、`systemctl is-active vehicle-autodrive-panel` 和 `GET /api/state`；旧临时unit使用的 `systemctl --user` 不适用于这个系统级服务。开机配置验收不等于已经通过真实重启或实车实验；以当次状态记录为准。

## 进入实验

相机与串口不能被监控和控制进程重复占用。新增入口：

```bash
./vehicle drive --max-runtime-seconds 30
```

默认电机禁用。入口暂停相机监控；仅参数含 `--enable-motors` 时才暂停独立串口监控。原现场确认、标定与电机门禁仍生效；真实运动按当次现场安排执行。退出后恢复本次暂停且之前active的服务，保留实验程序退出码。

真实控制中的遥测借用已有底盘会话。读取跳过忙碌的控制锁和空缓冲；HTTP、JPEG、磁盘读取不进入控制循环。其它工具直接占用相机/串口前，先停对应监控；操作雷达/深度前检查 `vehicle-sensors`。不要因面板有数据而以为另一个程序已经获得设备。

## 状态含义

- 电池同时显示实测电压和估算百分比。按用户于2026-10-09提供的参考值：10 V为0%，12 V为100%，中间线性换算，结果限制在0–100%。例如11 V为50%、11.4 V为70%；这是电压估算，不是电池管理板实测SOC。缺失、异常、过期或已停止的数据不提供当前百分比，网页断线时清空电量卡片。接口`/api/state`中的`sensors.battery.value`提供`percentage`、`percentage_estimated`和`empty_voltage_v`/`full_voltage_v`；该换算仅用于显示，不改变车控停止阈值。
- 有效值、采样年龄与当前boot心跳共同决定在线状态。过期、坏数据、未接入、占用和已停止分别显示。节点存在不代表正常采集。
- LCC置信度、目标检测置信度、云端置信度分开。云端契约没有数值置信度，不制造分数。模型结果过期即隐藏实时目标，控制循环心跳不能替代模型新鲜度。
- 最新 `cloud_arbitration` 可显示触发/等待/保持停车/恢复候选及车端仲裁；按当前run/event读取已归档场景、风险、建议和耗时。2026-10-11已兼容恢复状态、带连字符UUID和`road-recovery-v1`归档结构，显示语义判断、选中候选、危险、不确定性，以及实际执行来源（本地/云建议短步/保持停车）、执行候选和拒绝理由。HTTP自身不请求云端、不读 `.env`，不返回原始响应、request_config或密钥。待机监控强制关闭云端，页面此时显示“未启用”是监控配置；点击开始后的驾驶子进程按visual_feedback profile启用云恢复。
- 最终目的地、路线约束与局部前视点分开。没有定位/全局导航发布时显示未接入；填写一个目的地字段不会实现导航。
- 2026-10-11 c8a60d2b复盘后，持续黄线/语义安全停车也可发起仅观察请求：图像和系统门禁仍须有效，候选为空，回复不能解除运动否决。面板区分未请求、等待回复、已回复和实际执行来源；请求用途显示“仅观察分析”或“本地候选复核”。同一停滞仍最多3次尝试。`diagnostic_only/response_status`为新增运行字段；旧面板后端可由事件stop_code与响应文件显示主要状态，前端刷新即可更新，完整字段在后端下次加载时生效。
- 雷达/深度当前用于展示；这不表示它们已进入驾驶避障链路。
- 2026-10-11 05a3修正后，`event_id`只表示当前事件；`last_event_id`独立保留最近事件，取消后仍可读取对应归档回复。页面显示最近处理原因，并区分等待、取消、收到但未采用、短步获准后停止。`step_authorized`仅表示车端批准短步，实际轮速和最终否决仍以applications日志为准。历史显示不能重新授予行驶权限。

## 发布协议

默认 `outputs/vehicle_dashboard/sensors/serial.json`、`ros.json`、可选`camera.json`。`publish_snapshot(path,sensors)`原子写入boot ID、单调时钟、发布时间、PID。每个传感器含`status/value/age_s/reason`，其中age_s是原采样年龄。HTTP额外加入发布后的经过时间，默认3秒截止；坏值不能计为在线。

运行时沿用`status.json`和`latest.jpg/latest_birdeye.jpg`。旧格式结合文件时间、主机启动时间、`termination.stopped`判断实时性，只说明数据到达，不证明硬件动作已确认。默认观察监控目录、`outputs/onboard_runtime/rosmaster_jetson_yolopv2`及`outputs/onboard_runtime`，取最新状态。其它输出目录可添加重复的`--runtime-dir <目录>`参数。

未来导航可发布：

```json
{"navigation":{"status":"online","task":"沿外圈巡航","destination":null,"route_constraint":"outer-loop","local_target":null,"frame_id":"camera"}}
```

## 验证边界

验收包括单元与HTTP、真实Firefox渲染和断线、静止传感器/GPU、服务enabled/active与LAN HTTP。未实际重启只能称“已配置开机启动”。不据此宣称避障、停车距离、定位或目的地导航已验收。准确版本、数据和命令见`coordination/2026-10-08-vehicle-dashboard.md`。
