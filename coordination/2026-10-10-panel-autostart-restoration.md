# 2026-10-10 启停恢复与开机待机部署

- 用户当次指令：“帮我恢复启停程序……最好是开机自动启动”，并询问当前是否支持车上实验。授权范围为恢复/持久化启停程序；未授权助手点击开始或实车运动。
- 时间：2026-10-10 21:56（Asia/Shanghai）；HEAD `bd6fe9c88f6c441eda2f4abbac14054ba3e9570b`、main；本次未提交。开始时已有本聊天的PROJECT_INDEX修改与screen-panel-sensors诊断未跟踪文件，均保留。
- boot ID `611755af-c0db-476f-90c6-969e86780eac`。没有新实验run ID或驾驶session。

## 部署

新增仓库 `deploy/vehicle-autodrive-panel.service`，安装至 `/etc/systemd/system/vehicle-autodrive-panel.service`；两者SHA-256相同：`b54f6df2c0340c1d3b7296aab23be27d6e6f320c2ef3c3d0b9a41192aef3b2ee`。

系统级服务以jetson用户执行现有 `bash run_autodrive_panel.sh --authorize-motion I_UNDERSTAND_MOTORS_WILL_MOVE`。这只恢复实车按钮的待机程序，不打开相机/串口或恢复旧驾驶会话；开机/进程恢复均为idle，Start由用户操作。保留现有总时限0、3秒网页租约、记录/时效/取消门禁。Restart=on-failure，KillSignal=SIGINT、KillMode=mixed、TimeoutStopSec=30，先由supervisor通知运行程序停止。未设置NoNewPrivileges，以保持现有实验wrapper通过sudo协调监控服务的能力。

部署前检查系统级和用户级同名服务均不存在、HTTP无活动驾驶会话；安装命令拒绝覆盖不同内容的已有unit。实际执行：

```bash
sudo -n install -m 0644 deploy/vehicle-autodrive-panel.service /etc/systemd/system/vehicle-autodrive-panel.service
sudo -n systemctl daemon-reload
sudo -n systemctl enable --now vehicle-autodrive-panel.service
```

系统级multi-user.target.wants链接已建立，不依赖用户登录或linger。旧临时用户unit不存在，未同时启动第二个supervisor。网页、相机、串口、雷达/深度服务未主动重启；没有修改功耗或安装依赖。

## 验收与实车条件

- `bash -n run_autodrive_panel.sh`、`git diff --check`通过；宿主 `systemd-analyze verify deploy/vehicle-autodrive-panel.service`退出0。现有OLED权限、snapd旧键和varlink占用警告与本次unit无关，未修改这些服务。初始校验发现现有panel脚本无可执行位，unit改为显式/bin/bash运行，最终校验无本unit错误。
- `systemctl is-enabled`=enabled、`is-active`=active，MainPID18904；`journalctl`显示Panel session supervisor ready; mode=hardware。`ps --ppid 18904`无子进程。当前内核不提供/proc的children字段，验收改为ps查询；未将该缺字段当成服务故障。
- 21:56:44 `GET /api/state`：available=true、state=idle、running=false、mode=hardware、session_id=null、runtime_limit_seconds=0。网页Start禁用条件已不成立，已恢复按钮就绪；未发任何Start/End/heartbeat请求，没有代按按钮。未实际重启整车，只能称开机自动启动配置完成、当前服务已验证，不能称重启验证已通过。
- 此时 `/dev/myserial`仍不存在，lsusb无CH340；电量/IMU/姿态/编码器无当前值（busy/stale随重试时刻变化）。相机、雷达和深度online。电池当前电压无法读取，无法远程确认电量。
- 当前支持现有静止感知观察；不具备底盘行驶实验条件。必须先现场恢复控制板供电和USB枚举，再确认myserial、当前电压及IMU/编码器健康。HDMI异常不直接等于CUDA不可用，但其供电/驱动故障原因仍未排除；不能仅凭按钮可用判定整车就绪。

精简原始部署验收：`outputs/diagnostics/20261010-panel-autostart/deployment-verification.json`。更新用法于`docs/vehicle-dashboard.md`、`docs/visual-feedback-replay.md`。本轮无硬件运动、GPU压力测试、原始数据改写或向其他聊天发信。下一负责人仍本车端聊天与现场用户：恢复供电/USB后只读复核，再依用户当次实验安排执行。

后续22:08用户自行重新插拔/开机：boot变为`29762023-526d-4076-af3b-007244756332`，助手未手动加载，systemctl panel active、HTTP hardware/idle/available=true/running=false/session_id=null。因此现在增加一次真实重启后自动待机恢复证据；底盘串口仍缺失，行驶实验条件尚未满足。新boot电压采样与局限见screen-panel-sensors诊断补节。
