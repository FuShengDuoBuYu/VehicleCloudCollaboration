# 2026-10-10 22:23–22:24 整体只读诊断

## 范围与当前条件

- 人类请求：重新整体诊断。用户确认期间没有再次充电，充电器已拔，当前仅电池供电。因此本轮不是充满后的复测。
- 主机jetson-desktop-car-2，HEAD `bd6fe9c88f6c441eda2f4abbac14054ba3e9570b`，main。boot `29762023-526d-4076-af3b-007244756332`，检查开始约开机17分钟。时间按Asia/Shanghai记录，宿主显示10:23–10:24 EDT。
- 保留本聊天已有PROJECT_INDEX、两份docs、两份coordination报告和新panel unit改动；无新提交，无实验run ID。继承前轮已经核验的真实传感器/串口身份，不从照片空口推断漏线。
- 只读systemd、USB sysfs、现有HTTP状态、日志、驱动规则、功耗查询和现有INA3221/温度节点。未另开采集/推理、发送串口帧、操作电机、关机重启、切换功耗、重新绑定USB或重启服务。

## 故障分层与证据

| 部分 | 本轮结果 | 判断与限制 |
| --- | --- | --- |
| 系统基础 | 可用内存1973MiB，swap仅3MiB使用，根盘剩39GiB/使用66%；本boot筛选没有OOM、EXT4错误或明确NVMe/I/O错误 | 未见资源耗尽导致整机失常的证据；不是完整磁盘硬件检测 |
| 供电 | 10次Jetson VDD_IN为10736/10744mV，即10.736–10.744V；之前同boot约10.9V，后10.79V | 当前Jetson测量未低于9.6V；不能替代电池/底盘控制板/显示屏电源测量，不代表电量百分比或覆盖启动、电机瞬时压降。充电是否完成仍未排除 |
| 功耗/GPU | nvpmodel报告20W/id3；开机oneshot正常退出0。现有模型3秒completed_frames14173→14216、error null，约69.35/70.58ms，结果年龄0.178/0.146s；GPU56.187℃、CPU57.125℃ | GPU现有计算链路可用。不能由此前切10W或黑屏判显卡损坏，也不能证明显示输出硬件完好。前轮TPC掩码差异未解释/未改动 |
| 显示 | DP-1 disconnected；xset显示Monitor On、屏保timeout0、DPMS超时0。GNOME当前active/running、NRestarts6，早先六次启动超时后重试，631秒进入running。Xorg显示此前模式在NULL/1920×1080间反复切换 | 当前不是仅“桌面还没启动”或普通自动息屏；显示链路仍未连接。屏电源、DP转HDMI/线、显示驱动均未独立隔离；GNOME已运行不等于屏已恢复 |
| 底盘USB | lsusb及sysfs无CH340 1a86:7523、无myserial；ch341模块文件/匹配alias及udev规则仍在。串口监控exit2，每约10秒重试，计数95→97 | 电量/IMU/姿态/编码器缺失的直接原因是底盘USB设备没被系统识别，不是仅网页显示错误或别名丢失。故障可位于板端供电、USB通信路径/单端口、桥接芯片等；未确定具体器件 |
| 其它USB及遥测 | VIA USB2 HUB、下级HUB、相机、深度及雷达均authorized1/runtime active；雷达CP210x ttyUSB0。HTTP相机age约0.10–0.15s、雷达约10Hz/age0.32–0.34s、深度约28Hz/age0.27–0.32s | 不是整个USB系统不可用。空闲USB3 HUB/root runtime suspended，不直接视作故障；CH340是独立缺失，不以雷达串口替代 |
| 网页启停 | dashboard/sensors/perception/panel active；panel enabled、hardware/idle、available true、running false、session null | 已恢复的开机启停程序仍有效；没有新的驾驶会话。当前底盘通信及必要遥测缺失，不能进行正常闭环行驶实验 |

内核筛选中`Finished tegra23x overcurrent event probing`是驱动初始化语句，不能将其中overcurrent字样当作已经发生过流。nvpmodel为oneshot，因此执行完成后的inactive/dead配合Result=success并非功耗服务失败。`systemctl --failed`为空也不能掩盖serial-monitor处于自动重试。

## 命令与采样依据

- `git status --short`、`git rev-parse HEAD`，`date -Iseconds`、boot ID、`uptime`、`free -m`、`df -h / /tmp`。
- `systemctl show`七个相关unit的ActiveState/SubState/Result/NRestarts/ExecMainStatus；`systemctl --user show gnome-shell-x11.service`及`systemctl is-enabled vehicle-autodrive-panel.service`。
- `lsusb`、`lsusb -t`、设备节点、`modinfo ch341`及`/etc/udev/rules.d/usb.rules`；sudo Python仅read_text读取USB设备authorized/power/control/runtime_status/autosuspend等现有属性。
- `sudo -n /usr/sbin/nvpmodel -q`、`journalctl -b -u nvpmodel`及unit/启动脚本读取。sudo Python仅读取`/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input`10次（0.2秒间隔）和thermal_zone属性。十次mV：10736,10744,10744,10736,10736,10736,10736,10744,10744,10736。
- 只GET `http://127.0.0.1:8080/api/state`两次间隔2秒验证遥测；另外两次间隔3秒核实现有`runtime.diagnostics.yolopv2_fusion`计数和错误，无POST。前一对输出的顶层model/perception不存在，不据null误判模型停机；有效指标在runtime.diagnostics。
- 现有X会话只读`xrandr --query`/`xset q`，system/kernel journal有界筛选；未读取崩溃转储内的环境秘密，也未把既有无关python crash当作桌面崩溃证据。

## 结论与下一步

已确认两个尚未恢复的故障边界：底盘USB设备没有枚举；显示接口当前断开且本boot有桌面启动异常。它们是否由低压/供电引起尚未证实。当前计算/网页/感知仍工作，不能判整机或GPU已损坏；同样不能保证充满即恢复。

1. 先由现场正常关机、主电源OFF，使用配套充电器充至绿灯，拔充电器后保持原接线开机。依据前轮已核实的Rosmaster官方充电说明，当前用户未完成此对照。
2. 同一轮先查CH340/myserial、四路新鲜遥测和显示连接。若两者均恢复，只能支持供电/重新上电相关，不能倒推唯一器件原因。
3. 若仍失败，对屏幕使用确认正常的显示器/线或独立供电做单变量对照；对底盘先核实准确PCB版本与原通信线两端，再用已知良好数据连接绕过HUB，必要时独立主机只查枚举。只有这些对照才能区分板端与Jetson/线路问题，不继续全车混合插拔。

当前可做静止的视觉/雷达/深度观察，正常闭环行驶实验尚不具备条件。本轮只读诊断完成，硬件恢复未完成。
