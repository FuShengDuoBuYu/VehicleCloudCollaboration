# 2026-10-10 HDMI黑屏、启停按钮与底盘遥测只读诊断

- 时间：2026-10-10约21:46–21:48（Asia/Shanghai）。用户确认实体屏幕为HDMI大屏；本次仅诊断，没有实验/运动或服务修改授权。
- 版本：HEAD `bd6fe9c88f6c441eda2f4abbac14054ba3e9570b`，分支main，诊断开始时`git status --short`为空。此版本已不同于旧索引中fce828b；本次未提交或修改业务源码。
- 当前boot ID `3a1b2487-f1c2-4d93-89bb-7b09d18e5523`，最后采样`/proc/uptime`为188.26秒。系统墙钟在启动后有调整，不能以systemctl显示的“46min ago”推算本次运行时长。无新实验run ID。

## 已确认状态与原因

1. HDMI：实际X桌面`:0`的`xrandr --query/--verbose`显示DP-1连接到LTM LONTIUM，列出1920×1080等模式，但没有活动输出几何/CRTC或当前模式标记。`/var/log/Xorg.0.log`截至单调时间188.206秒已有747条DFP-1连接状态日志，最近连续约0.2秒交替connected/disconnected。`xset q`显示DPMS Enabled但Monitor is On，`display-manager`/gdm active。证据支持显示链路反复掉线、未稳定启用输出；线缆、转接器、显示器供电或驱动原因尚未区分，不能断言硬件损坏。OLED服务虽active，与用户所指HDMI屏幕不同。
2. 网页按钮：`systemctl --user show vehicle-autodrive-panel.service`返回LoadState=not-found、ActiveState=inactive。旧`outputs/vehicle_dashboard/control.sock`仍存在但没有可用supervisor；`GET /api/state`返回drive_control.available=false、state=unavailable、running=false。页面源码据available禁用Start。历史部署为临时用户unit，重启不会恢复；本轮boot已改变，因此旧“hardware/idle”记录不能代表当前状态。
3. 底盘数据：`/dev/myserial`不存在，`lsusb`没有CH340 `1a86:7523`，仅`/dev/ttyUSB0`/CP210x `10c4:ea60`对应雷达。现有udev规则仍匹配`1a86:7523`创建myserial，因此不是只丢了设备别名。`vehicle-serial-monitor`以exit2反复自动重试；其源码捕获串口打开异常并发布“串口不可用或由控制器占用”。串口设备未枚举是本次已确认的故障边界；控制板供电/USB连接/硬件原因需现场检查，不能从缺设备直接确定是哪一种。
4. `GET /api/state`约21:46:33：camera/lidar/depth online；battery/imu/attitude/encoder stale且无当前值。battery.device_present=false。四类缺失数据均由同一底盘UART发布，相机/雷达/深度服务active。监控runtime.wheel_driver.mode=dry-run，其PWM为虚拟提案。

## 实际只读验证方法

- `git status --short`、`git rev-parse HEAD`、`git log -5 --oneline`及相关源码/历史部署文档读取。
- 宿主`systemctl status vehicle-dashboard vehicle-sensors vehicle-serial-monitor vehicle-perception-monitor --no-pager -l`；用户级panel的status/show；`journalctl -b -u vehicle-serial-monitor`；boot ID和uptime。
- 宿主`lsusb`、`ls -l /dev/myserial /dev/rplidar /dev/video* /dev/serial/by-id`、udev规则和内核日志读取。不打开底盘设备。
- 仅GET `http://127.0.0.1:8080/api/state`，按字段汇总状态，没有任何Start/End/heartbeat POST。
- 从现有桌面进程仅取DISPLAY/XAUTHORITY路径，执行`xrandr --query`、`xrandr --verbose`和`xset q`；只读统计Xorg连接状态日志。没有xrandr设置、显示服务重启或桌面配置改动。
- 沙箱不能访问宿主systemd/netlink，已通过获批的宿主只读命令复核；不把初始沙箱失败作为设备异常。

## 下一步

现场先检查底盘控制板供电及Jetson到控制板的USB连接，并检查HDMI屏供电、线缆和转接器。重新枚举后可只读确认myserial/CH340及四路遥测恢复。恢复启停supervisor需要用户明确授权加载；实车模式仍需当次安排，助手不代按Start。HDMI链路稳定后再决定是否需要重新启用输出；当前没有执行任何恢复操作。

本次只写诊断记录和索引，未安装、重启服务、打开新硬件采集、发送车控指令、运行测试或改写原始实验数据。

## 21:52–21:53 GPU及功耗补查

- 用户起初补充此前尝试20W→10W、收到重启提示，当时电池已没电。后续更准确叙述见文末：并未立即执行提示重启，而是继续使用至低压蜂鸣后关机。此为用户现场信息，不等于已测得电压或已确认黑屏因果。
- 本次boot已再次变化：`611755af-c0db-476f-90c6-969e86780eac`，初始`/proc/uptime`56.66秒；助手未触发重启。journal仅保留当前boot，无法由其还原上一次断电经过。
- 普通`nvpmodel -q`报告20W/id3但有sysfs权限错误；进一步`sudo -n /usr/sbin/nvpmodel -q --verbose`只读复核仍报告20W/id3，持久状态`/var/lib/nvpmodel/status`为`pmode:0003`。6CPU在线及GPU最高408MHz与该配置相符，但TPC_PG_MASK实际15/配置240不一致，原因未核验，不将模式名当成全部底层设置都正确的证明。nvpmodel开机unit退出0。
- 现有HTTP模型两次查询completed_frames159→675、latest_sequence287→1044、error均null，TorchScript/FP16，推理约75.95/80.08ms，结果年龄约0.195/0.181秒；现有配置来源确认CUDA观察模式。短只读`timeout 3s tegrastats --interval 1000`两次GPU温度54.906/54.843℃，GR3D98%，但频率字段为0，不据其报告真实GPU时钟。未另开推理、压力测试或设备采集。证据仅支持当前GPU计算功能可用，不证明所有GPU/显示硬件完全正常。
- 当前boot内核存在5条`NVRM rpcRmApiControl_dce: NVRM_RPC_DCE: Failed RM ctrl call`；DCE初始化也有Get queue info failed日志。`xrandr --query`当前DP-1 disconnected但保留1920×1080几何，显示状态与前一boot不同。需要将显示驱动/固件通信异常与HDMI线路/供电同时排查，不能仅归为线缆或仅归为GPU损坏。
- 筛选当前boot内核未发现明确欠压/掉电事件记录；该缺失不能排除此前电池耗尽，因为先前boot日志不在当前journal。底盘串口仍缺失，四类遥测stale，相机/雷达/深度online，监控dry-run。
- 核对对应R35.3.1 [NVIDIA官方功耗文档](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)：Orin NX 8GB支持10W/15W/20W；切换涉及GPU TPC掩码时要求重启是正常机制，不能据提示判断硬件故障。本次没有改模式、改驱动或重启。

## 启停恢复后串口连接补查

2026-10-10用户询问如何恢复串口，并认为电池应有电。宿主再次`lsusb`/`lsusb -t`仍无CH340、`ls -l /dev/myserial /dev/ttyUSB*`仅有雷达ttyUSB0；当前boot没有CH340连接日志。`modinfo ch341`确认匹配当前5.10.104-tegra内核的驱动文件存在，USB alias包含1a86:7523；`/etc/udev/rules.d/usb.rules`仍有myserial规则。未加载模块、reload规则或改变串口绑定。该证据不证明驱动实际加载成功，但目前故障早于串口节点/权限层，优先现场检查控制板电源灯、控制板USB线/接口及必要时更换可传数据线，不能把雷达ttyUSB0替代底盘。控制板具体供电接线未核验，不假定电池有电就代表控制板已通电。恢复后通过myserial出现及四路新鲜遥测确认；启停服务当前enabled/active，未点击Start。

## 用户接线照片补充

2026-10-10用户提供三张照片`845be9a9822d90276521bf6f42f5ca52.jpg`、`e9c915d7845eedb7a78793484c777efd.jpg`、`dde548897c89558973570a6e4971ebaf.jpg`，分别位于本聊天附件目录`/home/jetson/.codex/attachments/4bcc7bc1-a64b-485f-8818-6ac3c971ca2d/`、`67d9392e-ad59-44b7-9cbb-c99b36c9ff31/`、`101cc438-5f28-4274-ae24-854308700896/`。可见下层红色指示灯、上层灯光及小OLED显示，但不能由照片测出电池电量或确认底盘MCU/CH340供电健康。第二张带D标识的粗插头为显示连接，不能作为底盘USB通信线；三张照片尚不足以追踪底盘数据线两端。同步宿主只读复核仍无1a86:7523/myserial，panel active。查阅[Yahboom官方扩展板说明](https://www.yahboom.net/public/upload/upload-html/1713173945/1.About%20expansion%20boardV3.0.html)：对应扩展板Micro-USB用于通信，Type-C为供电口，照片中实际版本需进一步核对。未操作硬件、重新安装驱动或改绑串口。

## 22:08–22:09 重新插拔后直接测量输入电压

- 用户报告全部重新插拔仍不恢复，询问是否低于9.6V。新boot `29762023-526d-4076-af3b-007244756332`、采样uptime93.35秒；助手未触发重启。CH340/myserial仍完全未枚举，雷达CP210x在ttyUSB0，当前接到USB拓扑1-2.3。当前boot筛选未看到明确欠压告警、CH340连接或USB枚举失败记录；不等于已排除瞬时掉压/板端故障。
- 找到已运行的INA3221 hwmon：`/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3`。普通权限读属性失败13，改为sudo仅read_text读取现有sysfs，不扫描I2C/写寄存器。首6次VDD_IN为10.888–10.912V；22:09:22再10次为10.872–10.888V，电流同步保存。**测量是Jetson模块VDD_IN，不是底盘控制板或电池端独立电压，不覆盖电机负载或此前瞬时掉压。**只能确认当前静止监控采样中该输入未低于9.6V，不能据此算底盘电量或排除板端供电问题。
- 第二组原始数字/单位/单调时间/boot/source/局限已保存`outputs/diagnostics/20261010-low-voltage/voltage-readonly.json`；未开新GPU任务/车控/电机负载测试。底盘HTTP电压仍null，IMU/姿态/编码器stale，相机/雷达/深度online。
- 核对[对应Rosmaster官方电池说明](https://www.yahboom.net/public/upload/upload-html/1743656067/2.%20Precautions%20for%20battery.html)：低于9.6V对应蜂鸣报警及MCU快速闪灯；文档没有证明该阈值必然使CH340从USB枚举消失。不能引用搜索中其它7.4V型号电池说明替代本车，也不能将Transbot LED规则直接当作Rosmaster规则。
- 用户只观察到最下面Jetson主机板红点持续闪烁，未确认底盘控制板MCU灯或持续蜂鸣。该红点不构成已确认的底盘低压报警证据，不能仅由灯光诊断电量。
- 持久启停服务在本次新boot自动active，HTTP hardware/idle、available=true、running=false、session_id=null，支持真实重启后已恢复待机；与前次仅验证enable配置相比增加一次重启恢复观察，无驾驶动作。下一故障隔离需已知良好数据线/底盘通信口直连与必要时独立电脑枚举对照；若连其它主机也无CH340，才进一步检查板端USB/供电，不直接判板损坏。

## 后续近照与用户纠正：空口不能证明缺线

- 2026-10-10用户进一步提供近照。已直接查看附件`/home/jetson/.codex/attachments/0aa9d875-a300-443c-8549-ff0936cb185e/4ef867da3e23a7b2e4f75b988080f9df.jpg`及`/home/jetson/.codex/attachments/14415c14-5219-48dc-a68f-c19234409f0e/3f8e6646516024d3d5eb56f4bd04bf99.jpg`。铜柱小塔最上层带MCU及白色按键，按键左侧的小USB口没有插线；中层可见电感、电容和电源接头；小塔下层可见VIA芯片、USB接口及散热片，外观支持USB集线器判断。具体PCB版本/全部接口功能尚未由丝印或原理图核实，不将USB集线器与底盘MCU混淆。
- [Yahboom官方扩展板接口说明](https://www.yahboom.net/public/upload/upload-html/1697617458/1.About%20expansion%20board.html)说明该文档版本Micro USB为主机通信接口、Type-C仅供电。助手据空口提出缺数据线判断，但用户明确纠正“我没动过，之前一直都有的”。撤回“照片空口很可能就是缺线原因”的结论：实际PCB版本及原通信物理路径未查实，不能将通用接口文档当作该实物接线证明；板间接线或其它通信入口仅为待核验可能性。
- 用户纠正后复核历史`outputs/hardware_acceptance/20261008/serial-main-verified/summary.json`，被动5秒收到487帧、TX0、健康、四路遥测有效；历史基线报告记录CH340 1a86:7523/myserial。当前宿主`lsusb -t`/`lsusb`仍识别VIA HUB、相机、深度及雷达，但无CH340/myserial；当前boot内核USB筛选没有CH340或明确枚举失败记录。只能定位为底盘USB设备未被主机识别，不能进一步断定线、供电、HUB单口或控制板哪一处故障。
- 保持原接线，先追踪原控制板通信线两端并核实板版本；已知良好USB数据线直连可作为后续单变量隔离方法，不能作为已确认用户漏接的修复步骤。不并联未知通信路径、不发送Start。本轮仅只读及记录，未更改USB绑定/驱动、供电/功耗或服务。

## 用户补充完整断电/充电经过与再次只读采样

- 2026-10-10用户明确顺序：原接线正常使用，屏幕切换NVIDIA到10W收到重启提示但没有立即重启，继续使用至所见9.6V及底盘滴滴蜂鸣，随后关机充电；充电时认为开关在OFF但仍看到灯亮。追问回复充电器一直红灯，没有确认变绿，现在充电器已拔掉。替代此前“按提示立即重启”的不准确转述。电压9.6V及历史蜂鸣是用户现场报告，不是本轮直接测得；新增历史低压报警信息不等于当前仍低压。
- 官方Rosmaster电池说明要求充电时主电源OFF，红灯代表充电、绿灯代表充满；因此按OFF充电本身符合流程，用户未观察到绿灯不能当作已充满。OFF时亮灯的具体位置/供电路径未确认，不据此判板损坏或断言反向供电。建议先正常关机、主电源OFF，以配套充电器在有人照看时完成至绿灯，拔充电器后保持原接线再次开机复核。助手未执行关机/重启或任何硬件操作。
- 本轮命令`sudo -n /usr/sbin/nvpmodel -q --verbose`仍报告20W/id3，`/var/lib/nvpmodel/status`仍pmode0003；TPC实际15/配置240及GPU power/control差异保留，未解释为确定故障。boot仍29762023-526d-4076-af3b-007244756332，采样uptime835.64秒。sudo Python仅read_text读取现有INA3221 VDD_IN五次：10784、10792、10792、10792、10792mV，即10.784–10.792V。此为Jetson输入，不是电池/底盘控制板端电压，不测电机负载；不能据此判电量百分比、当前底盘低于9.6V或电池损坏。
- 10W为NVIDIA支持模式、某些TPC切换需重启，与当前GPU计算功能可用一起不支持直接判显卡损坏；未完成充电和先前低压断电使供电条件需先核验，但仍没有证明底盘CH340消失/HDMI黑屏的共同根因。未改功耗/驱动/串口映射/服务或启动实验。

## 用户补充曾短暂显示错误界面并点击Log Out

- 2026-10-10用户报告期间HDMI屏曾亮，提示出现问题并有Log Out按钮，用户点击；未提供精确文字/发生boot/时间，不能将其直接对应某次日志。外观描述与GNOME会话失败界面相符，仅为推断。Log Out本身是退出桌面会话，不构成改功耗/卸载串口驱动的证据。
- 当前boot只读`journalctl -b -o short-monotonic`查到gnome-shell-x11.service于单调109.39、204.89、300.39、395.89、491.39、586.89秒启动超时，各随后stop-sigterm超时并status9/KILL，进程反复重试。631.125秒gnome-session进入running；634.154秒gnome-shell的meta_display_get_monitor_geometry断言失败。显示日志628–630秒同时DP-1连接/断开、1920模式与NULL交替；只能确认存在桌面启动及显示链路异常，不证明两者因果或与10W切换同因。
- `systemctl show display-manager`仍active/running；`loginctl list-sessions`仍有jetson seat0 tty2；现有Xorg/gnome-session/gnome-shell进程存在。`DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority xrandr --query`当前DP-1 disconnected、1920×1080缓存几何。不能只用用户点击Logout解释当前无显示信号，也不能以桌面启动失败证明GPU硬件损坏。
- 保留现有NVRM_RPC_DCE失败证据，后续待充电完成复核；桌面故障与CH340完全未枚举分别检查，当前未证实共同根因。本轮未注销会话、重启显示服务、更改扩展/桌面配置、修改驱动或执行运动。
