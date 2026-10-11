# 2026-10-11 充电后复查

- 时间：约17:20–17:22，Asia/Shanghai。用户报告已充满电，要求再次检查；未独立核验电池满充电压/容量或充电器状态。
- HEAD `bd6fe9c88f6c441eda2f4abbac14054ba3e9570b`，main，原有本聊天未提交改动保留。新boot `19cbd0cb-708c-465e-8a30-0f264b5a9060`，首次检查开机约2分钟；助手没有触发重启。无新实验run ID。

## 本次结果

- sudo仅读现有INA3221 VDD_IN五次：11672、11680、11672、11680、11672mV。Jetson输入11.672–11.680V，较昨天10.736–10.744V升高。非电池端/底盘板端独立测量，不能证明各路供电健康或满充容量。
- `lsusb`无CH340 1a86:7523，myserial不存在，仅雷达CP210x ttyUSB0。serial-monitor exit2/自动重试，初查10次；电量/IMU/姿态/编码器仍stale。充电并重新启动没有恢复底盘通信，不能继续仅以未充电解释当前故障。
- 现有模型2秒计数1229→1257，error null；相机、雷达online。功耗查询20W/id3；网页panel active、hardware/idle、available true、running false。没有开始驾驶。
- 显示连续三次xrandr（0.7秒间隔）为DP-1 connected→disconnected→connected，没有当前模式星号。Xorg至单调162秒已有633条DFP-1连接状态记录；GNOME已发生一次启动超时/KILL后重试；当前boot仍有5条NVRM_RPC_DCE失败。显示链路异常仍在，不能将某一次connected报告当作恢复。
- 深度本次新增异常：设备存在且ROS已启动深度流，但HTTP持续valid_fraction=0、距离null、帧率初查约26Hz（保存时20Hz）、采样新鲜，状态error。源码sensor_publishers.py确认无任何正深度值才判该error，不是USB设备不存在。镜头遮挡/环境、发射器/驱动、供电等具体原因尚未区分，未更改激光或启动新采集。

## 新发现：整条外接VIA HUB分支缺失

这次使用`rg -a`读取含NUL的`/var/log/kern.log`历史记录，找到正常启动时的物理USB拓扑（原始日志Oct 10 03:24:13为宿主时区，不直接当作上海时间；仅用于历史启动对照）：

```text
Jetson USB → 2109:2822（1-2，当前仍在）
             └─ 2109:2817（1-2.3，当前缺失）
                 ├─ 0483:5750 QDtech MPI7003（1-2.3.2，当前缺失）
                 └─ 1a86:7523 CH340（1-2.3.4，当前缺失）
```

同一外接VIA HUB的USB3侧2109:0817（历史2-1.3）也缺失。历史MPI7003明确绑定hid-multitouch，为触摸输入设备。相机/深度则在另一个05e3:0608 HUB下，不能把它仍在线当作这块VIA HUB在线的证据。雷达及相机分支端口号相较历史已有变化，不能要求用户按逻辑端口号寻找某个固定实体孔位。

**当前优先故障范围应上移到缺失VIA HUB及其上游USB线/接口/供电。** 这比单独猜CH340芯片或照片某个空口更有依据；尚不能判HUB已损坏。用户照片铜柱小塔下层的VIA芯片USB扩展板是候选实物，具体型号/两端线需现场核对。屏幕USB触摸与底盘同属缺失分支，但HDMI/DP视频链路不同；缺HUB不能独自证明视频黑屏原因，供电是否共用未核验。

## 证据、边界与下一步

- 忽略目录`outputs/diagnostics/20261011-charged-recheck/`保存历史USB枚举摘录、当前lsusb及HTTP/boot/电压摘录snapshot.json。snapshot中电压是前一只读命令的实际五次结果，HTTP是保存时的新采样。
- 命令：git状态/HEAD、boot/uptime、lsusb及-t、节点查询、systemctl show、nvpmodel -q、sudo Python读取现有hwmon；HTTP GET两次比较模型/遥测；现有X会话xrandr；journal、Xorg及历史kern.log只读筛选；静态核对深度状态实现。
- 下一步先核对这块外接HUB的既有电源插头和原上游USB线两端；用同接口、已知能传数据的线/正常主机端口做单变量对照，观察2109:2817/0817是否重新出现，之后再看CH340与触摸设备。先恢复共同上游，避免继续对空MCU口额外并联未知连接。
- 显示可另以已知正常显示器/线做对照；深度先确认镜头无遮挡且有合适距离目标，再复查有效值。当前仍不具备闭环行驶条件。
- 未重启/改服务、切功耗、写USB/sysfs、打开新硬件采集、控制电机或更改接线；只写诊断资料。恢复未完成。

## 同boot约328秒后的状态对比

- 用户询问是否变化，助手再次只读lsusb/节点、HTTP两次间隔1秒、xrandr三次间隔0.5秒和现有hwmon。没有助手侧设备/服务操作。
- 明确变化：深度已online，两次有效像素比例0.82958/0.83163、约23Hz，最近距离0.561m；原因未核实，不能把先前短暂全零归为已确认硬件故障。
- 未恢复：VIA2817/0817、CH340/myserial及触摸设备仍缺失，电量/IMU/姿态/编码器仍stale。GPU现有计数3992→4006/error null，panel hardware/idle。
- 显示三次均DP-1 connected，但没有活动输出几何/模式星号；GNOME仍activating/start、NRestarts3。短时间识别到屏幕不等于已稳定输出。Jetson VDD_IN单次11.432V，仍非底盘端测量。

## 用户拔除USB-A对USB-A上游线后

- 用户明确所拔线连接铜柱小塔最下层板与Jetson主机，且当前仍拔着。约同boot452秒只读lsusb/-t与journal，USB列表和前轮一致，未出现相应USB disconnect；VIA2817/0817及其下游原本就不在枚举列表。支持该上游通路在拔线前已未被主机识别，不能将用户本次拔线解释为原始故障。
- 用户的两端描述支持该线为缺失外接HUB的上游连接，具体PCB型号仍需现场核实。当前故障范围集中于线及两端接口/HUB供电/板端，不足以单独判线坏或HUB坏。下一步先将原线接回原位置观察枚举事件，再以已知良好同接口线或正常主机端口做单变量对照。
- HTTP相机/雷达/深度online，底盘四项busy/device absent（重试周期状态，不当作设备已被其他控制器占用的证明），panel hardware/idle。无助手侧硬件或服务操作。

## 用原雷达端口接HUB、雷达换到另一个端口

- 用户报告将HUB的USB-A线接到原雷达接口，雷达移到另一个接口。约同boot565秒只读核对：雷达于485.213秒从1-2.3断开，527.934秒在1-2.1重枚举10c4:ea60，绑定ttyUSB1，rplidar别名随之更新。这不是底盘串口恢复，不能按ttyUSB编号认设备。
- HUB2817/0817及CH340/myserial仍无，未看到对应HUB新连接/报错。原雷达口此前能用、替换后HUB仍无枚举，降低“仅原Jetson端口整体不通”的可能，但不能完全排除电源负载/高速接触差异。下一单变量对照为已知良好同接口数据线；若仍无HUB，核实板端供电并对照其它主机/连接路径。没有确定线或板损坏。
- 雷达USB识别已恢复，但HTTP lidar stale、数据年龄81.46→83.48秒，说明现有采集没有自动恢复实时数据。相机/深度仍online，panel idle，四路底盘无数据。此轮没有重启采集服务；用户请求为检查，不把旧数据误报online。

## 不使用A对A线的替代通信方案核对

- 用户询问能否不用A对A线、是否可无线。重新查看现有近照并核对[Yahboom ROS板接口说明](https://www.yahboom.net/public/upload/upload-html/1713173945/1.About%20expansion%20boardV3.0.html)，对应ROS板Micro USB用于主机通信，Type-C为供电，不可当作通信入口。可用接头匹配、已知可传数据的USB-A→Micro-USB线从Jetson直接连接底盘MCU通信口，绕过失联HUB；此为替代通路隔离方案，不撤销原接线曾正常工作，也不证明照片空口是原故障原因。实物版本仍应对照接口，不能硬插不匹配接头。
- [官方VL817 HUB说明](https://category.yahboom.net/products/usb-hub)把该HUB的Micro USB列作供电接口，不将HUB上的小USB口推荐为上游数据替代。两块板的同形小USB口作用不同。
- 现有rosmaster配置myserial及串口后端未提供已验证/已配置的无线底盘通道。网页WiFi连接Jetson不自动替代Jetson与底盘MCU的链路；无线串口桥需另加硬件并验证软件/断连处理，当前不是可直接切换的恢复步骤。
- 建议采用直接有线旁路时保持既有主供电，先断开失联HUB原A对A上游，避免未经核实的双上游连接；只确认CH340/myserial及被动遥测，不点击Start，不刷固件。此次仅资料/源码核对，未实施新接线或任何系统/硬件修改。

## 第二个开关打开后：共同供电问题确认恢复

- 用户发现有两个开关，此前只开启另一个，刚打开第二个。助手在同boot约830秒复核：762.055秒VIA0817、762.175秒VIA2817出现，763.174秒CH340出现并绑定ttyUSB2，765.719秒MPI7003触摸设备出现。当前myserial→ttyUSB2、rplidar→ttyUSB1，现有A对A通路已实际承载USB通信，不能继续将该线或HUB损坏作为当前已确认结论。
- 三次HTTP（2秒间隔）电量/IMU/姿态/编码器全部online，新鲜年龄约0.03–0.13秒；底盘报告12.1V。百分比100%是既有电压估算，不是容量实测。serial-monitor自动重试后已active/running，未由助手重启。
- xrandr显示DP-1 connected primary 1024x600+0+0，1024x600 60.04Hz模式有当前星号；GNOME active/running。因此此前HUB/底盘/触摸及显示链路异常在第二开关打开后一起恢复，强支持该开关控制的供电未开启为本轮主要故障。具体开关名称/独立电路仍未由丝印/线路核验，不推断一定是某个固定PCB按钮。
- 助手先前未充分核实两个开关状态，以空口推断缺线的建议已撤回；现在根据恢复事件更新判断。此前10W切换、GPU硬件损坏和A对A线损坏没有本次证据支持，当前不需使用替代无线/直连方案。
- 剩余独立问题：雷达实体已在新口枚举，但旧ROS采集约346–375秒stale；深度新鲜帧全零，状态error，原因仍待查，不能称七传感器全部恢复。panel active/hardware idle、running false，未有驾驶会话，尚未进行运动验收。
- 保存`outputs/diagnostics/20261011-second-switch-restored/snapshot.json`及`lsusb.txt`；没有助手触发重启、切功耗、改线或车控。已静态核实剩余恢复候选`systemctl restart vehicle-sensors.service`只启动既有雷达/深度ROS节点，容器只挂雷达串口而非底盘串口；重新初始化会控制雷达扫描，未在本轮执行。依项目AGENTS.md明确服务重启边界，需用户授权后再做。
