# 小车硬件验收实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 按用户2026-10-08任务单，修复影响验收的底层缺陷，取得实际设备数据，交付逐项通过/失败/未验证的硬件验收报告。

**Architecture:** 保留现有profile及FourWheelChassis/ServoGimbal调用方式，默认Rosmaster后端改用不自动发送力矩、严格报告错误的串口会话。会话负责帧解析与有时间/有效性的遥测；传感器诊断独立于底盘业务进程，复用现有ROS镜像中的传感器驱动。

**Tech Stack:** Python 3.8、unittest、PySerial 3.5、OpenCV 4.7、已有arm64 ROS Foxy镜像。无新增安装。

**Spec:** `docs/superpowers/specs/2026-10-08-jetson-vehicle-bringup-design.md` 的4.1–4.4、4.6和验收B/C/D部分；以用户本次硬件验收任务单为当前范围约束，暂不实现云端、SLAM和论文算法，也不宣称整车驾驶调通。

## Global Constraints

- 保留当前26个跟踪文件修改和未跟踪适配；实施前保存快照及哈希，使用应用创建的隔离工作树，复用原始dirty baseline。
- 物理执行器需当次现场安排；默认读遥测不发送电机、PWM舵机、UART力矩、永久配置或flash命令。
- 用户明确要求本聊天按任务顺序执行，采用本聊天实现与测试；现有完整调通设计的未完成阶段仍保留。无需安装、部署或提交既有用户代码。
- 主机原始视频/日志不改写；新增证据使用独立目录 `outputs/hardware_acceptance/20261008/`。
- 不启动厂家完整导航launch、底盘节点、ROS业务服务或云端接口。传感器容器限制设备，结束即退出。
- 配置中的motion/gimbal标定flag保持false；实际轮径、轮序、方向、编码器单位换算和云台机械限位仍需现场标定。
- SDK协议依据当前安装Rosmaster_Lib 3.3.1源码快照，不修改厂商包。

## Review Focus

- 字符串/数字冒充bool、非有限数和错误轮序：在打开设备与写入之前拒绝。
- 短写/超时/断口：明确抛出通信错误，停止失败不能报告成功，关闭仍释放资源。
- 零值、旧值、坏校验和错长度：零值仅在有效报文后成立，读取不能使缓存变新，坏帧不污染有效数据。
- 不同组件/进程竞争串口：禁止第二个独立会话，允许显式注入共享会话，组件关闭不误关借入资源。
- 底盘静止时反馈：报告收到的协议类型、原始tick与SDK尺度，不把静止零编码器等同于方向/距离标定。

---

### Task 1: 参数门禁与停止语义

**Files:** Modify `car/control/vehicle_control/base.py`, `factory.py`, `platforms/rosmaster.py`; Create `car/test/test_hardware_safety.py`.

**Interfaces:** 保留现有create_chassis/create_gimbal、set_pose、stop、close签名。calibrated值必须为bool；停止尝试三次零输出，任一失败显式报错且资源关闭仍执行。

- [x] 写回归测试：字符串false拒绝；重复/小数轮序、无效限幅/方向和云台ID在连接之前失败；无效组合角度零写入；非有限输入被拒绝；停止遇到第一次写失败仍尝试后两次并报错；close即使停车失败也关闭。
- [x] 在隔离工作树运行 `python3 -B -m unittest car.test.test_hardware_safety`，确认新增用例因现有缺陷失败，保留日志。
- [x] 修改最小实现：完整前置验证、严格bool、整体姿态验证、停止重试并保留错误。
- [x] 运行新增测试与既有硬件映射/云台测试，保留日志。

### Task 2: 严格Rosmaster通信与遥测

**Files:** Create `car/control/vehicle_control/platforms/rosmaster_transport.py`, `car/test/test_rosmaster_transport.py`; Modify `platforms/rosmaster.py`。

**Interfaces:** `RosmasterSerialSession(serial_port, timeout=0.1, write_timeout=0.2, delay=0.002, serial_factory=None)`；`set_motor(*four_commands)`；`set_pwm_servo(servo_id, angle)`；`poll()`；`telemetry(max_age=0.5)`；`get_status()`；`close()`；上下文管理。默认构造不写串口；poll有界读取，telemetry每次读取反映实际数据年龄。显式注入session由调用者统一poll/close；默认适配器自行拥有会话。传输错误抛出`RosmasterCommunicationError`。

- [x] 写边界测试：构造无TX、无力矩；手工核对零PWM报文字节；短写和写异常显式失败；读异常影响状态；真实伪终端验证独占、有限超时和资源关闭。
- [x] 写解析测试：噪声/碎片/错校验/错长度/未知报文后恢复；编码器小端四int32；speed电池、MPU/ICM、姿态按SDK尺度；初始invalid/null、有效全零、stale、读旧值不刷新；断口后invalid。
- [x] 运行测试确认RED，按SDK帧格式实现最小会话。TX头FF FC，RX头FF FB；len包含type/payload/checksum；checksum为len+type+payload之和mod256。
- [x] 默认Rosmaster adapter使用新会话，保留controller注入；验证共享会话生命周期。跑新增及既有五组测试（旧103项）验证回归。

### Task 3: 真实设备数据验收

**Files（执行调整）：** 使用独立证据目录中的限时采集脚本，保留镜像驱动/launch依据。统一产品CLI `car/control/tools/hardware_acceptance.py`本轮未新增，未将它作为硬件验收通过条件，产品化留待后续任务单。

**Interfaces:** 诊断入口默认只读；明确duration、device、output参数。每次归档UTC/Asia-Shanghai时间、monotonic时间、设备配置、统计、源数据和文件hash；无数据必须显式失败/未验证。

- [x] 被动读取新会话20秒，TX字节数0；归档原始串口字节和解析JSONL，统计各类型速率、坏帧、反馈年龄和静止状态。没有反馈时如实记录；任何请求另行明确记录。
- [x] RGB单独持续30秒，记录有效/失败帧、时间、分辨率、图像变化、原始视频和设备元数据，不运行LCC。
- [x] 只读审阅现有镜像Astra/sllidar包和launch；执行仅传感器的设备探测和连续采样，设备挂载排除底盘，不使用privileged/host全/dev挂载，不启动导航。
- [x] 雷达使用实际info/health/scan确认型号、波特率、距离/角度与频率。深度确认型号、分辨率、编码/单位、有效像素与连续帧。首次试验失败保留原始错误，使用诊断定位后再有依据重试。
- [x] 底盘与云台仅在用户给出现场安排后按有界动作实测；若缺少安排，列为未验证并给出操作清单。

### Task 4: 复核、补丁及验收报告

**Files:** Create `coordination/2026-10-08-car-hardware-acceptance-report.md`, `coordination/2026-10-08-car-hardware-acceptance-manifest.json`；append `PROJECT_INDEX.md`；按需补充`car/control/README.md`接口文档。

- [x] 形成仅本轮增量补丁，相对dirty baseline核对；主工作树文件哈希匹配基线后才回写本轮必要修改，保留其余文件内容。
- [x] 完成独立复核：协议尺度、无启动TX、门禁、通信故障、停止语义、共享会话、证据真伪；必要故障用例RED→GREEN验证。
- [x] 执行相关所有安全测试与既有五组回归；记录未运行的旧闭环入口（存在缺失导入及云端/硬件副作用）。
- [x] 报告逐项通过/失败/未验证，区分功能实测、软件故障注入与历史资料；附设备清单、配置、运行ID、准确路径、配置/代码/日志hash、遗留问题。
- [x] 最终交回用户统筹核对证据，预计产出验收意见及是否允许进入本地行驶调试；若需现场补测，下一执行方为用户现场安排+本聊天测试。
