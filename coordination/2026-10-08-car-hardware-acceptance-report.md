# 小车硬件验收报告

日期：2026-10-08（Asia/Shanghai）。任务：`2026-10-08-car-hardware-acceptance`。状态：`reported`，待用户统筹审查。执行主机：`jetson-desktop-car-2`；项目：`/home/jetson/VehicleCloudCollaboration`。

## 结论与适配程度

当前小车的四路电机、四路编码器反馈、IMU原始反馈、RGB相机、深度相机和雷达已取得实际数据或动作证据。Astra重新插拔后，连续深度采样和一次停止后重新打开均成功。云台指令发送后没有确认相机转动，不能验收通过；是否安装了电动云台仍未知。

这些结果支持继续进行机械与相机标定，尚不能放行为地面自动行驶。实际轮序/前进方向、编码器距离换算、IMU轴向和偏置、相机透视及制动距离均未完成。现有项目已具备Jetson/Rosmaster适配和RGB经典LCC入口，但雷达、深度、编码器/IMU尚未证明已接入项目导航或驾驶控制；YOLOPv2权重与推理环境也未完成车端部署。固定相机可继续进行安装姿态与透视标定，云台不应被当作必有硬件。

## 范围、授权与证据等级

- 本轮按用户任务单集中完成硬件盘点、底层缺陷修复和验收，没有并行推进云端、SLAM或论文算法。
- 初次只读盘点保持独立，原报告未修改：[基线报告](/home/jetson/VehicleCloudCollaboration/coordination/2026-10-08-car-baseline-report.md)。本轮动作依据为用户随后明确授权：“可以架空,可以急停,所有动作都可以测试”，并确认“已架空并固定，现场有人可急停”。用户提供实物照片，并确认“四个轮子都动了,而且正常”。本轮采用架空低输出短时测试，未进行落地行驶。
- **实测**：真实宿主查询、驱动输出、原始数据和串口反馈；**现场反馈**：用户观察，单独标明；**代码推断/软件测试**：接口实现、协议解释和故障注入；**历史**：树莓派或旧演示，不替代当前Jetson结果。测试通过不代表物理距离、制动或算法性能通过。
- 未安装依赖、拉取镜像、重启或修改业务服务，未改厂商SDK，未提交或推送Git。传感器仅使用已有镜像的临时容器，排除底盘设备，退出自动删除。原视频和日志保留原样，分析使用独立产物。

## 实际硬件清单

| 部件 | 当前识别与接口 | 驱动/配置 | 依据与未知项 |
| --- | --- | --- | --- |
| 计算平台 | NVIDIA Jetson Orin NX，device-tree `p3767-0001`，6核CPU，7.2GiB可见RAM；aigo P2000 NVMe 128GB | Ubuntu 20.04.6 aarch64，内核5.10.104-tegra，L4T R35.3.1 | 系统只读盘点实测；未做GPU负载或推理性能验收 |
| 底盘 | 实物四麦克纳姆轮；Rosmaster兼容串口控制板，固件3.5，车型报文type=1、reserved=0 | `/dev/myserial -> /dev/ttyUSB1`；CH340 `1a86:7523`；115200 baud；SDK 3.3.1协议、新会话实现 | 四原生电机通道实测；本地SDK将type=1解释为X3。板卡铭牌/MCU、电机及减速比未核验，不能从协议推断完整SKU |
| RGB相机 | Orbbec相机组件的UVC RGB接口；USB `2bc5:050f` | `/dev/video0`、`video1`；OpenCV/V4L2；640×480 | 实际RGB采样通过。双节点不代表两台RGB相机；RGB与深度外参未验收 |
| 深度相机 | OpenNI设备名称 **Astra**；序列号 `ACR7C33004M`；USB `2bc5:060f` | 镜像内`astra_camera`，depth-only，640×480、请求30fps、ROS `16UC1` | 实测有效深度；具体Astra子型号、绝对测距精度、近距离能力未知。用户重新插拔后节点为`/dev/bus/usb/001/014`，不能硬编码旧USB编号 |
| 二维雷达 | 实物RPLIDAR；设备型号码 **0x71**；固件1.02、硬件版本18；序列号 `9A7DE386C8E699D3C1E19BF563A84A61` | `/dev/rplidar -> /dev/ttyUSB0`；CP210x `10c4:ea60`；`sllidar_ros2`，1,000,000 baud、DenseBoost | 身份查询、health和扫描实测；S2类配置已可用，但具体商业型号S2/S2L仍需铭牌，未据桥接芯片或旧配置确定型号 |
| 编码器 | 控制板四路原生累计tick反馈 | RX `0x0D`，小端四int32，约24.4Hz | 双向单轮动作对应单通道tick变化；每轮物理位置、每圈tick、轮径/距离换算未标定 |
| IMU/姿态 | 实际RX `0x0B`的MPU类九个int16及`0x0C`姿态报文 | 按当前厂商SDK尺度解析gyro/accel、磁场SDK原始尺度；姿态除10000为rad | 反馈可读取且更新；“MPU”仅为报文协议类，实际芯片未确认。轴向、偏置、磁干扰、roll/pitch真实性未验证 |
| 云台 | **未确认安装电动双轴支架** | 项目配置PWM ID1/2；这是代码配置，不是实物证明 | 三轮小角度命令后图像无明显视角变化，用户确认没有观察到转动；电动机构、接线、供电、型号均未知 |

主机识别的命令依据为基线报告中的device-tree、`lscpu`、`free -h`、`df -hT`、`lsusb`、sysfs/udev及Python依赖检查。本轮驱动、配置与协议快照在下面的证据目录。用户实物照片原路径保留在manifest，不把照片中未读清的标签当作型号依据。

## 逐项验收记录

所有下列run ID均位于：

`/home/jetson/VehicleCloudCollaboration/outputs/hardware_acceptance/20261008/`

| 验收项目 | 结论 | 实际方法与结果 | run ID / 证据 |
| --- | --- | --- | --- |
| 被动串口反馈 | **通过** | 20.003秒收到32650 bytes、1949有效帧；TX=0，校验/长度错误=0。motion/IMU/attitude各487帧，encoder488帧；静止velocity=0、电池12.3V | `serial-passive/raw.bin`、`telemetry.jsonl`、`summary.json` |
| 控制板身份 | **通过** | 两个只读请求共14TX bytes；实际固件3.5、type=1/reserved=0。初次实现遗漏两字节车型回应，已修复并在主工作区再次实测成功 | `serial-identity/`保留原异常；`serial-main-verified/summary.json`为修复后证据 |
| RGB持续输出 | **通过** | 30.072秒、874帧、640×480、读失败0；帧时间实测30.006Hz，最大间隔0.03875秒；画面有变化 | `rgb-continuous/raw.mp4`、`first.png`、`frames.jsonl`、`summary.json` |
| 深度首次采样 | **失败，已保留** | 554帧全部深度为0；随后IR联合启动及再次depth-only启动失败 | `depth-continuous/`、`depth-with-ir/`、`depth-laser-after-stream/`及console |
| 重新插拔后深度 | **通过** | 20.386秒、553帧全部有非零深度，29.315Hz；有效像素56.34%–60.60%，最大帧间隔0.0999秒；640×480、16UC1、小端、step1280 | `depth-after-replug/depth.raw`、`samples.jsonl`、`first_depth.npy`、`driver.log`、`summary.json` |
| 深度停止后重开 | **通过（一次复测）** | 无再次拔插，10.386秒、262帧全部有效、29.553Hz；有效像素57.82%–61.18%。一次复测不证明冷启动或长期可靠性 | `depth-reopen-with-tmp-logs/`；前一尝试`depth-reopen-console.txt`为ROS日志只读目录错误，未启动驱动，不能算设备失败 |
| 雷达持续扫描 | **通过** | 20.236秒169扫描，10.023Hz；每扫描3240点，首/末有效1972/1983点，最小有效0.207m，场景最远约10.785m；health OK，退出日志含Stop motor | `lidar-continuous/samples.jsonl`、`driver.log`、`command.json`、`summary.json`；身份见`lidar-identity/` |
| 四路电机及编码器 | **通过（架空通道功能）** | M1–M4分别原生+20/-20，各0.4秒，其他轮命令0；对应tick双向变化，其余通道不变。用户确认四轮正常动作 | `wheels-lifted/raw.bin`、`samples.jsonl`、`summary.json`；脚本`actuator-bench.py` |
| 常规停止 | **通过（本轮架空条件）** | 每次动作后三次全零命令，随后反馈velocity=0；最终主工作区零输出复核最后TX=`fffc07100000000017`、反馈有效且velocity=0，串口关闭 | `wheels-lifted/`、`serial-main-verified/`；传输写入没有物理执行ACK |
| IMU可解释原始反馈 | **通过（读取/协议尺度）** | MPU类gyro/accel/磁场原始报文持续更新，静止accel量级约10m/s²；未实施轴向、偏置、动态姿态或融合校准 | `serial-passive/telemetry.jsonl`、`wheels-lifted/samples.jsonl` |
| 云台相机转动 | **失败；硬件存在性未验证** | PWM ID1/2请求pan90→85→95→90、tilt80→75→85→80，后两轮每步2秒。发送完整帧不证明舵机动作；第一轮SIFT配准位移<0.22像素，用户后续确认没有观察到转动 | `gimbal-bench/`、`gimbal-bench-repeat/`、`gimbal-bench-repeat2/`；`gimbal-bench/image-motion-analysis.json` |
| 轮序/里程计、姿态标定、地面行驶/制动、故障失联停车、导航与算法性能 | **未验证** | 本轮未落地、不运行完整导航/算法；不能以设备枚举、mock、架空点动或零速报文替代 | 后续任务单 |

原生编码器变化：M1 +228/-309；M2 +404/-476；M3 +375/-440；M4 +279/-344 tick。各轮原生+20不是20m/s或统一转速；单次无负载tick差异不构成电机故障或已完成调速标定的证据。物理FL/RL/FR/RR与前进正方向未独立确认。

深度单位来自镜像驱动选用`PIXEL_FORMAT_DEPTH_1_MM`，ROS原值以mm解释；驱动`depth_scale=1`是图像缩放参数，不是米转换。约3.8m的中位值仅为当前场景SDK读数；较远读数与绝对精度未经量尺校核。原始553帧339763200 bytes，重开262帧160972800 bytes，逐帧offset/长度已核对。派生预览：[实际深度首帧](/home/jetson/VehicleCloudCollaboration/outputs/hardware_acceptance/20261008/depth-preview.png)，白色区域为无效0值。

RGB视频封装为20fps，而设备实际采集约30fps；播放时长不能作为采集时长，计算以`frames.jsonl`时间为准。雷达原始JSONL保留Python输出的Infinity非有效距离，分析须用finite检查及range_min/range_max，不能直接把所有点计为有效。

## 软件修复、版本与验证

主仓库分支`main`，HEAD `33a5a712b97af15feadc11f4751e60edaa05dc08`，本轮没有新commit。开始有26个跟踪文件修改及未跟踪适配，本轮保存48个dirty文件哈希和跟踪diff。隔离工作树：`/home/jetson/.codex/worktrees/car-hardware-acceptance/VehicleCloudCollaboration`，分支`codex/car-hardware-acceptance`；复用原有dirty内容，完成独立审查后，逐文件确认主工作区仍匹配原始哈希，再合回六个本轮文件：

- `car/control/vehicle_control/base.py`：严格有限数/方向/限幅、整体云台姿态前置校验、停止三次尝试且传播错误、停止取消旧渐变、关闭期间拒绝新非零输出。
- `car/control/vehicle_control/factory.py`：标定状态必须是真bool；公共工厂保留显式共享会话所有权；Rosmaster全零emergency_stop不受错误轮序/限幅标定阻断。
- `car/control/vehicle_control/platforms/rosmaster.py`：连接前验证轮序/云台参数；默认使用新串口会话，借入共享会话可显式`owns_controller=False`。
- 新增`platforms/rosmaster_transport.py`：构造/关闭不发执行器命令，有界超时、独占串口、短写/断口显式错误、故障后禁止非零/舵机输出但保留零输出尝试；解析校验与长度、初始None/invalid、接收时间/年龄/stale。
- 新增`car/test/test_hardware_safety.py`和`test_rosmaster_transport.py`：故障注入、边界、共享会话、停止/关闭竞争及伪终端验证。

原26个跟踪文件内容保持原样；原48项中除本轮三个必要修复文件外的45项已核对未变，索引后续仅追加本轮事实。原始跟踪diff SHA256仍为`9523bed7dabd72270b53dd79c2dae73d6c4895097486e95d7139b39903b51478`。恢复和增量依据：`preexisting-worktree.json`、`before/`、`preexisting-tracked.patch`、`acceptance-incremental.patch`、`integration-record.json`。

主工作区实际验证命令：

```bash
python3 -B -m unittest car.test.test_lane_centering car.test.test_camera_transform car.test.test_camera_gimbal car.test.test_hardware_mapping car.test.test_lcc_web car.test.test_hardware_safety car.test.test_rosmaster_transport
```

结果：**134项通过，1.857秒，退出0**；日志`main-regression.log`。包含原有103项与新增31项；属于软件测试，不证明失联制动和实车算法。RED→GREEN及独立评审裁定见[评审记录](/home/jetson/VehicleCloudCollaboration/outputs/hardware_acceptance/20261008/review-record.md)。独立评审发现遥测锁等待可能误判新鲜和工厂漏传所有权，均补回归后修复；同时修复停止标定阻断与关闭竞争。

主工作区随后实际被动采样5秒TX=0，再发两条只读身份请求和三次全零，总TX41 bytes；579有效帧，校验/长度错误0，固件和车型有效，最终零速及closed=true。脚本`final-serial-check.py`，数据`serial-main-verified/`。新会话不自动启动SDK接收线程，读取方须显式`poll()`；底盘和云台不能各自独占同一串口，需共用一个会话并统一管理生命周期。上层完整LCC/Web与双执行器共享生命周期尚未集成验收。

旧`car/test/closed_loop_test.py`仍有缺失controller导入及云端/硬件副作用，未运行。实施计划中统一产品化`car/control/tools/hardware_acceptance.py`本轮未新增：验收采用独立、限时、归档的采集脚本；报告不宣称统一多传感器SDK已交付。

## 环境、配置与复现依据

Python3.8.10；基础numpy/cv2/PySerial环境沿用已有安装，相关测试与实际RGB/串口采集证明这些路径可运行。宿主未核验到ROS安装；传感器复用已有arm64镜像`yahboomtechnology/ros-foxy:4.0.5`，image ID `sha256:0e2a3a0c7df2a41235bb9e8c760f993795313cddd6f3992f73a817ec8ef4763c`。镜像内ROS Foxy及`/root/yahboomcar_ros2_ws/software/library_ws/install/setup.bash`，只用Astra/sllidar独立节点，未运行厂家完整导航launch。

深度节点参数见各run的`command.json`：namespace `/camera`，enable_depth=true，color/IR/UVC/点云/TF=false，640×480/30fps，订阅`/camera/depth/image_raw` best-effort。雷达参数：serial_port `/dev/rplidar`、1000000 baud、frame_id laser、inverted=false、angle_compensate=true、DenseBoost；订阅`/scan`。配置、驱动源码读取依据：`ros-driver-inspection.txt`、`depth-device-list.txt`、`depth-driver-parameters.txt`、`depth-image-publication-source.txt`、`lidar-config-source.txt`、`Rosmaster_Lib-3.3.1-source.py`。

临时容器无网络、read-only根文件系统、cap-drop ALL、no-new-privileges、pids-limit128、/tmp tmpfs128m，只挂相应传感器设备及新证据目录，不挂底盘。深度重开明确设置`ROS_LOG_DIR=/tmp/ros-log`及`ROS_HOME=/tmp/ros-home`。第一次新增目录写权限错误及本次ROS日志只读错误保留console，不将未启动驱动的容器失败算成设备故障。新增输出目录采集结束恢复755。

项目默认`run.sh`和公共配置仍选`raspbot_pi5`；本车必须显式选`rosmaster_jetson`。该profile的motion/gimbal calibrated保持false，YOLO关闭、透视为空，轮参数是待标定占位值。报告不把门禁改true。基线中的旧Pi权重路径不可用；本轮项目范围内按`.pt/.pth/.onnx/.engine`名称再次搜索无结果，未安装torch/ONNX Runtime或进行GPU推理。模型部署应在硬件/标定任务完成及统筹分配后开展。

新原始视频和日志现位于上述outputs证据根目录（Git忽略），不是此前不存在的历史Pi记录。此前5秒电机禁用LCC run `20261008_034402_061387_dryrun`仅作开发验证，当前桌面场景曾给出虚拟转向，不能证明道路可行驶或闭环驾驶正确。

证据完整路径、字节数、SHA256、源码/配置哈希、时间来源、run列表见[manifest](/home/jetson/VehicleCloudCollaboration/coordination/2026-10-08-car-hardware-acceptance-manifest.json)。当前原始证据尚无异机独立备份，Git不携带ignored原始数据；后续统筹同步须核对manifest并保留车端原件。

## 遗留问题与按依赖顺序的下一任务

| 顺序 | 负责人 | 下一工作与前置条件 | 预计产出 |
| --- | --- | --- | --- |
| 1 | 用户：论文总纲与统筹 | 审查本报告、manifest和架空证据，确认“可进入标定、尚未放行地面自动驾驶”；确定固定相机路线，或现场确认确有云台后另列任务 | 《硬件验收意见》及下一阶段任务单 |
| 2 | 本远程小车聊天＋用户现场配合 | 保持架空固定和急停条件，逐原生通道确认FL/RL/FR/RR及正方向；读取电机/板卡/雷达铭牌。确认云台是否存在，若无则按固定相机测安装姿态；量测轮径/轮距及编码器换算，检验IMU轴向/偏置与相机透视 | 《机械与相机标定记录》、实物硬件清单补充、带证据的Jetson配置；是否解除各标定门禁须依据相应验证 |
| 3 | 本远程小车聊天，按用户新的场地安排 | 前项验收后，在封闭场地验证低速直线/转向、常规停止和失联保护；再确认传感器同时采集、时序与资源冲突、深度重复启动。地面动作按当次明确安排执行 | 《低速行驶与停止验收报告》、同步原始日志/视频、遗留问题；决定是否进入导航/感知集成 |
| 4 | 用户统筹分配；车端与本地实现评测聊天分别承接 | 车端功能可靠后确定接口和实验协议；车端负责实际感知/控制/导航及采集，本地聊天负责分配的云端接口、模型评测和分析 | 版本明确的接口/实验任务单，以及各自实现与评测产物；不能提前将硬件验收写成论文算法结果 |

给下一车端任务的可复制文字：

> 复用2026-10-08硬件验收报告和manifest。先在架空固定、现场可急停条件下核对四原生电机通道的物理轮位和前进方向，补铭牌清单、轮径/编码器换算和IMU轴向；明确是否安装电动云台，若未安装则按固定Astra相机完成安装姿态与透视标定。交付标定记录、配置、原始证据路径和通过/失败/未验证结论。门禁只按验收证据调整，地面行驶等待当次场地安排，不并行推进云端或SLAM。
