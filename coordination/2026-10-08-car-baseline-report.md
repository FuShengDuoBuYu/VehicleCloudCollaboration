# 2026-10-08 车端只读基线盘点报告

- 任务 ID：`2026-10-08-car-baseline`；状态：只读盘点完成，完结文件校验见末节。
- 主机：`jetson-desktop-car-2`；仓库：`/home/jetson/VehicleCloudCollaboration`。
- 实际采集时间：2026-10-08 15:14–15:24（Asia/Shanghai，UTC+08:00）。宿主机时区为 America/New_York，同期为 03:14–03:24（UTC−04:00）。未改系统时间或时区。
- 分支：`main`；HEAD：`33a5a712b97af15feadc11f4751e60edaa05dc08`；工作树有既有未提交适配，不能用 HEAD 单独代表本次源码。
- 实验 run ID：无。本轮未采集相机、雷达或底盘数据，未运行推理、回放、测试套件或运动实验。
- 写入范围：本报告及 `/home/jetson/VehicleCloudCollaboration/PROJECT_INDEX.md` 的盘点摘要。未修改业务代码、车型参数、服务、原始数据或 bootstrap manifest；未安装依赖、启动/停止服务或容器、发送云端请求、创建相机流或串口/I2C控制对象、输出电机或舵机指令。

## 1. 结论与证据口径

**当前是 Orin NX/串口 Rosmaster 车型；硬件接口可枚举，双车型适配源码已经存在，但没有本轮证据证明 Jetson 的感知、控制或导航已调通。** 三份入口材料齐全，主要缺口是 Jetson 实车标定、推理依赖与权重、原始运行证据，以及传感器数据级核验。

| 类别 | 本报告含义 | 不能据此宣称 |
| --- | --- | --- |
| 实际检查 | 本轮执行系统只读查询、文件/配置解析、少量基础依赖导入 | 硬件数据正常、闭环稳定、实车安全或性能达标 |
| 源码/配置推断（E2） | 在当前脏工作树中定位到接口、算法、门禁、路径 | 已通过测试、已部署、已完成实车验证 |
| 历史记录（E3） | 旧文档、旧测试 JSON、旧主机路径/实验描述 | 当前 Jetson 的运行结果；有完整原始证据时才可进一步升级 |
| 未知/缺口 | 未执行数据采集或未找到对应证据 | 不能用目标架构、设备名称或 SDK 函数补齐 |

特别注意：`dry-run` 指电机输出未启用，**不自动等于硬件只读**。默认树莓派配置会初始化云台；`Rosmaster()` 构造也有串口写操作。本轮没有运行这些入口。

## 2. 入口材料、来源与工作树

已读取以下真实路径：

- `/home/jetson/VehicleCloudCollaboration/AGENTS.md`
- `/home/jetson/VehicleCloudCollaboration/PROJECT_INDEX.md`
- `/home/jetson/VehicleCloudCollaboration/coordination/2026-10-08-car-baseline.md`
- `/home/jetson/VehicleCloudCollaboration/skills/vehicle-cloud-thesis/SKILL.md`

入口三文件及两项项目技能的初始 SHA-256 均与 `/home/jetson/VehicleCloudCollaboration/coordination/2026-10-08-bootstrap-manifest.json` 对应项一致。`.agents/skills/` 的两项链接分别解析到仓库内 `skills/` 的实际目录。manifest 记录的来源是 `C:\Users\fushe\Desktop\thesis`，source HEAD 为 `15a2b1d877215055832b54ca8947b3e83e67901a`，文件来源版本是 working-copy；这不是当前车端 Git HEAD。

开始时 `git status --porcelain=v1 -uall` 显示 **26 个已跟踪文件修改、18 个未跟踪文件/链接条目**；普通 `git status --short` 折叠目录后为 9 个 `??` 条目。交接材料的旧计数不能代替本轮计数。`git diff --stat` 为 642 行增加、296 行删除；忽略行尾空白后的统计相同，改动包含实质代码和配置修改。

既有修改按仓库根目录 `/home/jetson/VehicleCloudCollaboration` 分组如下：

| 范围 | 既有文件 |
| --- | --- |
| 文档和启动 | `README.md`、`car/autodrive/README.md`、`car/control/README.md`、`run.sh` |
| 相机及配置 | `car/autodrive/camera/gimbal.py`、`car/autodrive/config/onboard_runtime.yaml`、`onboard_runtime.example.yaml` |
| 控制及感知 | `car/autodrive/control/{drive_runtime,lane_centering}.py`、`car/autodrive/perception/{outer_loop,visualization,yolopv2_fusion,yolopv2_onnx}.py` |
| 运行及 Web | `car/autodrive/runtime/onboard.py`、`car/autodrive/web/{cli,server}.py` |
| 工具 | `car/autodrive/tools/{align_camera_gimbal,calibrate_perspective,capture_onboard,check_wheel_directions,pi_self_check}.py` |
| 适配和测试 | `car/control/vehicle_control/{__init__,hardware}.py`、`car/test/{test_camera_gimbal,test_hardware_mapping,test_lcc_web}.py` |

未跟踪业务适配文件共 8 个：`car/control/vehicle_control/{base,factory,profile}.py`、`platforms/{__init__,raspbot,rosmaster}.py`、`profiles/{raspbot_pi5,rosmaster_jetson}.yaml`。其他未跟踪项属于接入规则、索引、技能和 coordination 材料。

已检查相关差异：新增按 profile 选择后端和深度合并、串口四轮/云台适配、标定门禁、运行归档记录车型、Web/工具传递车型参数；六个控制/感知模块增加 `from __future__ import annotations`，自检的 Python 下限调整到 3.8。公共配置同时把 `expected_lane_width_ratio` 从 0.58 改为 0.70。上述都属于**本轮开始前已有改动**，本轮未修复或验证它们。

初始内容快照位于 `/tmp/car-baseline-precheck-20261008.json`，覆盖 1,688 个可读的已跟踪/未跟踪普通文件，仅用于本轮保留性核对；不是新的实车数据或永久备份。

## 3. 实际执行的检查与限制

以下为实际命令/脚本操作摘要，可按相同范围复核；没有把建议命令列作已执行。

| 编号 | 实际检查依据 | 输出或限制 |
| --- | --- | --- |
| C01 | `pwd`、`git status --short`、`git status --porcelain=v1 -uall`、`git branch --show-current`、`git rev-parse HEAD`、`git diff --stat`、`git diff --numstat`、相关 `git diff -- <path>` | 仓库、HEAD、脏工作树及平台适配差异 |
| C02 | `sed` 读取入口/源码；`rg --files` 定位入口；Python `hashlib.sha256` 对入口及配置校验 | 入口齐全、初始来源哈希一致；未执行项目模块 |
| C03 | Python `platform`；读取 `/etc/os-release`、`/etc/nv_tegra_release`、`/proc/device-tree/{model,compatible}`、`/proc/meminfo`；`lscpu`、`free -h`、`df -hT / /home/jetson/VehicleCloudCollaboration /tmp` | 平台、内存、存储 |
| C04 | `date -Iseconds`、`date -u -Iseconds`；子进程仅设置 `TZ=Asia/Shanghai`；`timedatectl show --property=Timezone --property=NTPSynchronized` | 宿主时区 America/New_York，NTP 同步 yes；进程级显示转换不改系统设置 |
| C05 | `lsusb`、`lspci -nn`；读取 `/sys/bus/usb/devices/*/{idVendor,idProduct,manufacturer,product}`、`/sys/class/video4linux/*/name`、tty driver/uevent、`/sys/devices/gpu.0/of_node/compatible` | USB、视频、串口、GPU 节点的枚举证据 |
| C06 | Python `glob`/`stat`/`realpath`/`os.access` 查询设备；读取 `/etc/udev/rules.d/{usb,video}.rules` | 设备映射和权限；没有打开 `/dev` 中的任何硬件设备 |
| C07 | Python `sys.version`、`sys.executable`、`sys.prefix`、`sys.path`、`importlib.metadata.distribution`；有界列出常见 venv/Conda/ROS/CUDA 目录；`shutil.which` | 当前解释器和安装元数据；未找到不能扩大为全盘绝对不存在 |
| C08 | `python3 -B` 仅导入 `numpy`、`cv2`、`yaml`、`serial`、`smbus2` 并打印版本和模块路径 | 五项基础依赖导入成功；未创建相机、串口、I2C、GPU 推理会话 |
| C09 | `dpkg-query -W` 查询 `nvidia-l4t-core`、`nvidia-jetpack`、`nvidia-l4t-cuda`、`cuda-toolkit-11-4`、`ros-*` 及 CUDA/TensorRT/OpenCV 包 | L4T 基础包存在；部分查询非零是未匹配包，不能当作安装失败 |
| C10 | `docker ps`、`docker ps -a`、`docker image ls`（限定字段）；`docker image inspect yahboomtechnology/ros-foxy:4.0.5 --format '{{.Id}}\t{{.Size}}'` | 没有运行或停止容器；保留一份 ROS Foxy 镜像；未启动/进入容器 |
| C11 | `systemctl is-active` 查询 OLED/Docker/旧服务；`ps -eo pid,comm`，读取 Python 进程时只返回 `.py` 路径；`lsblk -o NAME,SIZE,TYPE,MOUNTPOINT` | 当前服务和磁盘；未输出完整进程参数或环境值 |
| C12 | Python `yaml.safe_load` 及独立递归合并脚本；`ast.parse` 读取源码；`zipfile.ZipFile` 读取已安装 Rosmaster egg 中的函数 | 静态确认配置与接口；未实例化驱动、运行项目自检或测试 |
| C13 | 有界 `Path.iterdir`/`os.walk`、文件 stat/可读性；仓库 `rg --files --hidden --no-ignore` 按模型/视频/日志扩展名筛选并排除 `.git` | 原始数据和模型位置检查；未解码视频、分析整库或复制/改写数据 |

沙箱最初用 tmpfs 覆盖 `/dev`，只看到 sysfs 中的相机/串口；`lsusb` 返回 libusb -99，系统总线和 Docker socket 查询受限。随后两次获批的只读宿主机查询重新核验 C04–C06/C10–C11，成功看到真实节点和服务。**先前的空 `/dev`、空进程视图不作为“硬件未连接/服务未运行”的结论。** 不需要用户补充权限即可完成本轮查询。

环境只报告代理变量名 `HTTP_PROXY`、`HTTPS_PROXY`、`http_proxy`、`https_proxy`；代理文件 `/home/jetson/.config/codex/home_proxy_env` 权限 600，其目录权限 700。未输出代理值、密码、Token、私钥、认证 URL，也未测试网络推理。

## 4. 计算平台与已枚举硬件

| 项目 | 本轮实际检查 | 配置/历史与仍未知事项 |
| --- | --- | --- |
| 主板/系统 | device tree：`NVIDIA Orin NX Developer Kit`，`nvidia,p3768-0000+p3767-0001`、`nvidia,tegra234`；Ubuntu 20.04.6，aarch64，内核 5.10.104-tegra，L4T R35.3.1 | 历史记为 Orin NX 8 GB、JetPack 5.1.1 系列；本轮未发现 nvidia-jetpack 元包，不能据此否定 L4T，也没有验证完整 JetPack 推理栈 |
| CPU/GPU | 6 个在线 ARM CPU，最大频率字段 1971.2 MHz；GPU sysfs compatible 为 `nvidia,ga10b` | 没运行 GPU 负载或测温/功耗/吞吐；GPU 节点存在不等于 CUDA/TensorRT 可用 |
| RAM/存储 | RAM 7.2 GiB，采样时可用 5.1 GiB；zram swap 3.6 GiB 未用；aigo P2000 128GB，NVMe 119.2 GiB；ext4 根文件系统 df 117G，总已用 31G、可用 81G（28%） | 容量是采样值；未做存储性能或健康测试 |
| 底盘串口 | `/dev/myserial -> /dev/ttyUSB1`；USB `1a86:7523`，ch341-uart；节点权限 0777，当前用户可读写 | `rosmaster_jetson.yaml` 描述 Rosmaster X3，backend rosmaster、car_type=1；只能确认配置和接口。未读固件/反馈，未实测轮序、方向、底盘几何或轮型 |
| RGB/深度相机 | USB `2bc5:050f`、product `USB 2.0 Camera`；`video0`/`video1` 均由 uvcvideo 枚举；另有 `2bc5:060f`、product `ORBBEC Depth Sensor`。真实 `/dev/video0`、`video1` 均存在且可读写 | 两个视频节点不能当作两台相机。配置 index=0、640×480、20 fps、旋转/翻转关闭；本轮未开流，RGB 尺寸/FPS、深度数据、时间戳和具体相机型号未知 |
| 激光雷达接口 | `/dev/rplidar -> /dev/ttyUSB0`；USB `10c4:ea60` CP2102N，cp210x；节点 0777、可读写；udev 别名匹配 | 外部旧 Compose 配置 `RPLIDAR_TYPE=s2`。这是型号配置证据，CP210x/别名不能证明实体 S2；扫描流、扫描频率、距离质量、安装外参和所需波特率未核验 |
| 编码器/IMU | 没有独立 IIO 设备枚举；已安装 Rosmaster SDK 源码具有编码器、加速度、陀螺仪、磁力计、姿态和运动反馈 getter | 这些可能通过底盘 MCU 串口上报，IIO 缺失不证明传感器缺失。未接收数据，芯片型号、单位、轴向、有效帧率、固件支持和标定未知 |
| 云台 | Jetson profile 配置 pan servo 1、tilt servo 2，限制 0–180/0–100 度；禁止启动初始化 | 通道、方向、机械零位和限位尚未实测；树莓派 25 度朝前的历史不能移用于 Jetson |

厂商库真实位置为 **ZIP 格式 egg 文件** `/usr/local/lib/python3.8/dist-packages/Rosmaster_Lib-3.3.1-py3.8.egg`，内部 `Rosmaster_Lib/Rosmaster_Lib.py`。它不是可用 `sed` 拼接进入的目录；本轮随后改用 ZIP 只读解析：

- 第 14 行构造函数使用 `serial.Serial(com, 115200)`，并在构造末尾调用 `set_uart_servo_torque(1)`。因此不能通过“只构造对象、不 set_motor”保证只读。
- 第 506 行 `set_motor` 写串口，异常被捕获并打印后吞掉。源码推断：上层函数返回不能独立证明下发成功，需要未来真实反馈/错误观测验证。
- 第 1095/1102/1108/1115/1128/1144 行是遥测 getter，返回内部缓存，初始化缓存为零。不能把未更新的零值当作健康传感器读数。

## 5. Python、ROS 与推理环境

本轮实际解释器 `/usr/bin/python3`，Python 3.8.10，prefix/base_prefix 均为 `/usr`；没有活跃 venv/Conda 环境变量。常见项目 `.venv`、`venv`、`/home/jetson/miniconda3`、`anaconda3`、`.virtualenvs` 均不存在。其他未搜索的自定义环境仍未知。

| 组件 | 当前版本/状态 | 与仓库 requirements.txt 的关系 |
| --- | --- | --- |
| numpy | 1.24.3，实际导入成功 | 达到 >=1.24.0 |
| OpenCV Python | metadata 4.7.0.72，导入 cv2 4.7.0 | 低于 >=4.8.0；系统另有 4.2 开发包和 NVIDIA libopencv 4.5.4 包，不代表 Python 用这些包 |
| PyYAML | 5.3.1，实际导入成功 | 低于 >=6.0 |
| pyserial / Rosmaster-Lib | 3.5（serial 实际导入）/3.3.1（metadata + ZIP 源码） | Jetson 后端依赖；根 requirements 未显式列出这两项 |
| smbus2 | 0.4.2，实际导入成功 | 低于 >=0.4.3；主要用于树莓派兼容后端 |
| SciPy / Pillow / requests | metadata 1.3.3 / 7.0.0 / 2.22.0 | 低于 >=1.10.0 / >=10.0.0 / >=2.31.0；本轮未逐项导入 |
| Jetson.GPIO / Flask / pip | metadata 2.1.1 / 2.3.2 / 23.2.1 | 存在不等于业务用到；未操作 GPIO 或启动 Flask |
| torch / torchvision / onnxruntime / onnxruntime-gpu / ONNX / TensorRT | 当前 Python 无对应 distribution metadata；torch/onnx/TensorRT 模块目录亦未找到 | 当前解释器没有已核验的语义推理后端；未建立模型会话 |
| scikit-learn / transformers / ultralytics / tqdm / python-dotenv / pytest | 当前 Python 未找到安装元数据 | 根依赖未齐；unittest 属于标准库，不需要 pytest 才能运行已有 unittest |

`/usr/local/cuda` 不存在，PATH 中没有 `nvcc` 或 `nvidia-smi`；`nvidia-l4t-core`、`nvidia-l4t-cuda` 均为 `35.3.1-20230319081403` 已安装。**不能把 nvidia-smi 缺失单独视为 Jetson GPU 故障；当前只是没有核验完整 GPU 推理工具链。** 现有 ONNX 包装器显式使用 `CPUExecutionProvider`，即使未来加入 GPU 库，也不会自动转成 GPU 推理。

宿主 `/opt/ros` 不存在，`roscore`、`rosversion`、`ros2` 均不在 PATH，dpkg 未匹配 `ros-*`，当前 Python 未找到 rospy/rclpy。Docker 实际查询结果：

- `docker ps` 和 `docker ps -a` 都为空，没有现存容器。
- 只有镜像 `yahboomtechnology/ros-foxy:4.0.5`，大小 19,053,876,915 bytes，ID `sha256:0e2a3a0c7df2a41235bb9e8c760f993795313cddd6f3992f73a817ec8ef4763c`。
- 本轮没有启动镜像或复核镜像内包/源码；镜像内部 Navigation2、Cartographer、SLAM Toolbox、RTAB-Map 和厂商驱动的描述仅来自 2026-09-05 历史审计。
- OLED 与 Docker 服务 active；旧 cloudcore、edgecore、car-panel 服务查询 inactive。未修改其状态。

## 6. 控制、感知、导航和云端实现情况

以下全部是当前工作树的**静态实现检查（E2），不是测试通过声明**。

| 链路 | 真实源码入口 | 实现与当前缺口 |
| --- | --- | --- |
| 启动/Web | `/home/jetson/VehicleCloudCollaboration/run.sh`；`car/autodrive/run_lcc_web.py` → `web/cli.py` → `web/server.py` | Web 子进程管理、启动/停止/急停、状态和图像。run.sh 优先 CAR_PYTHON，其次旧 Pi Conda 解释器存在才用，最后 python3；默认 VEHICLE_PROFILE 仍为 raspbot_pi5，默认不武装电机。源码的 Web 急停后备硬件归零仅在 motors_enabled 时触发 |
| 实时闭环 | `/home/jetson/VehicleCloudCollaboration/car/autodrive/run_onboard.py` → `runtime/onboard.py` | 视频或相机输入、预热、边界/语义、控制门、四轮映射、watchdog、异步诊断和逐次归档；支持 --vehicle-profile、--video、--output-dir、--max-runtime-seconds、--no-run-archive。源码仍需实际入口导入和运行验证 |
| 硬件适配 | `/home/jetson/VehicleCloudCollaboration/car/control/vehicle_control/{base,factory,profile}.py`、`platforms/{raspbot,rosmaster}.py` | 共用逻辑轮序 FL/RL/FR/RR；树莓派 I2C Ctrl_Muto，Jetson 串口 set_motor；限幅、轮序/方向配置、停止、云台、标定门禁已写。Jetson native command_limit=100；轮命令不是已标定的 m/s |
| 基础感知/LCC | `/home/jetson/VehicleCloudCollaboration/car/autodrive/perception/{outer_loop,perspective}.py`、`control/lane_centering.py` | 黄色边界/HSV-Lab 路面、边界拟合及历史平滑、单边界宽度回退、中心线和横向/航向控制。当前是固定场地外圈控制；没有已核验的通用地图导航或雷达障碍物输入 |
| 语义融合 | `/home/jetson/VehicleCloudCollaboration/car/autodrive/perception/{yolopv2_fusion,yolopv2_onnx}.py` | TorchScript/CPU ONNX、只算可行驶区、异步最新帧队列、结果年龄和重叠门限、与传统走廊交集/校验、FP32/INT8 直道选择及切换结果排除。实现存在；Jetson profile enabled=false，权重和后端未就绪 |
| 本地安全 | `/home/jetson/VehicleCloudCollaboration/car/autodrive/control/drive_runtime.py` | SafeWheelDriver、PerceptionMotionGate、CornerContinuationGate、CommandWatchdog；低置信度/越界/黄线/超时停止及稳定感知恢复门。当前公共配置 watchdog=0.40s、maximum_inference_time=0.25s、恢复4帧；没有本轮物理停车时间/距离证据，底层阻塞或吞异常仍需验证 |
| 雷达/编码器/IMU导航 | 当前 `/home/jetson/VehicleCloudCollaboration/car/autodrive/` 和 `car/control/` | 搜索和入口依赖未发现串口遥测 getter、雷达、ROS/odom/SLAM/Nav2 的运行集成。SDK 支持接口和外部导航文件不能替代业务集成 |
| 长尾与云端 | `/home/jetson/VehicleCloudCollaboration/car/longtail/classifier.py`、`detectors/*.py`；`car/cloud_client/mock_client.py` | 有 CLIP/YOLOv8/YOLOPv2 加权分类和实际 HTTP OpenAI-compatible 请求/解析能力；mock_client 文件名不表示只会 mock。但当前 onboard 入口没有调用分类器/CloudClient，也未找到从新鲜语义 mask 时序突变到云端风险建议再到本地仲裁的整条链路 |

表中缩写文件位置均位于同栏给定的绝对项目目录下。两个必须区分的实现细节：

1. `/home/jetson/VehicleCloudCollaboration/car/longtail/detectors/yolopv2_detector.py` 的 `width_jump`、`center_jump` 是**同一帧内不同图像行**的宽度/中心差分，不是相邻时间的 mask 变化率。边界历史平滑、语义结果 freshness 也不能替代论文计划中的时序候选触发器。
2. 云端旧接口主要返回 left/right，并映射 lane-left/lane-right。当前目标需要结构化场景/风险/建议、请求时间和响应过期验证、本地停车/恢复仲裁；源码中未发现其已接入 onboard。旧接口/测试不能证明新目标已实现。

### Jetson 实际配置选择的影响

配置真实路径：

- 公共配置：`/home/jetson/VehicleCloudCollaboration/car/autodrive/config/onboard_runtime.yaml`
- Jetson profile：`/home/jetson/VehicleCloudCollaboration/car/control/vehicle_control/profiles/rosmaster_jetson.yaml`
- 旧 Pi 透视：`/home/jetson/VehicleCloudCollaboration/car/autodrive/config/onboard_calibration.yaml`

从 `profile.py` 和 YAML 静态解析/递归合并可确认：显式选择 `rosmaster_jetson` 时，profile 覆盖 camera=0/640×480/20fps、关闭云台初始化并清空角度、YOLOPv2.enabled=false、perspective.calibration=null，输出根变为 `/home/jetson/VehicleCloudCollaboration/outputs/onboard_runtime/rosmaster_jetson`。轮组参数为四轮基准 10、pwm_limit=20、minimum_moving_pwm=5，均被 profile 注释标为 dry-run 占位；motion_calibrated/gimbal_calibrated 都为 false。上层正常实车输出会先被门禁拒绝。

`motor_order=[0,1,2,3]`、`wheel_signs=[1,1,1,1]` 尚未实测。公共算法和 safety 参数仍继承旧 Pi 设置，不能只改 calibrated 标志就认为 Jetson 完成标定。

不显式选择车型时，默认仍为 `raspbot_pi5`，该 profile 标定标志为 true，会选择 Pi I2C、25度云台及旧透视。**不应在当前 Jetson 上原样运行默认入口。** 本轮既没有更改默认项，也没有触碰 I2C。

### 外部 ROS 导航材料

- `/home/jetson/x3/docker-compose.yml` 配置 `ROBOT_TYPE=x3`、`RPLIDAR_TYPE=s2`、`ROS_DOMAIN_ID=126`，启动 `ros2 launch hj_nav_launch ybcar_nav_launch.py`，挂载 `/dev`、privileged/host network。
- 它引用 `huajuan6848/x3-54demo-ros-foxy:0.0.11-jetson`，**本轮 docker image ls 中没有该镜像**，也没有对应容器。因此保留 Compose 文件不等于导航可启动；其自定义 hj_nav_launch 内容本轮不可复核。
- `/home/jetson/dwa_nav_params.yaml` 是 Nav2 参数：AMCL differential、DWBLocalPlanner、map/odom/base_footprint、/scan LaserScan、局部/全局代价地图；参数文件里 max_vel_x=0.12、max_vel_theta=0.6。没有证据证明当前由任何节点加载，参数值不是实车测量。
- `/home/jetson/maps/{yahboomcar,zhiyuan}.{yaml,pgm}` 存在，YAML resolution=0.05、origin=[-10,-21.2,0]；地图的采集 run、版本、场地对应关系和 TF 外参未知。
- 本轮没有启动 ROS graph、读 topic、运行 ros2 launch/Compose、执行导航目标或扫描流。LCC 直连串口与未来 ROS 导航必须先明确唯一底盘控制者，避免争用设备或并行下发。

## 7. 权重、原始数据、日志与标定

| 真实路径 | 实际检查结果 | 证据解释 |
| --- | --- | --- |
| `/media/pi/FSDBY/weights/yolopv2_drivable_fp32.onnx`、`yolopv2_drivable_int8_u8s8.onnx` | 公共配置引用，两个文件均不存在；/media/pi 未挂载；/media/jetson、/media/nomachine 的可见目录为空 | Jetson 关闭 YOLO 可暂避依赖；启用语义前必须取回/确定权重、来源和哈希 |
| `/media/pi/FSDBY/weights/{clip-vit-base-patch32,yolov8n.pt,yolopv2.pt}` | 长尾源码默认路径；本轮未找到对应权重 | 默认不是现机部署路径；.env 不存在，.env_example 存在，仅检查其变量名。未执行自动下载或客户端 |
| `/home/pi/miniconda3/envs/car/bin/python` | 不存在 | run.sh 已有回退，不再无条件使用它；旧 README 直接命令仍不能照搬 |
| `/home/jetson/VehicleCloudCollaboration/{outputs,weights,models,data}` | 这四个根目录本轮均不存在 | 没有本机逐次原始录像/CSV/status/calibration frame 的现有入口，不等于其他主机没有记录 |
| `/home/jetson/VehicleCloudCollaboration/readme.assets/video.mp4` | 可读，1,314,711 bytes；mtime 2026-08-04 14:37:21 +08 | 历史 README 演示资产；本轮未解码，缺 run ID/对应配置/同步原始日志，不能当正式实车证据 |
| `/home/jetson/VehicleCloudCollaboration/car/test/closed_loop_report.json` | 可读，7,317 bytes，顶层 ok=true；没有时间/run ID 字段；cloud_decision 为 inprocess 类演示 | 旧静态图+假执行器测试产物，不证明真实云端、Jetson 电机/导航或停车效果 |
| `/home/jetson/VehicleCloudCollaboration/car/test/test_image.jpg` | 可读，195,998 bytes | 测试图片，不是新采集 |
| `/home/jetson/VehicleCloudCollaboration/car/longtail/dataset/Long-tail` | 直接文件54个，共13,172,815 bytes，可读 | 当前目录清点，不重新验证标签或统计实验性能 |
| `/home/jetson/VehicleCloudCollaboration/car/longtail/dataset/Non-long-tail` | 直接文件744个，共71,723,490 bytes，可读 | 同上；没有 run 分组/独立性证明 |
| `/home/jetson/VehicleCloudCollaboration/car/autodrive/config/onboard_calibration.yaml` | 可读，770 bytes，calibrated=true；camera_pose pan=25、640×480，source_image 指向缺失的 outputs/onboard_capture/onboard_calibration_frame.jpg | 属于旧 Pi 标定；Jetson profile 已显式设 calibration=null，不能用该标志声称 Jetson 标定成功 |
| `/home/jetson/maps` | 两个 YAML，各关联369,725 bytes PGM | 外部历史地图，未绑定当前场地或 run |
| `/home/jetson/{agent.log,bagent.log,gossip_logs.txt,log,log1}` | 文件存在，分别253,359/162,895/14,143/70,707/23,311 bytes；仅 stat | 老工程日志，未读内容，不把它们归为本项目新实车日志 |

仓库包含被忽略项的后缀筛选仅找到上述 README MP4，没有 `.onnx/.pt/.pth/.engine` 权重或 `.avi/.bag/.mcap/.csv/.jsonl`。在 Videos、Downloads、Documents、Desktop、media、maps、x3、Rosmaster-App 等入口做了有界盘点；没有扫描全盘、Docker 镜像层或所有缓存，不能排除其他未查位置的资产。

`/home/jetson/VehicleCloudCollaboration/car/autodrive/YOLOPV2_ONNX_EXPERIMENT.md` 描述旧 Pi 的 run `20260803_211052_468830_hardware` 及其 raw.mp4/模型哈希。本轮本机缺对应 `outputs/onboard_runtime/runs/` 和模型，故其速度、IoU、多圈等数字只保留为 E3 历史描述，未复算、未转记成 Jetson 成绩。

现有运行归档代码计划输出逐次 `raw.mp4`、标注/鸟瞰视频、CSV、status、公共及解析后配置、车型 profile 与透视快照。这是源代码的产物约定，本轮没有生成这些实验文件。

### 当前可复核的版本指纹

| 对象 | SHA-256 |
| --- | --- |
| 初始 `AGENTS.md` | `00b072dbf6b1201c5e78a88143ec4b6e35961aa2095df991e98d3a271f3663fc` |
| 初始 `PROJECT_INDEX.md`（后续只追加摘要） | `a6a503daf7fcfdc02e86d4b6fda6092bc56b8149ef743df1c8848af0ac92948e` |
| `coordination/2026-10-08-car-baseline.md` | `2e63b12e04dbe251a8c0c182234a8b71502fb145a9c373254659d5f6eec6e787` |
| `car/autodrive/config/onboard_runtime.yaml` | `b050a42b4f563049062f712c4e62bd6575014ca04907baaea558ae1f979bb435` |
| `car/autodrive/config/onboard_calibration.yaml` | `0289a9e18cc2dfe5596dca6a6e734ab8aa1ded724255ce1373b60d6c4daff087` |
| `car/control/vehicle_control/profiles/rosmaster_jetson.yaml` | `96689b019e3ddbaa584ac5ca9fe8484967318fd86511f6bcd46d9caa240a7cba` |
| 既有 `git diff --binary` 原始字节 | `9523bed7dabd72270b53dd79c2dae73d6c4895097486e95d7139b39903b51478` |
| 旧 `car/test/closed_loop_report.json` | `475692721f3e52a485b182bdc7475f8ab9955ca8d7004286d9fb251350390991` |

未跟踪适配文件不包含在 git diff 指纹中，其 SHA-256：

```text
car/control/vehicle_control/base.py                       8196163956c3936611c61a676f6daf0c185c94d5ec9ae9e259bcc9b74eb6c9e1
car/control/vehicle_control/factory.py                    d4657a51e4ab857024b3f033a05d696260b36923749bb700e2171397754ae851
car/control/vehicle_control/profile.py                    98dd815f77012ccfcddfc21b62a0d2bcdd212a48cb226b9c14a78a8e02962ad2
car/control/vehicle_control/platforms/__init__.py          9d26eccc4b8fc8f5140b1cb88524e7e0c2fec7f70044f376f0e2557fd58a394e
car/control/vehicle_control/platforms/raspbot.py           2ce564d22a336be847cee5adaeb4e28218d32c8b23e2ddfc241191800e460591
car/control/vehicle_control/platforms/rosmaster.py         20d35f09b625ab77dde16d0bbaa5ec8a23dc0ba9cbf497be2555ab7557a6ce4c
car/control/vehicle_control/profiles/raspbot_pi5.yaml      13a4418b068118534dc8e0e0ac332d03fdf1cd33be0534151552e58d75427a17
car/control/vehicle_control/profiles/rosmaster_jetson.yaml 96689b019e3ddbaa584ac5ca9fe8484967318fd86511f6bcd46d9caa240a7cba
```

## 8. 缺失和未知项

入口材料没有缺失。下列材料/验证不足实际存在：

1. 当前 Jetson 的机械型号/轮型实物确认、轮序/方向/死区/直行 trim、云台通道/零位、相机姿态和独立透视标定。
2. 当前车型的编码器/IMU真实上报、雷达具体型号/波特率/扫描、深度流与时间戳。仅有枚举和 SDK getter。
3. 当前解释器的完整依赖、YOLOPv2 FP32/INT8 和其他所选模型权重、模型/config/source 的部署清单；没有本轮 GPU 推理证据。
4. 历史 Pi 的 raw.mp4、原始 CSV/status/配置、原始模型及独立备份路径；本机没有正式 run 清单。
5. `/home/jetson/VehicleCloudCollaboration/car/control/vehicle_control/controller.py` 不存在，但 `/home/jetson/VehicleCloudCollaboration/car/test/closed_loop_test.py` 仍导入它。该测试当前不能按 README 原样执行；旧 ok=true JSON 不覆盖此源码缺口。
6. 本地论文完整索引/实验计划/正文没有同步；本仓库的 `EXPERIMENT_VALIDATION_PLAN.md` 和 `papers/MATERIAL_INDEX.md` 不存在。它们不阻塞本轮盘点，后续确定正式实验、路线或论文结论时需要电脑端提供相应版本和来源。
7. ROS 保留镜像内当前确切源码/包与启动行为、外部导航配置的可复现版本和场地对应关系。未启动容器复核；旧 Compose 的自定义镜像缺失。
8. 目标时序触发、非阻塞减速/停车、异步结构化云端输出、本地响应有效性与安全仲裁的完整集成和端到端实车证据。

这些缺口已持久化于本报告，未自动向其他聊天发送请求。

## 9. 按依赖顺序的下一步调通建议

下列是后续任务顺序，不表示本轮已执行或已获得运动/安装授权。

| 顺序 | 依赖与建议工作 | 完成判据/后继 |
| --- | --- | --- |
| 1 | 以当前 HEAD+脏工作树指纹建立可复核代码基线；明确使用 rosmaster_jetson，核对自定义环境与旧数据来源。保留现有适配，不重置或直接改为已标定 | 车型选择和解释器一致、源码/配置可追溯；才能评估环境与算法 |
| 2 | 先执行无硬件输出的配置自检和 mock 单元测试，确认 Python 3.8 注解适配、深度合并、限幅/轮序/门禁/Web命令；依检查结果单独安排所需依赖环境，安装方案另列 | 获取真实测试输出，未标定车型必须拒绝正常电机输出；不能用 metadata 代替运行兼容性 |
| 3 | 后续安排纯相机检查，显式 Jetson profile、云台初始化关闭；确认 RGB 朝向/640×480/实际FPS/新鲜帧行为，再按现场安排核验云台并生成独立 Jetson 透视文件 | 图像、机械姿态、标定来源帧和配置一一对应；不沿用 Pi 25度/透视 |
| 4 | 先让经典黄色边界 LCC 在录制副本中完成 dry-run，检查失帧/边界消失/误差门/恢复门/watchdog；使用独立输出目录并保留原始素材 | 感知和虚拟轮命令可复现，日志有输入run与时间；不声称物理停车 |
| 5 | 如需语义链路，取回有哈希的 YOLOPv2 FP32 权重及兼容 CPU ONNX Runtime；先验证 FP32与LCC融合、掩码几何/重叠/结果年龄。随后单独评估 GPU 或 INT8；不要一开始并用所有长尾模型 | 当前 Jetson 模型确实运行、结果可追溯；既有CPU包装器需独立改造才会使用GPU，自适应INT8需重校验 |
| 6 | 在用户安排的实车实验中做架空单轮/四轮顺序、方向、最小有效命令和急停，再落地最低安全速度直行trim及弯道。只在取得实际证据后更新 profile 标定状态和参数 | 保存外部/车载录像、SDK下发与反馈、配置和run；证明物理执行、停车与恢复，再扩展现场闭环 |
| 7（可并行准备，依赖2–3） | 若需要雷达导航，先在执行器输出隔离的方案下读取雷达/编码器/IMU/深度，确定驱动、坐标、时间同步和TF；再核验保留ROS镜像源码。使用新而独立的配置，不直接启动旧 privileged Compose | 真实 /scan、odom、imu 数据和帧/单位对应；确认底盘唯一控制者，然后才讨论建图、定位/导航实车验证 |
| 8 | 在新鲜语义输出、本地安全闭环及原始run保全后，实现/核验目标时序触发→本地响应→异步云端场景/风险/建议→响应校验与本地仲裁→稳定恢复；先无执行器故障注入，后受控低速重复实车 | 响应超时/错误/过期不得突破本地安全门；云端不直接设置PWM；保持试开发证据与正式实车结果分开 |

第7项是条件分支，雷达/SLAM不必成为纯视觉LCC的前置条件；是否进入主实验仍需与电脑端完整实验计划核对。

### 候选验证入口与副作用（本轮未运行）

经过源码审阅，以下可作为后续第一组检查：

```bash
cd /home/jetson/VehicleCloudCollaboration
python3 -B car/autodrive/tools/pi_self_check.py --vehicle-profile rosmaster_jetson
python3 -B -m unittest car.test.test_lane_centering car.test.test_camera_transform car.test.test_camera_gimbal car.test.test_hardware_mapping car.test.test_lcc_web
```

不加 `--camera` 的自检仅查配置、基础依赖、路径、权限和空间，不构造硬件；也不检查 YOLO 模型真正可推理。上述 unittest 源码使用合成输入、FakeController/FakeChassis/FakeSession、临时目录和测试子进程；未发现需要真实相机或串口的调用路径，仍需在选定环境取得实际结果。未执行这组测试，不宣称通过。

| 入口 | 具体副作用/注意点 |
| --- | --- |
| `pi_self_check.py --camera` | 打开VideoCapture并读取一帧；不会初始化云台，但已不属于本轮被动枚举 |
| `run_onboard.py --vehicle-profile rosmaster_jetson` / `VEHICLE_PROFILE=rosmaster_jetson ./run.sh` | 读取相机、跑算法、创建运行输出/Web服务；电机默认关闭、该profile关闭云台；不是单纯只读盘点 |
| `run_onboard.py --video <原始录像副本> --vehicle-profile rosmaster_jetson --output-dir <新目录> --no-run-archive` | 不启用电机时可作离线算法验证；会写输出且运行感知，缺原始录像时不应用README演示替代正式run |
| `check_wheel_directions.py` | 真实电机脉冲；需要现场架空与当次运动安排，不能因为在tools/test附近就直接执行 |
| `align_camera_gimbal.py` / `capture_onboard.py` | 可能移动舵机、打开相机并写标定素材；先核对profile和确认参数，不能认作硬件只读 |
| `calibrate_perspective.py` | 图像/GUI处理及写标定，可能按 --force 覆盖；本轮未运行 |
| `closed_loop_test.py` | 已有缺失控制器导入；默认还会加载模型、访问云端并覆盖旧report。即使有FakeChassis也不是本轮合适的只读检查 |
| Rosmaster SDK/厂商示例、`ros2 launch`、`docker compose up`、`i2cdetect -y` | 构造/启动/探测可能写串口、使能舵机、驱动底盘、启动雷达或发送I2C事务；本轮均未执行 |

## 10. 完结校验

2026-10-08 15:31:32（Asia/Shanghai）执行标准库哈希/状态校验脚本，退出码0，断言全部通过：

- 开始快照的1,688个普通文件无缺失；唯一内容改变为 `PROJECT_INDEX.md`。把新增摘要分隔符前的原文重新计算SHA-256，与初始索引哈希完全一致，确认是追加。
- 唯一新增普通文件为 `/home/jetson/VehicleCloudCollaboration/coordination/2026-10-08-car-baseline-report.md`。旧未跟踪条目没有消失，Git status只增加该报告条目。
- HEAD未变；原有26个跟踪文件的 `git diff --binary` SHA-256仍为 `9523bed7dabd72270b53dd79c2dae73d6c4895097486e95d7139b39903b51478`；未跟踪业务源码、旧报告、视频、配置、manifest和技能内容哈希均保留。
- 文档检查确认三份入口齐全、报告存在、引用的索引/协调材料可读、代码围栏成对；本节只补充已实际取得的文件保留性证据。

没有测试套件、模型/相机/雷达运行或实车结果可汇报。只读盘点完成不等于Jetson系统调通，未进行Git提交、推送或跨聊天发送。

