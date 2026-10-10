# Jetson YOLOPv2 场地部署与验证报告

- 日期：2026-10-08，Asia/Shanghai。宿主日志/run ID使用America/New_York，不能按文件名误判为北京时间。
- 状态：`reported`。最新代码、GPU环境、YOLO主感知模式和电机禁用完整运行已验证；场地感知及地面自动驾驶尚未验收。
- 执行者：本聊天，远程小车车端开发；主机`jetson-desktop-car-2`，项目`/home/jetson/VehicleCloudCollaboration`。
- 人类任务：拉取最新代码、配置必备环境、YOLOPv2在GPU上运行并验证、必要时修改LCC。用户随后明确云端暂缓，待其选定；没有发出真实云推理请求或向其它聊天发送消息。
- 本次没有电机、云台、雷达转动命令，没有地面行驶；真实硬件操作仅RGB相机采集。上次架空测试是独立的历史验收，不能替代新场地运动安排。

## 交付与版本

主工作区已从`33a5a712b97af15feadc11f4751e60edaa05dc08`快进到远端`6bf86933ded040c64613515497eb75c9f77fe371`（本次fetch时origin/main）。最新远端修改主要是云接口。本轮没有新提交/推送。

原55项dirty文件先备份至`outputs/field_yolopv2/20261008/before/`，补丁`pre-update.patch`；README局部stash后快进、三方恢复，原stash保留。实施复用隔离工作树`/home/jetson/.codex/worktrees/car-hardware-acceptance/VehicleCloudCollaboration`，分支`codex/jetson-yolopv2-field`。原26个跟踪文件改动及未跟踪平台适配保留。本轮15个实现/配置/测试/使用说明文件合回前，7个已有文件均与实施前SHA256一致；其余目标确认不存在。记录`integration-record.json`、`implementation-incremental.patch`、`before-integration/`，均在本次证据根。

直接入口：`/home/jetson/VehicleCloudCollaboration/run_jetson_yolopv2.sh`。使用说明：`docs/jetson-yolopv2.md`。新profile：`car/control/vehicle_control/profiles/rosmaster_jetson_yolopv2.yaml`。执行：

```bash
cd /home/jetson/VehicleCloudCollaboration
./run_jetson_yolopv2.sh --max-runtime-seconds 30
```

默认真实相机、GPU、虚拟控制，30秒后停止；不自动打开底盘、初始化云台或调用云端。主工作区仍有未提交改动，不能只靠HEAD复现本轮；需结合增量补丁、源码快照和manifest。模型/venv/原始数据Git忽略。

## 实际环境验证（E1开发证据）

| 项目 | 本次实际结果 | 依据 |
|---|---|---|
| 计算平台 | Orin NX，Ubuntu20.04，L4T R35.3.1，Python3.8 | 既有硬件报告及本次GPU设备查询 |
| GPU运行库 | CUDA11.4，cuDNN8.6，cuBLAS/cuFFT/cuRAND/cuSOLVER/cuSPARSE、NVTX/NVRTC、OpenBLAS已安装 | `gpu-packages.txt`和各`*-install.log` |
| Python | 独立venv，NVIDIA torch2.0.0+nv23.05，复用已有NumPy/OpenCV/PySerial | `venv-freeze.txt`、`gpu-environment.json` |
| 真GPU运算 | cuda:0/Orin矩阵乘法，CPU对照最大绝对误差0 | `gpu-environment.json` |
| 权重 | 官方YOLOPv2 V0.0.1 TorchScript，156380200字节 | `model-manifest.json` |
| 模型运行 | 参数和输入为CUDA FP16，道路/车道线/检测头均运行 | `yolopv2-gpu-verification.json`、输入对比及最终运行 |
| 功耗/时钟 | 既有20W模式，GPU最高408MHz；本次没有改模式或锁频 | `power-mode.txt`、`clocks-readonly.txt` |
| 云端 | 暂缓，`.env`权限600并Git忽略，未配置真实密钥 | 仅记录状态，不归档文件内容 |

venv：`/home/jetson/VehicleCloudCollaboration/outputs/field_yolopv2/venv`。权重：`/home/jetson/VehicleCloudCollaboration/car/longtail/models/yolopv2.pt`，SHA256 `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c`。

依据[NVIDIA官方Jetson安装说明](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)选择对应wheel；依据[YOLOPv2作者release](https://github.com/CAIC-AD/YOLOPv2/releases/tag/V0.0.1)下载权重。没有安装整套通用requirements、ROS或TensorRT，没有重启业务服务。apt新增运行库，未升级或卸载已有包；隔离环境继承系统基础库，因此重建仍需记录宿主版本。

## 感知与LCC实现（E2代码，结合真实输入开发验证）

新增`perception.mode=yolopv2`：仅从模型道路、车道线和检测头建立候选通道；不运行黄色/绿色阈值路径，也不使用旧弯道历史续行。LCC仍负责中心线到有界控制提案的转换；YOLOPv2本身不是直接输出转向/PWM的端到端驾驶网络。

- 全模型、CUDA FP16、保留相机比例；新profile为320输入，实际张量`1×3×256×320`，去padding后为240×320掩码。
- 补齐官方trace的三尺度检测头解码与分类别NMS，不依赖torchvision。公式参考作者`utils/utils.py`；已有已知解码/NMS/坏数据用例，未完成所有官方输出逐元素数值对照。
- 模型缺失/异常、非有限输出、结果过期、空或异常全图道路、缺少近车与前视道路支持、拟合轨迹离开通道时停车。检测框在图像中遮挡前方可触发停止；这不是米制安全距离或通用避障保证。
- 近车支持与异常掩码在原相机坐标检查，避免透视黑边掩盖全图误检；再检查投影通道。语义模式禁止旧形态学闭运算跨越模型车道线，拟合曲线及实际控制前视范围必须留在通道内。
- 恢复要求5次不同模型观测，不能靠重复消费同一缓存凑数；0.3秒年龄从原帧采集时刻算起，含排队/推理/消费延迟。坏结果仍立即停止。
- 长尾检测包改为按需导入，YOLO运行不再强制依赖CLIP/YOLOv8。
- 同一进程重复构建模型时曾触发NVIDIA wheel的原生线程池异常退出；已避免重复设置相同线程池，双模型实测完成。原异常日志保留，不把它写成通过。

独立审查发现透视绕过、固定裁切不兼容、车道线重新填平、控制前视越出有效掩码等问题，均先复现失败再修复。主工作区188项相关测试通过（`main-final-tests.log`）；其中云接口是本机HTTP模拟测试，未测试真实云服务。启动脚本`--help`、`bash -n`、`git diff --check`通过。测试不证明地面停止距离或导航成功。

## 真实相机、GPU与运行结果

证据根：`/home/jetson/VehicleCloudCollaboration/outputs/field_yolopv2/20261008/`。

原始相机采集：`field-camera/raw.avi`，182帧，采集窗口约10秒（含开启过程），640×480；逐帧时间在`field-camera/timestamps.json`。AVI标称30fps，不能以视频播放时长替代真实采集时间。另有下面两次完整运行的原始MP4和逐帧CSV。

对同一真实静止场地图像`frame-0150.png`，同步CUDA后计时，完整预处理、三个头、后处理均包含；仅640 FP16为5次热身+20次测量，其余3次热身+10次。不是论文正式实验或场地准确率：

| 输入 | 精度 | 平均处理率 | 延迟中位数 | 观察 |
|---|---|---:|---:|---|
| 640保留比例 | FP16 | 6.59fps | 151.5ms | 近车道路严重漏检 |
| 640保留比例 | FP32 | 5.61fps | 178.1ms | 同样漏检，不能归因为半精度 |
| 480保留比例 | FP16 | 9.57fps | 105.0ms | 近车道路恢复，仍有绿色区域误检 |
| 320保留比例 | FP16 | 17.05fps | 58.6ms | 选作当前调试默认，仍需场地验收 |
| 640旧拉伸输入 | FP16 | 7.66fps | 129.8ms | 对照项，未作为新profile |

对比记录：`input-comparison.json`、`field-control-comparison.json`；图像`field-camera/overlay-320-native-fp16.png`等。FP32/FP16同帧掩码一致性见`derived-checks.json`，仅是精度路径对照，不是真值IoU。部分形状/精度切换触发nvFuser优化回退警告，进程正常完成；CUDA参数与实际运算已核验，没有据此宣称TensorRT优化完成。

**初始640运行**：`live-runtime/runs/20261008_074850_305446_dryrun/`，20秒，394控制周期，119次完成推理，394次全停（172过期、222近车支持不足）。现场问题被保留，后续没有降低0.3秒新鲜度门槛来放行。

**最终主工作区320运行**：`main-live-320/runs/20261008_080239_047274_dryrun/`，20秒，391控制周期，268次完成推理；338次虚拟右转提案、53次停止，结束后全部虚拟PWM归零。首47周期因启动期结果过期停车，剩余周期模型年龄P95=0.1633秒、最大0.1787秒；整个运行年龄P95=6.0555秒，不能隐藏启动开销。异步最新帧队列替换124项待处理任务，视频归档391帧、无归档丢帧、模型error=null。模型完成数含一次启动热身，不应直接解释为精确稳态fps。记录`main-live-320-summary.json`。

这证明当前配置能产生现场模型感知和虚拟控制，**不证明虚拟右转就是正确行驶路线**。当前画面模型对绿色区域有误检，未有人工标注指标；三张离线图来自同一次静止采集，不是三个独立场景。未安排真实障碍目标，本次检测框为空不能算目标检测/避障通过。

## 未验证项与依赖顺序

1. **场地感知验收**：本车端聊天在用户现场配合下，先采不同位置/弯道/边界/障碍的静止图像，逐帧核对道路、车道、中心线和停止决定。交付带run ID的场地感知验收集与失败清单。当前仅一个静止位置，不能进入无人看护行驶。
2. **模型适配评测**：由用户统筹明确分配给“本地实现与评测”聊天，使用本报告的原始数据和权重版本建立人工标注、独立位置划分与误检漏检评测；若仍不达标再决定权重适配/训练。交付场地模型评测与候选权重，不把相邻帧当独立测试样本。本文只是交接建议，未发送任务。
3. **车辆与相机标定**：仍由本车端聊天执行，用户提供当次架空固定/急停及地面试验安排，补物理轮序/前进方向、有效低速PWM/停止行为、固定相机姿态与透视。交付校准profile、标定文件、原始反馈。当前`motion_calibrated=false`、`gimbal_calibrated=false`、透视null；先前四轮架空正常不能替代这些数据。相机按固定支架处理，未确认真实云台机构。
4. **封闭场地最低安全速度验证**：上面感知、控制与现场条件验收后，再由本车端聊天做短距离行驶、停车、恢复，逐项记录通过/失败/未验证。交付《低速试跑验收记录》，含外部视频、车载视频、日志和实际停止距离。
5. **云端**：等用户选定供应商、接口与模型后接入；目前无需提供API Key。深度/雷达/编码器/IMU已有独立硬件验收见旧报告，但它们没有自动加入本次YOLO驱动链路。SLAM/全局导航、时序长尾触发—云端仲裁完整链路仍未验收。

由用户统筹核对本报告；下一步具体车端工作继续交给本聊天，不需先并行推进云端、SLAM或论文算法。
