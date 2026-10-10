# 小车硬件验收进展

日期：2026-10-08（Asia/Shanghai），状态：reported，等待用户统筹验收。

用户当次明确要求本聊天按硬件清单、底层修复、真实功能验证、逐项验收顺序执行；不并行推进云端、SLAM、论文算法。

执行器授权来源：用户回答“可以架空,可以急停,所有动作都可以测试”，随后明确回答“已架空并固定，现场有人可急停”。仅本轮按有界低输出点动/停止/云台测试使用，不扩大为地面自动行驶授权。

主仓库HEAD：33a5a712b97af15feadc11f4751e60edaa05dc08。原26个跟踪文件保持原样；本轮三个已有未跟踪底层文件作必要修复，并新增串口会话及两份测试。应用创建隔离工作树 `/home/jetson/.codex/worktrees/car-hardware-acceptance/VehicleCloudCollaboration`，分支 `codex/car-hardware-acceptance`；复用原有48项dirty文件快照。基线与阶段证据在 `outputs/hardware_acceptance/20261008/`。

实施计划：`docs/superpowers/plans/2026-10-08-car-hardware-acceptance.md`。执行方式：本聊天实现和测试。

## 已完成

- 保存原有binary patch，sha256 9523bed7dabd72270b53dd79c2dae73d6c4895097486e95d7139b39903b51478；保存48文件的内容hash和必要原始副本。
- Task1：参数门禁/整体姿态校验/有限输入与时长/停止三次尝试及错误传播完成；RED日志 safety-red.log，23项（含既有14项）GREEN，见 safety-green.log。
- Task2：新增严格Rosmaster串口会话，不使用厂商SDK构造；无启动TX、短写/故障门禁、坏帧恢复、初始invalid、freshness、小端反馈、伪终端独占验证。15项transport GREEN；整体127项GREEN，见 regression-after-transport.log。
- 兼容性决策：现有注入controller默认仍由组件关闭；显式共享须设置owns_controller=False，调用者负责poll和最终关闭。已通过原有云台关闭测试及共享测试。
- 根因补充：旧NaN轮命令被clamp映射到+100；无限渐变时长可无限循环。首次mock测试因此被终止（未连接真实硬件），随后给测试加入退出条件，保留有界RED证据并修复有限输入门禁。
- 已实际检查保留ROS镜像内Astra/sllidar可执行文件及单独launch，见 ros-driver-inspection.txt；没有运行完整厂商导航launch。

## 当前交付

- RGB、深度、雷达持续有效数据及架空电机/编码器证据齐备；云台转动验证失败，是否实际安装仍未知。
- 六文件增量已审查并按原始hash合回主工作树；主工作区134项指定测试通过，原有其他45项dirty文件保持原样（索引最终只追加事实）。
- 报告：`coordination/2026-10-08-car-hardware-acceptance-report.md`；manifest：`coordination/2026-10-08-car-hardware-acceptance-manifest.json`。
- 下一负责人为用户统筹，核对证据并形成验收意见及标定任务单；本车端聊天随后按现场安排完成机械与相机标定。

## 后续实测更新

- 串口被动20s：32650 bytes、1949帧、TX0、checksum/length错误0，四类周期遥测约24Hz，motion末值0/电池12.3V，MPU类原始数据有更新，四路编码器静止0。
- 显式非永久身份查询：固件major3/minor5；实际车型回应fffb051501001b带reserved字节，首实现发现长度缺口后literal RED→GREEN修复，兼容1/2字节；未以静态元数据过期否定已收到身份事实。
- RGB30s：874帧、读失败0、640×480、实际约30.006Hz；请求20fps未让设备变20fps，使用帧时间作为证据。
- 雷达20s：169扫描、约10.023Hz，DenseBoost 3240points/scan，约1980有效点；SDKhealth OK、FW1.02/hardware18；实际model字节0x71，保留raw identity，不以USB bridge/旧compose替代型号。
- 深度Astra、serial ACR7C33004M：首次554帧640×480/16UC1，但全部0（失败）；IR联合启动和后续depth-only启动均失败，保留driver/console日志；用户已拔插USB，新节点001/014，正在再次验证。初次容器目录权限错误保留独立console，非设备验收。
- 用户提供两张实际照片，确认麦克纳姆轮及Astra/RPLIDAR安装，已架空；用户确认“四个轮子都动了,而且正常”。原生M1..M4正反点动各0.4s，tick分别+228/-309、+404/-476、+375/-440、+279/-344；其他通道不变，每次停车后velocity0；物理FL/RL/FR/RR位置尚未独立确认。
- 云台三轮输出请求已获用户明确授权；第一次录像图像配准无明显视角变化，用户后续表示好像没看到Astra动；不能把串口发送成功认定云台通过，也不能把Astra固定支架假定为电动云台。
- 新增停止/渐变竞争用例先失败后修复：停止取消旧渐变，stop_event取消时写零；整体130项通过，见regression-after-ramp-stop.log。
- 请求独立只读代码复核，由review技能要求的hardware_review代理进行；不并行实施其它业务，不让评审者连接设备。

## 最终复核更新

- 用户重新插拔后深度20s：553/553帧有有效像素，约29.315Hz、56.34%–60.60%有效。停止后重开10s：262/262帧有效，29.553Hz；先前全零与启动失败证据保留，不声称长期启动问题彻底解决。
- 用户最终明确表示没有观察到云台转动，询问是否没有云台；照片不见明确双轴舵机。记录硬件存在性未知、转动验收失败；固定相机标定可继续。
- 独立审查与补回归修复：遥测锁等待freshness、公共工厂借入所有权、错误标定阻断全零停车、关闭竞争恢复非零。主工作区134项，1.857s，退出0（main-regression.log）。
- 主工作区5s被动TX0，再两条只读身份请求＋三次全零：固件3.5/type1/reserved0，579帧、校验/长度0错误，最后TX全零、反馈velocity0、closed=true。原始数据serial-main-verified/。
- 统一多传感器产品CLI本轮未新增，限时采集脚本及复现参数保存在证据目录；这是明确的实施调整，不宣称统一SDK完成。
