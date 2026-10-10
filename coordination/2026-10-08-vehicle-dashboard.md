# 小车实时面板任务记录

- 日期：2026-10-08（Asia/Shanghai）；状态：`reported`，面板和开机监控已部署，已做静止与浏览器验收；未做整车重启或行驶验收。
- 目标：扩展现有 LCC HTTP 面板，显示车辆/传感器/感知/控制/云端/目标状态，配置开机启动。
- 人类依据：本聊天用户明确要求实时面板、复用或新建、至少电量/相机/雷达/云端触发与结果/目标/置信度/YOLO、开机自动可用和 HTTP 浏览器访问。此前请求允许更新代码与环境配置；本聊天不据此执行车轮运动。
- 主机：jetson-desktop-car-2；项目：`/home/jetson/VehicleCloudCollaboration`；HEAD：`6bf86933ded040c64613515497eb75c9f77fe371`，main，工作树已有大量未提交修改，未作提交/重置。
- 设计：`docs/superpowers/specs/2026-10-08-vehicle-dashboard-design.md`。本轮新增设计及任务记录，尚未修改面板实现或安装服务。
- 已核对：`web/server.py`、`cli.py`、`index.html`、`run.sh`、Jetson profile、`runtime/onboard.py` 状态发布、Rosmaster 遥测协议、云端 scene schema 与已有 GPU/硬件验收报告。
- 命令依据：`git status --short --branch`、`git diff --stat`、`git diff -- car/autodrive/web/server.py car/autodrive/web/cli.py car/test/test_lcc_web.py`、`git rev-parse HEAD`、`hostname -I`、`systemctl is-system-running`、服务 unit 文件筛选。当前局域网 IP 为 `10.20.30.132`，systemd running，未找到面板服务。
- 更新检查：初次 `git fetch origin` 因沙箱 .git 只读失败；随后经自动审查允许在宿主完成 `git fetch origin && git rev-list --left-right --count HEAD...origin/main`，退出 0，结果 `0 0`，当前 HEAD 与此次获取的 origin/main 一致，无需合并，未覆盖用户修改。
- 参考聊天：只读读取 `01a11a5a-5ba3-7d62-ac39-cd0dd047e99f`，host `remote-ssh-discovered:car-cloud-edge-wifi(gpu)`，标题“小车代码”。最近工作为外圈标定与云端仲裁，快照 latestTurn 为 inProgress。未向其发送消息；本任务不占用其活动设备或同时编辑控制代码。
- 重要限制：电池百分比未标定；云端 schema 没有数值置信度；全局目的地导航未集成；现有数据归档不能作为当前在线状态。开机 HTTP 服务和自动驾驶授权分开。
- 下一步：用户审阅方案后，本聊天完成实施计划、面板/采集协议及服务部署，产出 HTTP 地址与当前真实显示能力清单；静止联合采集避开另一个聊天的正式实验；整机冷启动验收需现场窗口。

## 本轮基线验证

- `python3 -B -m unittest car.test.test_lcc_web`：6 项通过，退出 0；阅读测试源码确认使用临时脚本与临时目录，没有导入或启动硬件控制。仅证明现有过程管理与状态读取回归通过，不代表新面板已实现。
- 宿主 `ss -ltn '( sport = :8080 )'` 无监听；`docker ps --format ...` 无活动容器；只读 /proc/fd 采样未发现当前用户可见进程占用 ttyUSB0/ttyUSB1/video0/video1。该检查是瞬时快照，不承诺随后仍空闲。
- `git diff --check` 退出 0。方案自审覆盖空值/过期/旧缓存、串口与相机唯一拥有者、自动启动与运动授权、云端/导航未接入状态、局域网访问、真实重启验证边界。没有做新产品实现或硬件运行。


## 实施与最终验收（2026-10-08，Asia/Shanghai）

用户原始回复明确“按此方案实施，由你在本聊天完成”；静止采集的自动审批曾拒绝，理由是方案确认不等于本次硬件实验授权。随后用户再次明确“授权本次静止采集及开机监控，不动车轮和云台”。在第二项授权之后才进行实际设备采集与开机采集服务启动。没有向另一聊天发消息、没有真实云请求，也没有给车轮或云台发送运动指令。

- 主工作区最新HEAD `fce828b89b084eda293e83b99c8861f522858d8e`，main，仍有原有及本轮未提交修改；本聊天没有提交或推送。另一聊天推进了远端与云端仲裁代码，本轮刷新隔离工作树并保留其全部增量，仅加遥测接点。实现工作树 `/home/jetson/.codex/worktrees/vehicle-dashboard/VehicleCloudCollaboration`、分支 `codex/vehicle-dashboard`，基底6bf8693；最终复现以主工作区HEAD、已有差异和本轮增量共同为准。
- 合回20个文件，其中3个为既有文件的小幅接点改动，其余为新模块/页面/测试/服务/说明。合回前逐文件匹配最新主工作区SHA256，保存before、增量patch与最终源码，避免覆盖另一聊天对同一运行时的更新。准确列表及哈希见 `outputs/vehicle_dashboard/20261008/integration-record.json`。
- 实际地址 `http://10.20.30.132:8080/`；`GET /api/state`、`GET /api/health`；POST启动操作实测403。仅局域网HTTP，没有配置公网映射或TLS。App浏览器打开请求返回queued，不能把它当作用户已看到页面的证明。
- 四个新服务 `vehicle-dashboard`、`vehicle-perception-monitor`、`vehicle-serial-monitor`、`vehicle-sensors` 均enabled/active；实查NRestarts与MainPID见service-status.txt。未重启整车，开机自动启动配置已验，真正重启后的恢复仍需现场窗口。没有修改既有其它服务；unit校验输出提示宿主既有yahboom_oled权限及snapd旧键警告，本轮未顺手修它们。
- 开机观察路径强制电机禁用、云台初始化关闭、云端关闭、无视频归档；每30分钟重启该观察进程限制日志规模。底盘监控只被动接收。新 `run_vehicle_experiment.sh` 在实验前释放需要的设备、退出恢复原先运行的监控；有真实控制session时遥测借用它，不重新打开串口。其它直接启动工具仍须先停止相关监控服务，参见使用文档。
- 页面复用现有LCC图像/状态，并增加车辆/传感器/模型/云端/目标与系统状态；缺失、错误、过期、跨boot残留均不报在线。模型检测结果额外按原始结果年龄检查。只读诊断页面隐藏启动与急停控件，避免无效按钮误导。
- 电压不换算未标定百分比。云端当前为disabled；面板已兼容最新cloud_arbitration及本轮run/event绑定的请求/响应摘要，但本次没有真实云推理，云端数值置信度不存在。最终目的地/全局导航尚未接入，不将局部前视点当成目的地。

### 真实静止采集（开发证据，不是自动驾驶成绩）

证据根 `outputs/vehicle_dashboard/20261008/`，本轮证据ID `vehicle-dashboard-20261008`。

- 12秒被动底盘采集：TX bytes=0，电压约11.7V，IMU/姿态/四通道编码器有效，日志serial-test.log。部署联合运行快照电压11.5V，最终约11.4V；都是当时样本，不代表固定电量。
- 雷达/深度20秒独立复测：雷达约10.019Hz、3240点/扫描、2166有效点；深度约30.003Hz、640×480，末帧有效比例约0.466。第一次容器去掉全部capability后无法写jetson拥有的输出目录，未启动驱动；无硬件目录探针确认后只加DAC_OVERRIDE并将只读挂载缩至car子目录，复测通过。保留首次失败与复测日志。
- RGB+YOLOPv2电机禁用运行35秒，末控制样本692，模式dry-run，FP16；到时结束。该观察测试未作视频归档，保留status、配置、控制日志和程序输出，不作为正式实验。
- 部署后的10次联合样本显示7项传感器在线、YOLOPv2 torchscript FP16无模型错误、图像新鲜；当时雷达约10.01–10.04Hz，深度回调频率约8.81–28.08Hz，受联合负载影响，不能拿独立30Hz结果替代联合性能。最终健康检查all_sensors_online=true，模式dry-run，当前无归档实验run_id。
- 持续采集的原始状态文件在 `outputs/vehicle_dashboard/monitor/` 与 `outputs/vehicle_dashboard/sensors/`，会被新状态替换；验收样本已另存deployed-samples.json与final-state.json。雷达/深度用于面板观察，不代表进入控制避障链路。

### 验证与修复

- 新测试先复现缺模块和坏数值在线，再实现/修复。独立审查发现只读诊断的急停按钮、旧检测在线、退出未恢复监控、坏向量在线四项，均已修复并复核无剩余P1/P2；审查未接触硬件。
- 最终主工作区使用已部署venv在宿主运行200项相关测试全部通过：main-host-regression.log。初次系统Python缺torch、随后沙箱禁止监听socket的两次环境失败均保留，不能把这些失败写成通过。
- 最终命令覆盖lane_centering、camera_transform、camera_gimbal、hardware_mapping、lcc_web、hardware_safety、rosmaster_transport、yolopv2_primary/objects/torch_input、cloud_arbitration/onboard/worker、vehicle_dashboard、dashboard_sensors、dashboard_launcher共16个unittest模块。shell语法、git diff --check通过。
- JavaScript通过V8语法检查，模型新鲜度四用例online/stale/error/stale符合预期。真实Firefox Marionette检查7行传感器/JSON渲染、诊断页无可见控制按钮、HTTP断开后的连接中断提示；真实LAN桌面和窄屏（Firefox内宽450）无横向溢出。结果在browser-validation.json和deployed-browser-validation.json，截图dashboard-live.png/dashboard-mobile.png。截图在最后“感知监控”措辞及空run_id微调之前，仅标签随后修正。

### 下一负责人及产出

1. 用户可立即在同一局域网打开面板核对画面；本聊天维护状态/显示与后续接点，产出页面与实时接口验收记录。
2. 现场控制/标定仍由“小车代码”聊天的原任务推进；启动它的直接工具前先按使用文档释放相机/串口，或使用新实验入口。云端开启需要相应实验配置与本机密钥，面板本身不需要Key。
3. 空闲窗口由本车端聊天配合用户做真正的关机/重启恢复验收，产出冷启动记录；本次未擅自重启。
4. 全局目的地导航实现与验证仍是独立后续任务。传感器在线和GPU推理通过不等同于已实现目标到达、碰撞避免或实车停止距离。

使用说明：`docs/vehicle-dashboard.md`。本任务状态reported，等待用户实际浏览验收。
