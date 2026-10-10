# 实时视觉小步控制与离线回放

2026-10-10：用户明确要求主要通过运行数据反复离线排错，减少重复开车。“回放”首先是控制器调试入口，网页视频查看只是辅助。

## 控制行为

新 profile：`rosmaster_jetson_yolopv2_visual_feedback`。当前 YOLOPv2 可行驶区、车道线和目标检测经过车端门禁，形成局部方向；右出口目标从当前道路像素取值，不用历史道路宽度裁剪。近场方向仍平缓、前边界未接近且只需轻微差速时连续巡航，每个新的有效模型序号更新方向及许可；当前右向误差较大时小步原地右转，弯前也可小步前拱。每周期可撤销动作，取消固定90°目标和固定入弯距离。保留旧控制器以对照历史记录。

转弯/前拱的单个短步最多0.15秒，原地旋转还有5° IMU上限，现profile零输出等待0.25秒后必须有等待期结束后采集的新 YOLO 帧；原归档默认仍为0.7秒。巡航只由新序号续期，运动截止不超过当前消费帧采集时间+0.3秒及本次更新+0.25秒；缓存不能延长许可。巡航转短步须先停顿再观察，即使巡航到期与转弯请求同时出现也不能跳过。独立线程检查截止时间。PWM保留此前用户指定的最高30。以上是软件幅度/时效限制，不是预编程路线，也不是实测最低速度、刹停或角度保证。

visual模式在执行前完成道路和门禁计算，执行后再画同一消费帧；CSV另记 `control_render_ms`、`stationary_continuous` 和 `stationary_motion_deadline`，避免把绘图时间隐藏在控制处理里。旧配置默认不开连续巡航，用候选profile回放才比较新模式。

门禁缺少当前道路、相机/模型/IMU过期、归档错误/丢失时不继续输出。新的实车 profile 还要求当前运行的完整雷达/深度原始数据和UART采集已经开始。雷达/深度目前仅采集及记录健康检查，不宣称已用于避障。

## 页面启停的部署方式

已按用户本次明确授权重载现有网页和传感器服务，并通过临时用户服务启动实车模式supervisor，当前待机，用户点击才启动。部署需要在允许操作服务的时间窗口重新加载 `vehicle-dashboard.service` 和 `vehicle-sensors.service`。后者运行在现有容器中，加载完整雷达/深度记录代码。无需安装依赖、修改服务unit或启动电机。

终端中启动独立启停程序：

```bash
bash run_autodrive_panel.sh
```

默认是“开始预演（车轮不动）”。页面通过本机Unix socket提交启停意图，不提交命令、PWM或运行参数。原HTTP服务继续保持 `NoNewPrivileges=true`。程序本身启动时不打开设备；以后点击开始才调用实验入口协调相机/串口所有权。

用户安排实车实验后，可以在车端终端显式启用实车按钮：

```bash
bash run_autodrive_panel.sh --authorize-motion I_UNDERSTAND_MOTORS_WILL_MOVE
```

该命令只使后续“开始自动驾驶”点击可启动实车。开始/结束按会话记录；结束写停止请求并通知运行进程。启动页面发送心跳，离开页面、刷新或断线后不再续租，约3秒租约过期停止本次会话。按用户2026-10-10后续要求，面板会话传入总时限0，代表无总时长上限，由用户手动结束；局部指令截止、数据过期、记录错误及取消门禁仍保留。软件停止确认不替代车身实际停止观察。

## 每次归档

面板会话根目录：`outputs/autodrive_sessions/<session_id>/`。

| 路径 | 内容 |
|---|---|
| `session.json`、`command.json`、`console.log` | 会话、固定启动命令、终端输出、结束状态 |
| `runtime/runs/<run_id>/onboard_log.csv` | 每个控制周期、动作、虚拟/实车PWM、IMU/编码器及其时效、执行门禁 |
| 同目录 `raw.mp4`、`annotated.mp4` | 当前相机显示帧、YOLO/控制叠加；MP4为有损视频 |
| 同目录 `semantic_inputs/*.npz` | 实际消费的原始模型输入图像、road/lane掩码、目标、序号、采集时间；无损 |
| 同目录 `resolved_runtime_config.yaml` | 当次完整配置快照 |
| 同目录 `sensors/events.jsonl` | 全部收到的UART原始字节、发出的UART帧，以及完整遥测快照 |
| `outputs/vehicle_dashboard/sensors/recordings/<run_id>/` | 每次收到的全量LaserScan ranges/intensities、原始深度字节 `.npy`、消息时间/接收时间/编码与尺寸、记录状态 |

IMU、姿态、编码器、电压包含在UART和遥测中；记录收到的样本，不补未收到的固件样本。深度保持原始字节和步长，雷达保留NaN/Inf。各路接收时间使用同主机单调时钟；ROS源时间另存，未经时钟/外参标定不宣称硬同步。查看用的稀疏点云和深度JPEG不能替代原始数据。ROS目录与运行目录通过run ID绑定，搬数据时两者都要保留。

## 离线复现和比较

结束正式采集后执行；不要与实车正式运行同时做后台分析。

```bash
outputs/field_yolopv2/venv/bin/python -B car/autodrive/tools/replay_session.py \
  --session outputs/autodrive_sessions/<session_id> \
  --output outputs/visual_feedback/<新的回放目录>
```

默认读取当次 `resolved_runtime_config.yaml`，使用当前控制代码。它不打开相机、串口、ROS订阅、GPU模型或云接口，不创建实体底盘。要比较候选配置，在命令后添加 `--profile <候选profile ID或YAML路径>`；原配置保留，候选的完整解析配置与hash也保存。

输出：`controller/commands.json` 每行决策/PWM/理由/状态；`controller/summary.json` 动作和停车原因统计；`decision_changes.json` 与历史动作不同的行；`sensor_timeline.jsonl` 每个控制时刻之前最近的各路原始样本引用、年龄与缺失；`manifest.json` 原始文件hash、数据完整性与回放代码hash。不会拿未来样本填入过去。`.5`秒新鲜标识只用于索引，实际控制时效门禁使用其独立配置。

完整性失败默认拒绝。旧运行缺少原始雷达/深度时可明确添加 `--allow-incomplete-sensors`，结果标记为历史部分回放。日志序号缺口、未终止运行、缺失模型输入和路径越界仍拒绝。输入数据保持原样，输出目录不能在原始run内部、不能覆盖已有目录。

给已人工标注的片段添加 `--expectations <JSON路径>`，例如：

```json
[{"start_frame": 10, "end_frame": 15, "actions": ["stop"], "source": "人工复核：道路缺失"}]
```

这样才能判定特定bug是否通过回归；单纯动作数量变化不是成功率。旧 `live04` 的新控制回放见 `outputs/visual_feedback/20261010/live04-verified-replay/`。旧轨迹不变，修改控制后会出现的新视角无法从旧录像推出；真实运动仿真还需要轮速/惯性/相机外参与尺度依据。离线回放能减少重跑，不能独立证明首弯或整圈成功。

## 正确行驶参考：b6c人工辅助两弯

2026-10-10用户进一步明确：人工推动车身/摆正后正确完成两个弯道、未压线，这次摄像头视角序列可作为期望自主行驶的开发参考。后续需要比较算法转向时机、阶段及运动方向与该参考，而不把包含干预的自动控制日志当作期望动作、不按录像秒数写死路线。

本次完整无损回放1183周期动作/PWM/阶段均与原日志一致，但两个近似右转窗口分别81/92与75/82输出stop，尚不匹配期望运动。报告 `coordination/2026-10-10-b6c-reference-replay.md`；原数据/配置/hash/传感器对齐 `outputs/visual_feedback/20261010/b6c-reference-replay/`；期望运动时间线/人工说明/对照图/约100.8s一倍速视频 `outputs/visual_feedback/20261010/b6c-reference-analysis/`。原MP4固定20fps在实际控制不足20Hz时会加速，分析应按CSV时间对齐。

当前visual profile时效已为.6s、恢复2新结果；上文.3s是初版归档参数。回放默认冻结配置，比较候选时以各次完整配置快照为准。

2026-10-10后续已依据此参考修改并启用实际visual profile的 `perception.track_colors.surface_assist: true`。控制仍读取当前新鲜YOLO/目标头；该场地可用模型消费的同一图像中、当前黄色边界限定的灰/白地面补漏检。新增地面不来自历史帧，缺失横界列不补、侧界不外推；绿色/黄色排除。入弯依据当前边界/出口，原地转可在向前路径拟合失败但当前近车道路/方向仍有效时进行。此模式有意不再要求每个可行驶像素都来自YOLO road头，必须配套启用视觉转弯守卫。

最终同输入改版回放1183周期：995停/131前进/0pivot→697停/395前进/48短pivot，两个弯均产生右转输出，但第二弯仍有道路余量拒绝，不能算新实车正确过弯。报告 `coordination/2026-10-10-b6c-policy-fix.md`；一倍对照视频 `outputs/visual_feedback/20261010/b6c-policy-fix/candidate-reference-1x.mp4`；最终逐行输出在 `final-verified-replay/`，旧卡弯完整对照在 `final-c92-replay/`。下一用户Start加载已修改的profile，本轮没有新实车运动。

2026-10-10再修正第二弯：原地小转使用最近可见地面的当前支持/连通方向，与前进近路径检查分开。补充的浅黄只能与当前源图强黄块连通，不连接缺失像素；模型已有灰白路面可去除lane纹理，无当前黄边界时不增加模型缺口。最终目录为`outputs/visual_feedback/20261010/b6c-second-corner-fix/final-source-replay/`与`final-source-c92-replay/`，其他目录是保留的迭代结果。第二弯940–1021的道路拒绝37→0，前拱与新帧出弯确认后恢复连续行驶；接近段仍15条拒绝/恢复等待，不能把整条参考轨迹称为已完全匹配。报告`coordination/2026-10-10-b6c-second-corner-fix.md`，对应100.8s一倍速视频在同派生目录。67项针对CPU检查与无设备启动校验通过，本轮没有重启程序或新实车验证。
