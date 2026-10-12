# 外圈试运行与自动回放

2026-10-08（Asia/Shanghai）。当前 Jetson 配置，以车端实际记录为准。

## 已实现入口

在 `/home/jetson/VehicleCloudCollaboration` 执行：

```bash
# 只运行真实相机/GPU和虚拟控制，默认不连接电机
bash run_vehicle_experiment.sh \
  --vehicle-profile rosmaster_jetson_yolopv2_outer_trial \
  --max-runtime-seconds 30

# 会驱动车辆：限时试运行，停车后自动离线回放；Ctrl+C 停止
# 只用于当次已安排好的封闭场地、外圈方向、线缆和现场急停条件
bash run_outer_loop.sh 10
```

第二个入口默认30秒，可设1–120秒。四轮直行基准30 PWM，转向内侧减速，任何轮不超过30，停车0；低于30的内侧轮速仍不是已验证的负载转速。当前 `motion_calibrated=false`；该入口只允许已确认轮位与落地起步/停止的实验配置，不能当成完整运动标定。完整外圈、不压线和障碍绕行尚未验收。

入口临时暂停占用相机/GPU的只读感知监控，实跑时再暂停串口监控，退出恢复此前运行的服务。未修改服务配置。云端当前禁用，待用户选择接口。

## 每次保存什么

默认运行目录：`outputs/onboard_runtime/rosmaster_jetson_yolopv2/runs/<run-ID>/`。

- `raw.mp4`：控制循环读到的相机帧，MP4有损编码。
- `annotated.mp4`：模型实际消费的图像和对应掩码，异步模式下可能重复较早帧。**与 raw.mp4 是两个图像时间线**，按CSV中的 `semantic_sequence` 关联原始推理输入。
- `semantic_inputs/<sequence>.npz`：无损模型输入、原始道路/车道线掩码、目标检测、拍摄时间与序号。只存一次消费过的序号，当前采用无压缩NPZ降低写入耗时，约1.08 MB/份；本次30秒约348份。
- `onboard_log.csv`：与实际写入视频逐帧对应；包括序号、未舍入数据年龄、稳定滤波状态、行驶建议与虚拟/实发PWM。根输出目录的滚动CSV不可替代此逐运行侧文件。
- `status.json`、运行及车型YAML：终态、是否电机模式、丢帧、参数和配置。

试运行必须先写出首帧才允许发运动指令，检测到归档/日志写入错误或丢帧时停车。

## 回放

```bash
outputs/field_yolopv2/venv/bin/python -B \
  car/autodrive/tools/replay_outer_loop.py --run /绝对路径/到/运行目录
```

默认 `--mode auto`：存在NPZ时使用当时真实输出、消费序号和数据年龄，复核当前后处理与控制；旧记录没有NPZ时，使用压缩视频重新执行GPU模型。所有回放禁用电机、云台和云端。

`--mode recorded` 不重新评估网络权重，不重现CPU/执行器/归档耗时。它要求最终停车状态、完整连续日志、末样本和帧数一致，拒绝丢帧或截断记录。旧CSV的明确过期否决保留，新记录使用完整精度年龄。

`--mode video` 在压缩视频上同步推理，无法等价重现原异步输入选择及过期时序。必须按这种限制解读结果。

派生目录生成 `replay.mp4`、`commands.json`、`replay_config.yaml`、`summary.json`，不覆盖原始数据。没有提供逐段预期注释时，`expected_behavior_verified=null`；有注释时也只是开发核验，不是实车通过。回放结果异常先定位原始输入，再决定是否修改参数和再次实跑。

## 稳定处理参数

配置在两个 `rosmaster_jetson_yolopv2*.yaml` 的 `runtime_overrides.perception.semantic_stability`：时间常数0.12秒、道路支持阈值0.65、最大历史间隔0.25秒、相邻掩码IoU低于0.8时重置。按新模型序号更新，重复缓存不累计。

原始数据先经过新鲜度、近车道路、障碍和外圈边界检查，再对已选道路作短时滤波。输出始终是当前道路的子集；不补洞、不跨越当前车道线、不沿用旧道路绕过停车。它减少瞬态扩张与小幅边缘抖动，不能消除网络持续误判或证明完整外圈可用。

## 2026-10-09 实跑更新

外圈选择已修复“小封闭碎片被当成整行右边界”的问题。临时行区间计算可忽略很小的封闭洞，最终通道仍保留全部原始排除像素，拟合路径碰洞仍停车；大洞、长分隔线及外部连通缺口继续限制路线。

修复后5秒实跑预热后连续输出前进；随后7秒弯前实跑仍出现原始YOLO道路漏分和拟合失败。FP32、较大输入、16:9预缩放对照均未取得整段改善，因此实际配置保持352 FP16和4:3。当前不能直接按整圈完成理解此入口。

运行、回放、失败输入与验收结论见 `coordination/2026-10-09-outer-loop-followup-report.md`。后续先用保存的无损输入修复并回归，再进入下一段运动；原始数据保持不变。
