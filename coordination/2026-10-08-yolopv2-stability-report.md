# YOLOPv2 输出稳定性修复与实测记录

日期：2026-10-08（Asia/Shanghai）。执行环境：当前 Jetson。仓库 `/home/jetson/VehicleCloudCollaboration`，HEAD `fce828b89b084eda293e83b99c8861f522858d8e`，main 存在多项既有未提交及未跟踪文件，本次全部保留，未提交/推送。后续代码与证据哈希见同目录 `2026-10-08-yolopv2-stability-manifest.json`。

## 结论与边界

已修复异步掩码与显示图像错位，并在YOLO输出的已选道路上增加保守时序稳定处理。最终真实相机/GPU、**电机禁用**的30秒验证：569帧视频、569行日志、348份无损输入，归档和日志零丢帧、无写入错误；预热及恢复检查之后512个连续直行建议、没有再次停车。

同一批348份原始掩码比较关闭/开启滤波：已选道路逐帧平均变化减少35.54%，航向估计逐帧平均变化减少42.20%，横向误差逐帧平均变化减少50.63%；两组均无无效几何估计。这是静止直道的开发诊断，**不是模型精度、行驶安全率、完整外圈或不压线验收**。网络权重未重新训练。移动、弯道、交叉口的误判仍需验证。

用户已说明此前内岛画面来自手动搬车充电，随后移到另一条直道；不能把这些不同场景的差异都归因于模型跳变。本轮修复阶段没有新增电机动作。云端仍按用户要求暂缓选型，未调用真实云端。

## 已核验原因

1. **代码确认：异步显示错位。** 原运行循环把较早YOLO结果叠在新相机帧上。现按消费序号取该次推理的原始图像显示，并标注数据年龄。原始相机视频仍保留，两个图像时间线明确区分。
2. **实际GPU诊断：未发现相同输入随机变化。** 同一张无损图片在主线程/工作线程各重复推理，FP16/352、FP32/352、FP16/480每组均只产生一个道路掩码哈希。352/FP16对保存的无损输入重算，所测样本与原道路掩码IoU为1。FP32与FP16所测中位IoU约0.99989，暂无证据表明FP16是大幅跳变主因。测试位于 `outputs/outer_loop_trial/20261008/stability-probe/`，输入是搬车后的内岛画面，不能当成新直道准确率评估。
3. **实际输入变化：** 新直道静止146份模型输入的相邻原始道路掩码IoU中位约0.99547、最小约0.98451；需要处理小幅边缘变化。新滤波只作用于控制使用的道路，原始网络输出保持原样供诊断。
4. **已复现归档瓶颈：** 压缩NPZ在Jetson单份写入中位71.03 ms，无压缩NPZ约6.53 ms。中间30秒采集发生141帧归档丢失，现保留该失败记录；最终无压缩采集零丢帧。没有删除失败证据或通过增加队列掩盖问题。
5. **尚不能确定的原因：** 先前运动片段中的局部道路缺失没有保存无损推理输入，压缩视频重推又未复现；不能断言已查清该次大幅误分割的根因。现在补齐原始输入/掩码/消费时间留档，后续可准确定位。

## 修改内容

- 新增 `car/autodrive/perception/semantic_stability.py`。参数：EMA时间常数0.12秒、阈值0.65、最大历史间隔0.25秒、IoU重置阈值0.8。只在新消费序号到来时更新；缓存重复不累计，场景突变和时间中断清除历史。
- 原始道路/车道线、过期、近车支持、障碍和外圈原始边界检查优先；之后再平滑已选通道。滤波结果与当前通道取交集，当前排除的像素立即排除。保留所有停车门禁。
- 原始外缘先验顺序已加组合回归：旧道路从x=10起，新道路触及x=0，必须按当前外缘不可见停车，不能因历史滤波形成虚假边缘。
- `onboard.py` 保存无损 `semantic_inputs/*.npz`、原始序号、未舍入年龄、滤波状态；叠加图使用模型消费帧。原始道路掩码仍供云端观察/审查，滤波不改写它。
- `replay_outer_loop.py` 自动优先回放无损记录的输出、序号、年龄和目标检测，不开启设备或云端。明确拒绝缺最终状态、缺帧、截断、末样本/帧数不符的运行；旧CSV保留明确过期否决。
- `field_trial_recording_ready` 对视频/日志错误及丢帧停车。正常标定门禁仍存在，实验profile不把 `motion_calibrated` 改为true。

## 证据与命令

以下路径均相对 `/home/jetson/VehicleCloudCollaboration/`，原始文件保持原样。

| 类型 | 路径/结果 |
|---|---|
| 新直道修复前采集 | `outputs/outer_loop_trial/20261008/new-straight-stability/runs/20261008_100152_204986_dryrun/`；146份留存无损输入。归档有缺口，只用于留存帧成对后处理比较 |
| 中间压缩归档失败 | `outputs/outer_loop_trial/20261008/stability-live-after/runs/20261008_101348_433396_dryrun/`；577循环、436归档、141丢帧，不能完整时序验收 |
| 最终真实相机/GPU静止验证 | `outputs/outer_loop_trial/20261008/stability-final-live/runs/20261008_101844_471341_dryrun/`；569帧、348份输入、零丢失，电机禁用 |
| 最终采集统计 | `outputs/outer_loop_trial/20261008/stability-final-live/verification.json` |
| 完整无损消费序列回放 | `outputs/outer_loop_trial/20261008/stability-final-recorded-replay-verified/`；569帧，57启动停车+512直行，与记录一致；两段开发诊断预期通过，非独立实车地面真值 |
| 成对比较原始结果 | `outputs/outer_loop_trial/20261008/paired-final-original-straight.json`、`paired-final-live.json` |
| 可复现统计脚本 | `outputs/outer_loop_trial/20261008/compare-stability.py`；逐个无损掩码，先选路再比较滤波，不是重新推理/完整运动回放 |
| 写入开销测试 | `outputs/outer_loop_trial/20261008/archive-write-benchmark.json` |
| 测试记录 | `outputs/outer_loop_trial/20261008/stability-final-tests-with-replay-completeness.log`；213项通过，均不驱动硬件 |
| 使用方法 | `docs/outer-loop-record-replay.md` |

实际最终采集命令：

```bash
bash run_vehicle_experiment.sh --vehicle-profile rosmaster_jetson_yolopv2_outer_trial \
  --max-runtime-seconds 30 --output-dir outputs/outer_loop_trial/20261008/stability-final-live
```

相关测试覆盖当前障碍/缺路/过期立即否决、原始外缘、缓存序号去重、历史超时、掩码不扩张、无损归档、回放终态/缺帧检查、底盘参数/停止、云端本地仲裁、仪表盘与启动脚本。`git diff --check`、Bash语法及安全help入口通过。结束后四个原有监控服务均实查active，未安装依赖或改写服务配置。首次最终离线回放进程收到终止（退出143），其不完整派生目录保留，不能作为通过证据。

最终回放命令（重复执行时使用新的输出目录）：

```bash
outputs/field_yolopv2/venv/bin/python -B car/autodrive/tools/replay_outer_loop.py \
  --run outputs/outer_loop_trial/20261008/stability-final-live/runs/20261008_101844_471341_dryrun \
  --output outputs/outer_loop_trial/20261008/stability-final-recorded-replay-verified \
  --expectations outputs/outer_loop_trial/20261008/stability-final-live/replay-expectations.json
```

此前 `new-straight-recorded-replay/` 使用尚未检查归档缺口的版本产生，现已明确撤销其完整时序验证意义，目录追加 `INVALIDATED.md`，保留原始输出。最终验证以 `stability-final-recorded-replay-verified/summary.json` 为准。

## 先前已授权的两次短时实跑，补充持久记录

这些是本轮稳定修复之前的实际设备运行，不能与本轮静止验证混写。

- `outputs/onboard_runtime/rosmaster_jetson_yolopv2/runs/20261008_094824_769810_hardware/`：10秒、143归档帧、8个forward/135个stop，最终PWM0。启动斜坡一次约273 ms，下一异步掩码过期，导致反复停走。实验profile的阻塞斜坡现由0.25改为0；用户已实际确认30 PWM落地起步与停车。普通profile斜坡不变。
- `outputs/onboard_runtime/rosmaster_jetson_yolopv2/runs/20261008_095119_960468_hardware/`：10秒、191归档帧、30个forward/161个stop，最终PWM0。非零指令写入约0.3–0.4 ms，仍出现横向几何过大、拟合离开掩码及近车道路缺失；故继续诊断而非宣称整圈成功。
- 两次压缩视频同步重推分别为137/143、185/191个forward，无法重现原时序与局部失败，证明它们不能替代真实消费掩码回放。相关日志 `outputs/outer_loop_trial/20261008/first-cycle-console.log`、`second-cycle-console.log`，派生回放在 `outputs/outer_loop_replays/`。
- 已实现 `run_outer_loop.sh`：限时运行、停车、自动离线回放。YOLO负责道路/车道线/目标感知，控制器负责把几何转换为差速；没有启用原颜色边界导航或盲目弯道续行。车端本地仲裁保留，真实云端尚未验收。

## 下一负责人、任务、预计产出

1. **本远程小车聊天**：以30 PWM继续有时限的直道—首弯闭环；每次录制并自动回放，重点检查运动中掩码跳变、转弯内侧轮速和停车原因。预计产出每次run的视频、无损输入、控制/仲裁日志、对应回放及逐问题结论。现场用户负责看护和急停，不要求手动反复调参数。
2. **论文统筹聊天/用户**：收到上述实跑结果后核对证据，判断能否进入完整外圈验收；预计产出通过/失败/未验证清单。不能据本报告认定不压线或完整绕圈已通过。
3. **本地实现与评测聊天**：实跑稳定后按统筹分配做逐事件标注、独立运行对比和论文图表。用户确定云接口后，车端接入真实云端仲裁；密钥仅保存在本机秘密配置，不发到聊天。
