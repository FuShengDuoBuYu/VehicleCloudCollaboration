# b6c：两次人工辅助正确过弯的参考回放

2026-10-10，车端。用户报告本次两个弯道正确且没有压线，明确干预方式为推动车身/手动摆正方向；精确干预时刻未提供。此报告把成功路线作为开发参考，不把人为运动算成自主完成。未修改驾驶算法或profile，未开设备/GPU、启动运动、重启服务、安装或提交。

## 对应记录与可重复输出

- session：`b6c3246e667542939e40199e5cce8901`；run：`outputs/autodrive_sessions/b6c3246e667542939e40199e5cce8901/runtime/runs/20261010_061654_638526_hardware`。用户End，exit0、停止确认、cleanup_errors空，归档1183/1183、dropped0。
- lossless YOLO输入840个；每个控制时刻保留消费序号/捕获年龄、当前IMU、相机年龄、执行前许可。五路原始记录均closed、dropped0、error空：uart_rx4763、telemetry1183、uart_tx169、depth1561、lidar1007。run最后状态的ROS1006/closedfalse是退出前快照，独立最终journal为1007/closedtrue，本次核验使用后者。
- 用同一冻结配置、原始图像/掩膜/目标头及记录时钟，调用当前相同感知后处理与控制器；默认不重推模型、不连接相机/串口/ROS/云/底盘。完整回放1183周期，动作差异0、四轮PWM差异0、控制阶段差异0；steering与原CSV最大差约4.99e-6，符合原日志5位小数舍入。动作995stop/131forward/57turn-right，pivot0。

实际命令：

```text
outputs/field_yolopv2/venv/bin/python -B car/autodrive/tools/replay_session.py --session outputs/autodrive_sessions/b6c3246e667542939e40199e5cce8901 --output outputs/visual_feedback/20261010/b6c-reference-replay
outputs/field_yolopv2/venv/bin/python outputs/visual_feedback/20261010/b6c-reference-analysis/build_reference.py
```

首条命令输出目录必须新建，不覆盖已有回放。第二条为本次派生分析脚本，没有硬件接口。未运行全仓测试；实际全段回放、输出逐行比较、图像/图表检查和视频抽帧解码作为本轮验证。

## 正确视角与当前控制的差异

视频联系图与解卷绕IMU曲线支持两段右转，以下为离线推断的近似窗口，不是精确人为干预起止标注：

| 窗口 | 原run时间 | 观察航向变化 | 当前回放输出 |
| --- | --- | --- | --- |
| sample730–821 | 83.1569–90.088s | -83.096° | 81stop、3forward、8turn-right |
| sample940–1021 | 101.5035–106.257s | -83.193° | 75stop、7turn-right |

第一段92行只有44有效路径，35行近车road不足、9行预览左边界缺失、4行拟合离开corridor；有效路径中也有18行被front/exit转向许可拒绝。第二段82行只有14有效路径：40行视觉yellow安全区拒绝、23行近车road不足、5行road空/不合理。用户称实车未压线，故这40行是需要复核的相机ROI与真实轮迹差异，不能直接证明车轮压线，也不能直接删除边界门禁。

因此，“回放复现原控制输出”已经成立；“算法自行产生与正确过弯相近的运动”尚未成立。把原995条stop复制成目标不能解决驾驶；按录制时刻写死转向也不是视觉反馈算法。

## 1倍时间线与产物

派生目录：`outputs/visual_feedback/20261010/b6c-reference-analysis/`。

- `reference-replay-1x.mp4`：左原相机、右无损消费输入的算法叠加，显示实际run时间、sample、YOLO序号、观察航向、虚拟PWM、各原始传感器接收年龄和最终拒绝理由。视频1008帧/10fps，约100.8s；播放重复保持最近已有画面，不插造输入或决策。
- 原始MP4固定20fps、1183帧约59.15s，但CSV首末时间16.0626–116.6942s，经历100.6316s；因此原视频直接播放会加速。1倍视频按原控制时间戳保持各sample，10fps只是显示采样频率，不代表YOLO恒定10Hz或本轮实时计算性能。
- `reference-timeline.jsonl`：1183行，记录观察航向/原生编码器、因果向后0.2s变化率、当前回放动作/PWM、最近过去各路传感器引用和年龄。yaw-rate类别仅为分析代理，不用于实时控制，也不把短步等待自动判成错误。无距离标定，不将ticks转为米。
- `reference-annotations.json`保存用户成功/无压线/手动干预说明、近似窗口及来源；`comparison.json`、`window-diagnosis.json`保存量化对照；`reference-vs-output.png`展示观察航向、输出PWM与编码器变化；`raw-reference-contact.jpg`为原视频选帧。
- `source-snapshot/`、`source-sha256.json`、`verification.json`、`worktree-status.txt`保存当前控制/回放代码、方法和状态。完整原始各文件hash、配置hash及传感器对齐索引位于 `../b6c-reference-replay/manifest.json`及`sensor_timeline.jsonl`。复核选取raw.mp4、CSV、配置、local journal原hash未变；原始数据不写入Git。

HEAD仍 `fce828b89b084eda293e83b99c8861f522858d8e`，114项dirty/未跟踪入口保留，无提交；本轮业务代码未改变。后续负责人仍本车端聊天：以这次正确视觉变化为开发参考，复核近路/出口方向、可见黄线与轮下区域、road漏检，修正感知或决策后重放同一输入并比较转向阶段；用户提供精确干预时刻可提高参考标注。实际新轨迹仍需后续自主实车采集。

## 限制

这次不是动力学闭环仿真。人手产生的姿态和编码器变化无法用当前PWM解释，故不能用本run直接拟合可靠电机响应；算法改变动作后的新视角未被预测。相机/IMU/雷达/深度按同主机接收时间因果对齐，源时钟/外参未完整标定。雷达/深度目前被保存和显示，未接入驾驶决策。用户提供的“没压线”是参考标注，单凭车载画面不独立证明轮下边界。
