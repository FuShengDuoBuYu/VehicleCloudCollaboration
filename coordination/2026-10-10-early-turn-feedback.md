# 2026-10-10 实车仍卡弯：近场方向与连续曲线路径修复

用户结束实跑，并提供现场停车照片，反馈“弯道还是卡住了，好像转弯太早”。本轮依据真实相机帧、YOLO掩膜及IMU/执行日志修改；未新增实车运动，未重启服务或安装依赖。

## 原始证据

- Run ID `20261010_040429_978397_hardware`，session `7ec6663e8a0544cfb0833a2e0558e145`。原始目录 `outputs/autodrive_sessions/7ec6663e8a0544cfb0833a2e0558e145/runtime/runs/20261010_040429_978397_hardware/`；1438行，动作1255stop、164forward、12turn-left、7pivot-right。最后运动sample1144，用户结束软件确认stopped=true、cleanup_errors=[]。
- 首次原地右转sample153（24.7941s）：远heading=.21531，近near_heading=.04774，front未观察到。原控制使用远heading>.18立即pivot，照片/视频输入显示当时近处仍是直道。转弯偏早不是仅凭现场照片作出的判断；照片未作距离或车身坐标标定。
- 真正front/right-exit出现后sample1103/1127继续pivot，随后sample1159及末sample1437反复停。原原因：`right-exit-path-crosses-current-exclusion`，继而 `YOLO outer left edge missing at preview`。内圈曲线进入右下画面，直线连远出口穿黄线，但当前可行道路是弧线。这是上一轮射线检查的几何局限，不是简单将门槛降低。
- 原始UART/telemetry封口：4841uart_rx、1438telemetry、280uart_tx，closed=true、dropped=0/error=null。ROS封口：1006lidar、2305depth，closed=true、dropped=0/error=null。保留原始视频、CSV、NPZ和传感器文件。
- 现场照片 `56da95caf51a2c837c02f745c724a478.jpg` 保持原样，路径/sha256记录在派生目录。

## 本轮增量

- `semantic_outer_route.py` 仅visual_feedback模式先逐行选择当前ego可达分支，再寻找该分支的front/right-exit目标，保留全部当前road/lane排除像素。避免跨入同一连通大区域中的另一条分支。
- 去除“直线射线必须畅通”的曲线路径否决；由现有RoadCenterlineEstimator验证侵蚀后的连续路径和原始掩膜中的每一步。闭合排除孔的直接目标线否决仍保留，洞/缺失像素不会由时序或路径规划填充。
- 视觉备用路允许当前右边界或当前真实front支持，避免左边界在图像外就永久失效。没有观察到边界的全路掩膜仍不能由此放行；YOLO硬门禁、近车支持、时效、IMU及记录门禁不改。
- 视觉控制和原地右转选择使用当前near_heading_error；远heading保留用于记录/显示，不直接触发立即pivot。analyze只在visual selector模式拷贝estimate的近方向交原LCCController，不改原参数/其他模式。没有固定入弯距离、完成角度、计时路线；短步时长、角度上限与停顿不改。

## 验证和限制

- 修改前两项针对性复现均失败：近路径直但远目标偏右仍触发pivot；有真实连续弯路却被直线检查否决。修改后47项纯CPU感知/视觉控制检查通过，另保留录制回放相关检查结果；Python编译和本轮差异检查通过。只读复核未发现重要或阻断缺陷。
- `replay-windows.py` 使用同一run原始lossless NPZ、真实序号/年龄/IMU/最终执行否决，同时加载本轮before的selector、analyze和Runtime实现做before/after对照，无模型GPU、相机、底盘或云端。
- 每个窗口从新controller开始，因此不能称整轮动作原样复现：straight400:440有效40/40且动作完全一致；early-pivot140:190均有效，候选保持forward/stop（该冷启动窗口的baseline没有复现真实sample153的pivot，转早证据以原始CSV和两个针对性检查为准）；corner1080:1165有效70/85→85/85，baseline2pivot-right、候选0pivot-right及15forward；stopped1170:1240有效1/70→70/70，候选11forward而baseline全停。
- 当前记录末帧重算near方向约.006–.013，远方向约.669，新控制选择前进以接近弯道，而不是在旧停车点继续完成预定角度。由这个静止场景不能预测前进后的相机画面，不能宣称新轨迹已过弯，也不能证明不会转得过晚。
- HTTP只读确认当前finished、running=false、stopped_verified=true、7项传感器在线。已有supervisor每次点击Start都会从仓库启动新runtime进程，故这次不用重启服务，新代码在下一次用户点击Start时加载。助手未触发Start或电机命令。

## 版本和产物

HEAD仍 `fce828b89b084eda293e83b99c8861f522858d8e`，未提交/推送，已有dirty保留。产物 `outputs/visual_feedback/20261010/early-turn-fix/`：before/after源码、hash、只含本轮修改的patch、原始输入选段hash、窗口命令/summary、源相机道路派生图、原始照片provenance、验证与ready-check。下一步用户实车点击开始/结束，本聊天读取新run继续修正；本轮不把有限回放提升成整圈自动驾驶结论。
