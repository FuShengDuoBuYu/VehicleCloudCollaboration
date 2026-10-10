# 2026-10-10 场地颜色规则与白线停车修复

用户本轮明确：白色标记都是斑马线，黄色为不可越过边界。对当前 visual_feedback 配置启用颜色辅助；YOLO road 仍是可行驶区域上限，颜色不生成道路，不指定入弯地点或预先执行转角。没有启动运动、重启服务、安装依赖或提交。

## 实现

- 新 `car/autodrive/perception/track_colors.py` 在被消费的 YOLO 同一原始相机帧上做 HSV 分类。lane 连通块具备白色支持、至少90% YOLO road 支持且不接触黄色时才释放；无法确认的 lane 保留。黄色在模型 lane 漏检时仍加入排除掩码，近车中间黄线超过既有视觉危险比例即取消 motion/pivot 安全许可。
- effective exclusion 同时传到 semantic route、路径估计器及显示，防止后续再次减去原始白线。时序滤波仍与当前掩码相交；不填补原始 road 孔洞。原始 road/lane/frame NPZ 不修改。新 CSV 记录启用标志、放行白线像素、排除黄线像素，已有近车黄线比例继续记录。
- 默认关闭，只有 `rosmaster_jetson_yolopv2_visual_feedback` 启用。透视校正配置被拒绝，避免拿原图颜色错配变换后的掩码。录制回放读取开关并输出颜色计数；新 run 额外保存 `track_color_parameters.yaml`。原始旧配置回放保留旧规则，候选规则需显式采用新 profile。
- 确认颜色只来自 `_consumer_result.source_frame`，不是更新的页面画面；缺帧、掩码不匹配、过期、错误、近处目标均保留停车。取消120秒总时限和独立运动截止不变。

## 实际录制输入对照

原始 session `0b2c18d8d3374d68ad2bad33a709357c`，run `20261010_043922_911806_hardware`，选90–109、265–275，共31行；原始 session `7ec6663e8a0544cfb0833a2e0558e145`，run `20261010_040429_978397_hardware`，选1154–1163、1432至归档结尾，共16行。分别用相同当前代码、颜色关闭/开启做配对，窗口控制器冷启动，消费原始帧/掩码/时钟/IMU/外部否决，并保留原始最终应用过期否决。

- 新近直道/斑马线窗口有效路径5/31→30/31；仅sample108仍因YOLO源过期无效。sample95由白线切断变为有效 forward 提案；sample275放行1537个白线lane像素，排除5962个黄色像素，但仍因历史最终应用年龄0.31844s停车。
- 旧弯道窗口有效路径16/16→16/16；增加黄色后部分右出口判据变为未确认，仍可给出短步前拱提案。这是当前冻结输入的结果，不是转向后新画面的预测。
- 所有47行检查 candidate corridor 为原始 road 子集且不包含 effective exclusion 像素。颜色处理单独CPU中位7.106ms、最大15.576ms；不是端到端100ms保证，也未测GPU/真实链路。
- 新近窗口最终指令仍28stop/3forward；旧弯窗口11stop/5forward。不可将路径恢复等同于实车持续行驶，历史时效和启动恢复门禁仍会阻止动作。

方法/精确逐行结果/图像/input SHA-256：`outputs/visual_feedback/20261010/track-colors-fix/replay-colors.py`、`paired-records.json`、`replay-summary.json`。原始归档保持原样；候选图像仅写派生目录。

## 验证与加载状态

`outputs/field_yolopv2/venv/bin/python -m unittest car.test.test_track_colors car.test.test_yolopv2_primary car.test.test_control_render_timing car.test.test_outer_loop_replay car.test.test_stationary_corner_onboard.StationaryCalibrationTests.test_main_logs_native_encoder_snapshot_once_per_control_frame car.test.test_stationary_corner_onboard.StationaryCalibrationTests.test_final_application_veto_stops_mask_that_expires_after_initial_gate -q`：33项通过，11.252s。只有CPU合成帧、临时录像、虚拟驱动和原始数据对照；未打开相机/底盘。先前系统python执行旧回放测试时缺少torch，改用项目现有venv通过，不安装依赖。`git diff --check`通过。独立只读审查未发现重要回归。

通过 `SessionClient().get_state()` 只读查询本机socket：available=true、state=finished、running=false、mode=hardware、runtime_limit_seconds=0、stopped_verified=true，仍为0b2c18d8会话。准确状态副本 `loaded-panel-state.json`。下一次用户点击开始时由新 runtime 进程读取本版代码及profile，不需要重启启停服务；当前后台观察画面可能仍是旧monitor进程，不当成新规则验收。

HEAD `fce828b89b084eda293e83b99c8861f522858d8e`，大量已有dirty保留，无commit。本轮before/after、增量patch、source/hash及验证记录在上述派生目录。下一负责人仍为本车端聊天读取用户下一run，核对实际跨白线、黄线位置、直道执行时效和弯道逐帧控制。场地光照/色偏、真实车身避线、实际通过斑马线和过弯尚未实车验证。
