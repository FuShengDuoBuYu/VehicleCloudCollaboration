# 2026-10-10 直道连续反馈与无总时限面板

用户确认要“直道持续有效时连续低速修正；弯道才短步转动或前拱”，随后明确要求移除120s总时限。沿用此前加载授权，仅待机时重载启停程序，不由助手启动运动。实现计划：`coordination/2026-10-10-continuous-cruise-plan.md`。

## 停车及卡顿依据

本次照片 `b3297eb9969848aec065ac45e58672c0.jpg` 保持原样；哈希见产物 `source-inputs.json`。不能由顶视照片断定前进后的相机边界可见性或车身安全余量。

源session `a3dde8b5ae9a44cea9681aca2d0f3f5a`、run `20261010_042127_172648_hardware`：结束为operator interrupted，stopped=true。CSV共407行，末150行74次visual-current-evidence-veto、34次执行前语义/相机过期，18次运动建议。最后401仍有当前有效道路建议；403语义执行年龄0.3052s触发最终否决，后续需再攒5个不同有效观察。此前每步0.15s、强制零输出0.7s后再等新采集帧，导致起步间隔中位1.794s、最长6.652s；不能等同YOLO推理频率。末260行推理中位95.51ms、语义执行年龄中位0.2800s。

四个真实存储输入的纯CPU profile：debug render平均约43ms、曲线路径约38ms。是测量本机处理开销，非在线GPU周期。原始文件未修改。

## 修改

- visual profile启用 `continuous_cruise=true`：当前近方向和横向偏差平缓、前边界尚未接近且控制建议仅需轻微差速时连续更新；保留原建议左右轮修正。近方向决定原地右转，弯前接近边界或修正较大时仍短步，不按固定地点或固定90°转弯。
- 新模型序号才能续巡航许可，独立deadline不超过消费帧captured_at+0.3s及更新now+0.25s。缓存不得续期；过期、道路/记录失效、取消仍撤销。巡航→短步先停车并等待停顿结束后的新采集帧；短步0.15s/5°上限不变，本profile停顿由0.7改0.25s，旧归档配置默认保持原值。
- analyze默认仍同步渲染，visual实车入口通过deferred renderer先算道路/门禁并执行命令，再画同一消费帧。曲线规划预先建索引矩阵，替代逐行逐offset临时数组；原掩码排除和逐像素检查不变。
- CSV新增 `control_render_ms`、`stationary_continuous`、`stationary_motion_deadline`，状态继续保留当前消费序号与时效。
- 面板固定启动参数 `--max-runtime-seconds 0`，页面提示“无总时限，手动结束”。受页面会话控制的实车入口允许0，独立CLI实验原默认边界保持；3s页面租约、取消锁存、局部deadline、watchdog、记录错误和时效门禁均保留。总时限0不等于持续保证非零PWM。

## 验证和限制

先复现缺失巡航/延迟绘图接口以及120参数断言失败；新增巡航续期、缓存不续期、源到期、微调保留、近前边界切短步和延迟绘图用例。审查发现巡航到期与弯道请求同周期可能跳过停顿，先复现1.251s停车→1.27s pivot的断言失败，再修正为所有非巡航恢复都需停顿及停顿后采集；修后用例通过，独立只读复核关闭Important，无其它Critical/Important。

最后命令：`python3 -m unittest car.test.test_visual_feedback car.test.test_control_render_timing car.test.test_semantic_bend_path car.test.test_drive_sessions car.test.test_stationary_corner_onboard.StationaryCalibrationTests.test_main_logs_native_encoder_snapshot_once_per_control_frame car.test.test_stationary_corner_onboard.StationaryCalibrationTests.test_final_application_veto_stops_mask_that_expires_after_initial_gate -q`，51项通过，纯CPU/模拟底盘/合成子进程。compile和git diff --check通过。不运行全项目设备测试。

短CPU配对回放107行（直道60–99、接近160–189、末尾370–406），每窗口controller冷启动；前后107行几何有效性、远近方向、横向误差及路径像素完全相同。控制前处理耗时中位100.27→45.32ms，候选绘图延后另算，不包含GPU推理。保留原捕获/控制/IMU/外部门禁及最终否决时间；旧时间表仍产生过期停车，仅7行候选巡航，不把CPU节时替换成虚构的新历史时间戳。详见 `replay-summary.json`、各窗口replay.json与脚本；不能证明新物理轨迹、停稳或首弯通过。

检查待机准备加载时发现用户又自行启动了session `3344a9492a614ffe8dc95fc31ec2a3e4`、run `20261010_043203_252187_hardware`，随后用户结束，stopped_verified=true。该轮已启用新巡航/延后绘图配置，但进程加载早于最后停顿修正，仍由旧supervisor传120s。636行中618stop、18非stop，338次执行前过期，执行年龄中位0.3061s；实际持续巡航效果尚未验证，时延仍是下一轮重点。短离线测量不能替代该真实时延。记录 `additional-user-run.json`；检测到运行时未重启服务，结束确认后才加载。此次用户自行启动不作为助手安排的正式性能实验。

## 加载与交付

带状态检查的只读SessionClient核验当时会话finished且停止确认后，执行 `systemctl --user restart vehicle-autodrive-panel.service`。返回available=true、idle、running=false、hardware、runtime_limit_seconds=0。仅启停程序重载，无设备打开/运动请求，无安装或unit配置更改。现有临时user unit仍不随重启自动恢复。下一次用户点击由新进程加载最终源码及新profile；刷新页面获取“无总时限”文案。

产物：`outputs/visual_feedback/20261010/continuous-cruise-fix/`，含before、after、增量patch、源码hash、photo/input hash、profile、回放、校验记录。HEAD仍 `fce828b89b084eda293e83b99c8861f522858d8e`、main、大量原有dirty保留，无commit/push；仅HEAD不能复原当前代码。下一负责人仍为本车端聊天：用户下一次实车记录结束后，优先比较新的执行年龄、连续巡航段与弯前停止原因；完整过弯/整圈、实际停止距离均未验收。
