# 放宽配置后的启动退出：校验规则遗漏

- 日期2026-10-10。用户反馈点击Start立即退出。HEAD fce828b89b084eda293e83b99c8861f522858d8e，保留既有dirty。此次是上轮已授权时效/恢复修复的遗漏补齐，不启动新硬件实验。
- 失败sessions `437d879ab65846eca3250ef842862d46`、`87a367dd0bc840ecb5d22dce10f26380`、`7cb65646aab741da974efa2dc775af0a`、`7da53de733b345c7bb96ead8f5f18f43`。末轮session.json为error/exit1/runningfalse；console.log traceback由onboard.main:1543调用field_trial.validate_stationary_corner_config，报仍需five startup/recovery observations。没有进入模型/相机/底盘构建，不是感知停车或心跳过期；失败轮无runtime终态，因此stopped_verified=false，不虚构停止验证。
- 上一轮profile已resume2/age.6，但field_trial.py遗留最低5帧和YOLO最大.3s。上一轮“执行链没有残留.3s”结论未覆盖启动校验，已撤销完整启动就绪判断；54项控制/感知检查和固定输入回放事实仍成立。
- 本次仅field_trial.py和test_outer_loop_trial.py：enabled visual允许最低2个整数恢复观测、YOLO时效≤.6；旧stationary模式最低5及旧trial≤.3保持。布尔/字符串恢复观测、布尔/NaN/Inf时效拒绝。原标定、PWM、相机.25s、watchdog.4s、记录及人工启动条件保留。
- 测试先重现两处旧校验报错，再修复。`outputs/field_yolopv2/venv/bin/python -m unittest car.test.test_outer_loop_trial car.test.test_visual_feedback -q`：43通过。无设备构建。
- 直接加载实际visual配置并调用两个motors_enabled校验函数，读取未mock的现存pulse证据，均通过；它们虽参数表明检查实车资格，但只读文件不连接设备。另 `outputs/visual_feedback/20261010/startup-policy-fix/preflight.py` 使用最后失败轮实际参数，输出和panel路径隔离到临时目录，运行真实onboard.main的所有相机之前校验，并在CameraSource构造器拦截退出；build_components同时设设备访问哨兵。通过，未打开相机/GPU/串口/电机，没有运行启动shell或操作服务。
- 独立只读补审确认上述遗漏及两函数实际通过，旧模式/disabled模式不获得新权限，无重要问题。git diff --check通过。
- 精确before/after、增量patch、hash/HEAD/失败ids/验证、panel-state及入口哨兵脚本见 `outputs/visual_feedback/20261010/startup-policy-fix/`。没有安装/服务重启/助手Start/提交推送，原始失败记录保持。
- 最后socket仍为历史error/runningfalse，按钮不因error被禁用；下一用户Start新建进程读取已修正校验，无需重启服务。真正打开设备和实车移动/过弯仍由用户下一轮验证。
