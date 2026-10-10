# 移除 LCC 详细诊断页

- 日期：2026-10-08（Asia/Shanghai）；状态：完成并部署。
- 人类依据：用户明确要求移除 LCC 详细诊断入口并删除对应代码；将末尾“diamante”按对应页面代码理解。
- 变更：删除综合页链接及 `/diagnostics` 路由、只读补丁，删除旧 `index.html`、Web server/CLI、`run_lcc_web.py` 和仅服务于旧进程管理器的测试；移除包初始化中的旧模块导入。LCC 控制、驾驶运行时、遥测及 YOLO 算法没有修改。
- 启动：`run.sh` 现在仅启动只读综合面板；旧网页电机环境开关不再生效。`run_vehicle_dashboard.sh web` 支持透传参数，保留 Python 选择并新增通用运行输出目录。正式实验入口仍为 `run_vehicle_experiment.sh`。
- 文档：同步 README、autodrive/control README、Jetson部署说明和面板说明；原设计/实施记录作为历史证据保留，以本次变更为准。

## 验证

- `outputs/field_yolopv2/venv/bin/python -B -m unittest car.test.test_vehicle_dashboard car.test.test_dashboard_sensors car.test.test_dashboard_launcher`：19项通过，0.881秒；全部为临时数据/模拟设备测试。
- `bash -n run.sh run_vehicle_dashboard.sh run_vehicle_experiment.sh`、`git diff --check`通过。生产代码、当前操作文档中不再引用旧 Web 模块或启动入口。
- 临时HTTP通过真实 `run.sh --host 127.0.0.1 --port 0` 启动：综合页/JS/CSS/状态/健康200，旧诊断路径及`index.html`404，POST运动请求403；即使旧`LCC_ENABLE_MOTORS=1`环境存在也仅提供只读服务，`--enable-motors`参数被拒绝，SIGTERM退出码0。
- 只执行 `sudo -n systemctl restart vehicle-dashboard.service` 刷新HTTP；实测 `http://10.20.30.132:8080/`、状态及三路图像200，诊断页404，POST运动403，7项传感器均在线。
- 相机/GPU监控PID291267、串口PID336499、雷达/深度服务PID291268在部署前后相同；仅HTTP从PID298517变为358301。没有本次硬件启停或运动测试。

## 证据与交接

- 证据目录：`outputs/vehicle_dashboard/20261008/remove_diagnostics/`。包含修改前16文件备份/哈希、前后服务与HTTP快照、入口验收、测试日志、本次增量patch和最终manifest。
- 主工作区保留既有未提交修改；旧页面相关文件的先前未提交版本已随备份保留。本轮不提交、不推送。准确HEAD和最终文件哈希见manifest。
- 用户刷新综合面板即可验收；后续本聊天维护综合面板，车端实验继续使用统一实验入口。未自动发送其他聊天。
