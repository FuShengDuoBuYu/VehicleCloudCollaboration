# 2026-10-11 云端接入补齐与实测准备

- 人类任务：保存提供的云配置、补齐依赖、修正面板和云回放、改异步写盘，再由用户启动实车。用户随后明确同意图像外发，以及仅重启vehicle-dashboard.service。
- 版本：main HEAD `f211ff6c00a770b4c46f5c82b3cae92859f585c8`，本轮产品修改未提交/推送；保留此前全部本地资料和unit。17:58–18:00（Asia/Shanghai）完成面板重载和最终核验。
- 结论：本次请求的接入修复和部署准备完成；实体车仍待机，没有助手启动运动。真实云协议已通，固定输入的模拟云回放可复现控制输出；当前场地自主过弯仍由后续实车验证。

## 修改和运行配置

1. 提供的云设置已更新仓库根`.env`，权限0600、Git忽略，不在此报告或源码存认证值。保留通用`road-observation-fast-v1`设置；实际云恢复入口仍显式使用`qwen-realtime / qwen3.8-omni-flash-realtime / road-recovery-v1`，两者用途不同。真实CloudConfig加载、Key形状和业务空间endpoint校验通过。
2. 实际驾驶Python3.8.10 venv安装`websocket-client 1.8.0`和`Pillow 10.4.0`。先前Pillow7.0缺Image.Resampling，首次针对性检查由此失败，修复依赖后通过。系统Pillow未卸载；没有改Torch/CUDA/OpenCV/ROS或安装整套最新版requirements。`pip check`仍报告system-site-packages的既有launchpadlib缺testresources，与云/驾驶入口无关，未因此安装额外开发组件，不声称整台系统所有包依赖均无缺口。
3. `dashboard.py/js`兼容新phase、UUID、request.context及response.result.scene；输出固定白名单，呈现assessment/candidate_id/uncertain/hazards和fusion source/candidate/reason，兼容旧场景合同。只读取当前run的事件文件，不传出原始响应/认证内容。
4. `RecoveryCoordinator`使用有界AsyncRecoveryJournal，控制线程只提交快照，不做JSON编码、文件打开或写盘。后台保持两类记录的顺序，队列溢出或I/O失败锁定错误并通过原仲裁停车，不覆盖旧记录。结束流程原本先driver.stop，再close排空队列；关停排空有5秒上限，异常进入cleanup记录。归档新增决策序号、原本地命令/许可、收到的云回复和消费时刻、原始IMU及journal完整状态。
5. 新`tools/recorded_recovery.py`由原`replay_recorded/replay_session`调用，重新运行现有规则融合与最终虚拟driver；按记录消费时钟注入云回复，保留当前几何/黄线/IMU门禁。要求请求帧/候选/mask及原图一致、输入PNG hash有效，不匹配保持停车并计数。动作/PWM比较和融合状态写入输出。云启用归档缺日志、记录未排空或决策/执行顺序不完整时拒绝。旧无云轨迹不伪造云回复；旧缺消费时间的云日志明确标为worker完成时间近似。

## 实际验证

- 轻量回归：实际venv运行`car.test.test_cloud_integration_completion`、`test_cloud_recovery`、`test_cloud_recovery_contract`、`test_cloud_recovery_runtime`、`test_decision_fusion`、`test_stationary_corner_replay`、`test_session_replay`及3个原dashboard云白名单/事件测试；**42项通过，14.220秒**。不打开硬件、不做GPU推理或真实API调用。新增3项对应旧面板格式、阻塞I/O/队列满和记录云回复/当前候选撤销；原主循环模拟测试扩展完整回放。
- 模拟主循环归档→当前控制器→已记录模拟云回复→最终虚拟四轮：25周期，2个匹配云事件、2个回复，动作25/25、四轮PWM25/25一致，最大PWM30；包括两个前拱和到期停车。此数据是模拟模型/云，绝不是实车轨迹。保存原模拟归档和派生回放，原临时路径仍作为采集来源保留，不伪装成现场run。
- 人类批准外发后，使用既有监控图像的本地副本只做**1次真实云请求**，没有新开相机。空候选、恢复合同返回hold，总耗时1357.191ms。证明配置/认证/图像编码/协议和结构化返回可用，不证明恢复运动、语义准确率或尾延迟。
- 人类批准后，仅`sudo -n systemctl restart vehicle-dashboard.service`；先GET确认待机，再检查新dashboard.js已提供执行来源字段。未重启panel/相机监控/串口/ROS采集，未调用Start/Stop驾驶接口。
- 部署状态：available=true、hardware/idle、running=false、runtime_limit_seconds=0；相机/雷达及底盘四项online。深度有新鲜原始帧但valid_fraction0/status=error。本地记录门禁检查原始帧写入新鲜度，不检查有效距离比例；不能称深度正常或已进入避障链路。待机monitor自身cloud=disabled是既有监控配置，下一次驾驶子进程才加载云恢复。
- `git diff --check`通过；秘密未进入Git、产品代码、报告或派生产物，认证信息只由本地.env提供。实测仍由用户手动开始，不续跑旧会话。

## 产物与下一次实测

- 证据目录`outputs/diagnostics/20261011-cloud-ready/`：environment-check.json、connection-input.jpg、connection-result.json（客户端脱敏）、verification-tests.txt、deployment-state.json。
- `roundtrip/synthetic-run/`是模拟源数据，`roundtrip/synthetic-replay/`是派生输出；其中summary.json保存完整匹配统计，commands.json保存逐周期动作/四轮/fusion。本轮实际驾驶run ID：无。
- 实车继续使用`rosmaster_jetson_yolopv2_visual_feedback`，云恢复默认开启、PWM30、短步0.15秒/5°、停车观察0.25秒、sparse3、请求预算30秒、每事件最多3次；总运行时限0。先刷新网页，再由用户安排封闭场地低速运行并手动结束。
- 每次实车结束后，使用docs/visual-feedback-replay.md中的同一replay_session命令，即可读取真实cloud日志而不再外发或调用API。把人工推动/摆正发生时间与失败位置另作标注。云无安全候选时仍停车，不能靠放宽黄线/新鲜度门禁保证过弯。
