# 2026-10-11 云端代码拉取与车端接入检查

- 用户任务：拉取最新代码并检查还缺什么。检查时间约17:30–17:45（Asia/Shanghai）；主机时区EDT。
- 本次完成拉取、只读状态检查、少量无硬件检查和证据记录；没有安装依赖、修改驾驶算法、重启服务、打开硬件、开始驾驶或调用付费云API。
- 结论：云端恢复已有真实代码集成，但当前Jetson配置/依赖未就绪；面板和离线回放未完成新合同适配。不能称已完成云辅助实车恢复。

## 版本和本地修改保护

- 分支main，`bd6fe9c88f6c441eda2f4abbac14054ba3e9570b`快进至`f211ff6c00a770b4c46f5c82b3cae92859f585c8`（feat:添加云端仲裁），与origin/main一致。
- 初次fetch已下载最新提交，但遇到零字节`.git/refs/remotes/origin/main`遮挡旧packed ref；先备份空引用，再将远端引用更新到已核验FETCH_HEAD，随后快进。没有重写本地提交或清理历史对象；git历史gc警告没有触发gc/prune操作。
- 原有PROJECT_INDEX尾部40行追加内容、两份docs改动、四份未跟踪coordination报告及未跟踪panel unit全部保留。拉取前备份及操作结果：`outputs/diagnostics/20261011-cloud-integration-audit/{local-before.patch,before-status.txt,before-git.json,local-before/,empty-origin-main.ref,pull-result.json}`。本次仅新增此报告并追加索引，不提交。

## 当前实现到哪一步

默认visual_feedback profile开启`cloud_recovery`，旧`cloud_arbitration`关闭。`onboard.py`已连接本地控制、当前帧候选、异步RecoveryCoordinator、DecisionFusionAdapter及最终driver。正常有效路线仍由本地控制；持续感知/几何停车至少1秒和3个不同新鲜观测才触发云端。默认取约1/2秒历史及最新帧，最多3图、1个在途请求、最多3次尝试、总请求预算30秒。

合同为`road-recovery-v1`，入口显式指定`qwen-realtime / qwen3.8-omni-flash-realtime`。云端只能选当前本地候选，不能直接给PWM；目前支持短直行与小幅右旋。回复后要更新帧和当前候选复核，通过后一次最大0.15秒/5°软件受限短步，停车至少0.25秒再看新结果。当前融合策略是规则状态机，不是新训练的融合模型。时间/角度上限不证明实际机械位移。

## 确认缺口及处理顺序

1. **先补车端配置和运行依赖。** 实际panel使用`outputs/field_yolopv2/venv/bin/python`，Python3.8.10。用真实CloudConfig加载路径检查：仓库根`.env`存在、权限0600且Git忽略，但`CloudClient._require_key()`不能通过；未配置业务空间或Realtime地址覆盖，`realtime_endpoint()`不能通过。现有panel进程也没有云端变量或自定义env路径。实际venv缺`websocket`模块；`requirements.txt`已有`websocket-client>=1.6.0`，所以是部署环境未补齐。默认开启恢复时，下次驾驶子进程会先因缺有效Key失败；仅补Key之后，缺业务空间/WS依赖仍会使请求失败。本轮不安装或写入认证信息，不能把“代码在Git”当作“云端已可调用”。

2. **适配面板的新状态和新合同。** `car/autodrive/web/dashboard.py:254`仍只映射旧小写phase；新状态`CLOUD_PENDING / REVALIDATE / PROBE_STEP / OBSERVE / LOCAL_DRIVING`会得到unknown。事件读取仅接受32位hex，新RecoveryEngine使用带连字符的UUID，事件目录不会被读取。即使修好ID校验，当前读取`request.event_id`和`response.scene`仍不匹配新`request.context.event_id`、`response.result.scene`；SCENE_FIELDS及前端还未显示assessment/candidate_id/uncertain/hazards和fusion信息。合成快照已复现unknown及空建议；不是仅凭页面猜测。需要保留固定白名单，显式呈现云请求、选中候选、拒绝原因及实际执行来源。

3. **补云参与的离线回放。** `replay_outer_loop.py`和`replay_session.py`仍只重跑本地算法，没有消费新cloud目录的回复、FusionInput/Decision及实际执行记录，也没有按记录时钟注入云回复。回放工具只设置旧cloud_arbitration关闭，不能据此宣称新云融合已验证。需要无需新API调用的记录回复回放，保留候选/事件/帧匹配、当前证据复核、虚拟PWM和停车时序。固定视频只能验证记录视角下的决策一致性，不能证明不同动作会产生相同物理轨迹。

4. **检查控制线程的同步写盘。** `recovery_worker.py`在filter每周期同步打开/写入/关闭decisions.jsonl，`record_application`在实际driver执行后再同步写applications.jsonl。云网络请求已异步，但这两处日志不是异步；可能增加周期延迟或到期停车次数。尚未测量新增延迟，不能将其认定为本次实车卡顿原因。建议采用有界异步记录并保留记录失败的停车语义。

5. **明确触发和能力边界，再采集最小真实恢复证据。** 新触发是持续停车确认，目标方法中的道路掩码时序突变触发尚未在此链路实现；mask_change_threshold用于取消变化后的旧建议。无当前安全候选、黄线/目标否决或模型硬失效时仍不能恢复，加入云端不会自动解决所有弯道。上游报告的真实API两次均使用旧Pi图像、空候选并返回hold；不是当前Jetson候选被选中后自主过弯的证据。先解决1–4，再由用户安排少量当前场地真实云选择→短步→本地恢复记录，保留人工接管。未增加大型测试或新实验。

6. **独立传感器问题仍有一项。** 本次GET现有面板：hardware/idle/running=false、总时限0；相机、电量、IMU、姿态、编码器online。雷达本次已online，约10Hz、1881有效点，不能沿用先前stale结论。深度约26.8Hz但valid_fraction=0、nearest/median为空、status=error；有新帧不等于有有效距离，后续需单独排查。此快照没有证明深度被用作云恢复避障门禁。

## 实际验证与证据

- `git status --short`、HEAD/origin核验、拉取前后差异与备份比对；`git diff --check`通过。
- 实际venv只构造CloudConfig/CloudClient并校验认证形状/endpoint；没有调用request_scene。认证值和业务空间值不输出。只读systemctl show和panel `/proc/PID/environ`仅记录变量名/存在性；HTTP GET `/api/state`复核已有发布数据。
- `outputs/field_yolopv2/venv/bin/python -m unittest car.test.test_cloud_recovery car.test.test_cloud_recovery_contract car.test.test_cloud_recovery_runtime car.test.test_decision_fusion -q`：4个模块均在导入websocket时失败，**没有成功执行这些测试**。没有为测试安装依赖，也没有用假模块遮盖缺失。
- 最新差异中的12个Python文件在实际3.8.10解释器下语法compile通过；一个无硬件合成RecoveryEngine场景通过持续停车→候选回复→forward短步→stop。DashboardState合成恢复快照复现unknown及空结果。不是云网络或实车证据。
- 原始只读面板快照`outputs/diagnostics/20261011-cloud-integration-audit/live-state.json`；轻量核验`verification.json`。运行状态为待机，无本次驾驶run ID。上游实现和历史证据见`coordination/2026-10-10-cloud-recovery-implementation.md`；其中“未发布”描述是发布前历史状态，本次已拉到上述commit，不改写原报告。
