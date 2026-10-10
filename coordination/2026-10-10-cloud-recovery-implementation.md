# 持续感知停车的云端辅助恢复交付

发布状态更新（2026-10-10）：用户明确授权提交，指定commit标题 `feat:添加云端仲裁`，本记录随实现一并提交。尚未推送、部署或实车验证。下文“未提交/待review”描述的是v1/v2冻结时状态，原冻结包不改写。

任务：CLOUD-RECOVERY-01；2026-10-10，Asia/Shanghai。状态：本地实现和无电机验证 verified；车端部署/实车未验证。负责人：本地实现与评测。本轮没有连接小车、启动电机、发送其他聊天消息或提交/推送Git。

## v2 增补：统一决策融合模块（本轮最新）

用户追加要求统一本地与云端融合接口，并明确未来“jev”替换问题后续再说。现已新增 `car/autodrive/control/decision_fusion.py`：FusionInput / CloudAdvice → DecisionFusionAdapter → FusionDecision。当前默认 `RuleBasedRecoveryPolicy` 包装已有RecoveryEngine；协调器仅负责排队、网络和归档，实际onboard本地提案与云恢复均通过同一个接口。adapter规范接口和校验输出，policy承担规则融合判断；不是缺少仲裁逻辑后再引入一套竞争仲裁。

统一输出记录来源local/cloud_recovery/hold、理由、候选ID、事件ID、消费帧序号、有效期和policy ID。对硬条件失败、无截止/过期输出、凭空改写本地提案、没有实际回复支持的云候选、策略异常一律停车；同一云事件截止不能滚动续期，也不能消费两次。最终driver拒绝同步撤销融合许可。fusion字段写入decisions/applications和已有网页cloud状态。

策略通过 `RecoveryCoordinator(..., fusion_policy=...)` 注入，纯接口实现无需继承RecoveryEngine；尚未接入任何新模型，也没有“填模型名即可替换”的配置承诺。新的策略必须保留几何/IMU/预算约束并重新验证。接口和结构图见 `docs/superpowers/specs/2026-10-10-decision-fusion-interface.md`。

本轮最终回归 **219项通过，9.865秒**（原212项+7项融合接口测试，主循环和原直行/右旋测试也通过）；独立只读审查未发现可复现P1/P2，另行验证正常本地短步不再形成恢复死锁。本轮没有新的付费API/硬件测试，后文2次API仍属于v1组件证据。最新源代码/差异/验证冻结为本地论文目录 `backups/cloud_recovery_implementation/20261010_v2.zip`；v1归档保持原样。

车端接续不增加新的操作步骤：发布后仍使用同一profile，日志应出现 `fusion.policy_id=rule-recovery-v1`，检查普通驾驶为local、云选候选短步为cloud_recovery、拒绝或停车为hold。当前仍未提交推送，待用户review及明确发布授权。

## 来源、授权和版本

用户明确批准设计并要求本聊天实现，车端由用户拉取后再测试。基线 `bd6fe9c88f6c441eda2f4abbac14054ba3e9570b`，产品修改前仅本任务设计文档未跟踪。代码现为未提交工作树；精确差异、源文件哈希、真实API原始记录与验证记录冻结在本地论文目录 `backups/cloud_recovery_implementation/20261010_v1.zip`，清单同目录。未将未发布状态说成可直接git pull。之前“先不要上传git，我会review”的限制保留，发布需要用户确认。

## 已实现

1. 当前默认车端 profile `rosmaster_jetson_yolopv2_visual_feedback` 启用 `cloud_recovery`，旧 `cloud_arbitration` 保持关闭；两者不能同时启用。云配置显式选择 `qwen-realtime / qwen3.8-omni-flash-realtime / road-recovery-v1`，不依赖通用CLI的fast观测合同。启动时缺Key、缺归档或不兼容配置会失败，不能静默变成无云驾驶。
2. 只在持续感知/几何停车达到1秒且至少3个不同新鲜观测时异步调用。独立本地路线证据与云等待状态解耦，修复循环等待。正常短步停顿、出口确认、传感器/记录失败不触发请求。每事件最多3次，一个在途请求，失败冷却2秒；超时后的旧线程结束前不再发新请求。
3. 默认 `sparse3`：与YOLO结果绑定的最新原图，加约1秒和2秒前不同消费帧，按旧到新发送。容差0.45秒、最多40条/3秒历史；缺历史明确少发，不复制补齐。`single` 可用于配对实验。每张保留输入时间、序号与hash，候选只属于最新帧。
4. Realtime独立Manual会话支持恢复合同1–3图，JPEG/base64上限沿用；图间间隔1秒，合成静音只作传输载体。默认总请求预算30秒，**没有2秒硬截止**。保留原单图fast合同和HTTP接口。
5. 输出固定字段 `schema_version, assessment, recommendation, candidate_id, uncertain, hazards, reason`。未知候选、语义与候选不匹配、不确定、有危险、字段非法、过期和错事件均不能授权运动。模型不返回PWM/角度/时长；本地绑定事件、候选版本及hash。
6. 本地从当前原始road/lane、同帧灰白地面/黄线和目标检测生成候选。短直行要求近场中心78%–98%高度、左右6%宽度内支持及余量；仅当前灰白地面支持可以重新解释lane排除，黄线始终保留。右旋要求当前连通地面近车原点余量、右侧方向支持、已验证旋转映射及新鲜IMU。额外近场目标检测覆盖候选区域。候选允许绕过失败的路线拟合或阶段解释，但不能跳过 `semantic_hard_safe`，无几何时保持停车。
7. 回复之后必须有更新且捕获时间更晚的模型帧；同候选当前重新生成并通过场景变化检查后，仅执行一次最大0.15秒/5°的软件受限短步，PWM上限30；随后至少0.25秒停车且等待停止后的新帧。时间截止交给原最终driver及独立motion guard，IMU上限由控制周期撤销。软件限值不等于真实机械位移/角度保证。
8. 当前默认profile的正常驾驶仍由原本地控制器执行。复核连续有效的本地路线证据后才关闭云事件；正常step/settle零指令不再清空证据计数，但永不把本地正常stop强行改成运动。

核心文件：`car/autodrive/control/cloud_recovery.py`（候选/状态机），`recovery_worker.py`（异步/归档），`car/autodrive/runtime/onboard.py`（集成），`car/cloud_client/recovery_contract.py`、`realtime.py`、`contracts.py`、`cli.py`（合同与传输）。本次只做接入所需拆分，没有开展整仓重构。

## 验证记录及边界

- 最终回归 **212项通过，12.311秒**，Python 3.11，Windows；覆盖既有YOLO/颜色/弯道/回放/云客户端、HTTP与WS本机服务器，以及新增恢复测试。命令与模块列表保存在冻结包 `verification.json`。
- 新合同先出现“不支持合同”、新状态模块先出现“模块不存在”的预期失败，再实现并通过。新增测试覆盖不同颜色三图的发送顺序、输入边界、未知候选、错事件、过期/重复帧、场景变化、硬条件撤销、有限预算和忙线程不重发。
- 模拟链路覆盖真实 `build_evidence → RecoveryCoordinator worker → RecoveryEngine → apply_runtime_command → SafeWheelDriver` 的短直行、右旋及IMU上限停车；取消时在途结果不能复活执行。实际 `onboard.main` 也以灰色视频、模拟YOLO和模拟云运行，确认有虚拟PWM、零输出、完整归档和正常关闭，明确断言未创建底盘。
- 独立只读审查发现正常短步停顿导致恢复计数清零的问题，已修复并加入真实StationaryCornerRuntime节奏回归。审查未发现其他高置信度P1/P2。随后补足右旋完整链路、取消和不同图顺序测试。
- 首次部分回归遇到基线也出现过的Windows沙箱 `status.tmp → status.json` WinError5；相同无硬件测试在获准沙箱外复核通过。网络本机服务器同样需要沙箱外运行。不是实车故障结论。

真实API只做2次协议组件调用，原始文件：`outputs/cloud/recovery/20261010T132850Z/`，输入来自旧Pi运行 `20260803_211052_468830_hardware`（5秒当前图；三帧4/4.5/5秒），与默认运行时约1/2秒稀疏历史不同。最新Jetson原始场景尚不在本地；**没有捏造其当前mask或候选**，context candidates为空，两次均hold。

| 输入 | 本次总时间 | 首内容 | 输入/输出tokens | 结果 |
|---|---:|---:|---:|---|
| 单帧 | 1.531秒 | 1.062秒 | 756 / 69 | hold |
| 三帧 | 3.250秒 | 2.782秒 | 1129 / 54 | hold |

每种只有1次；不能据此判断准确率、尾延迟、三帧优于单帧或真实恢复成功。usage详分、会话事件及raw_response均保留，未把视频token数等同于模型完整理解每帧。没有新增最新场地语义评测结论。

## 车端接续执行

前置：用户review本地代码，明确发布后由本地提交/推送，车端才拉取；先检查车端工作树，保护未提交实验改动。Git不包含API Key、当前旋转标定证据和历史视频。

1. 车端在仓库根 `.env` 配置 `CAR_CLOUD_API_KEY` 和北京业务空间 `CAR_CLOUD_WORKSPACE_ID`（沿用现有配置方式；不要把Key放入YAML/日志/Git）。运行时会显式选定模型和recovery合同；`.env` 的通用fast观测默认不影响此入口。Python依赖沿用已接入的websocket-client、Pillow、numpy、OpenCV等；先只读核对环境再决定是否需要安装。
2. 读取 `car/control/vehicle_control/profiles/rosmaster_jetson_yolopv2_visual_feedback.yaml`。网页启动的正是该profile，现已启用云恢复。纯本地基线实验将 `runtime_overrides.cloud_recovery.enabled` 改为false并保存配置快照；配对单图实验只改 `frame_policy: single`。禁止同时打开旧cloud_arbitration。保持当前已验证轮序、IMU yaw_sign和旋转标定证据路径。
3. 先做无电机视频/回放和现场架空检查，检查事件目录、模型连接、超时/断网/取消保持零输出及driver截止。示例无电机视频入口：`python -m car.autodrive.runtime.onboard --video <本机视频> --vehicle-profile rosmaster_jetson_yolopv2_visual_feedback --max-samples 100 --output-dir outputs/cloud_recovery_precheck`。不要加 `--enable-motors`，不要把这个步骤说成实车验证。
4. 由用户安排封闭场地最低速实车。先分白标记、右弯、真实障碍/黄线保持停车三类单场景，再尝试完整圈。记录人工推动/接管；人工帮助不能算自主恢复。相同场景比较禁用云、single、sparse3，但需按独立run划分开发和最终测试，不能拿相邻帧当独立样本。
5. 产物：commit+dirty patch、完整有效配置、run ID、语义输入原图/模型mask、原始车载及外部视频、每事件request/response、decisions.jsonl、applications.jsonl、原控制CSV、IMU/轮PWM/取消时钟、人工干预与失败原因。云目录位于对应运行archive下 `cloud/`，不要只返回截图或成功例。

验收顺序：车端回传证据 → 本地分析触发/候选拒绝/响应时延/短步后实际恢复和接管 → 总纲审查方法与论文结论。当前不声称完成一圈、真实不压线、5°机械转角保证或全部停车均能恢复。无候选、黄线/目标、模型硬失效仍会保持停车；后续需要更强空间输出时另行定义合同。
