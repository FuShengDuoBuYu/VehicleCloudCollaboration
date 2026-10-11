# 2026-10-11 05a3：首弯通过，出弯回中与云回复取消复盘

## 本次结论与证据范围

用户明确反馈本次主动结束、第一弯成功且运动方式符合预期；随后持续偏左、压左线、右边界难以获取，最终停车。首弯通过和车轮压线分别作为用户实车反馈，车载前视图支持出弯后左边界接近，但不能单独确定全部车轮接地点。

session `05a3c25bc5784be48c9e22fd39d6073b`，18:27:19–18:29:31（Asia/Shanghai），run `20261011_062723_573638_hardware`；operator clicked end、exit0、stopped_verified=true、cleanup_errors为空。867控制周期覆盖112.418秒，首条控制到sample441约56.98秒进入cruise；最后运动sample669，随后约25.91秒全停，最终由用户结束，并非自动退出驾驶会话。

HEAD仍`f211ff6c00a770b4c46f5c82b3cae92859f585c8`，上一轮修正及既有未提交改动保留。本轮只分析原记录、CPU离线回放与保存报告，没有修改产品算法、启动设备、重启服务、安装依赖或外发新请求。

原始run完整路径：`outputs/autodrive_sessions/05a3c25bc5784be48c9e22fd39d6073b/runtime/runs/20261011_062723_573638_hardware/`。派生目录：`outputs/diagnostics/20261011-05a3-run-review/`，包含analysis.json、mask-change-pairs.json、消费帧原图副本、replay/。原视频/CSV/NPZ保持原样，analysis.json记录CSV、配置、状态和云日志SHA-256。

## 出弯偏左：纠偏被接近弯道的直行分支覆盖

sample441出弯后恢复连续驾驶。到480，lateral_error已为+0.32461，控制确有向右修正（左PWM30、右21）；到485误差+0.39257，随后当前前進检查受限停车。正误差表示图像目标在右方，不是米制横向距离。

首次云小转后重新前进，542–674共33条运动输出，其中19条左右相同PWM30/30的直行，14条同时满足：lateral_error约+0.305～+0.339、semantic_front_observed=true、semantic_corner_reject_reason=`front-spread-too-large`。具体sample：542、556、569、575、581、587、593、599、604、610、616、629、635、641。

代码依据：`semantic_outer_route.py::_front_corner_preview` 在验证跨列前沿高度差之前就设置front_observed=true；差异过大虽返回front-spread-too-large，标志仍然为true。`onboard.py::_filter_visual`把它用于approach和禁用cruise；随后desired=forward，短步输出左右同速、steering=0。因此斜侧边界/不成立的横向前沿仍可能激活接近弯道逻辑，覆盖本该持续的回中纠偏。没有充分依据把偏左单独归因于电机、底盘标定或YOLO完全未发现偏差。

near_heading较小并不代表车已居中。后续应区分可信横向前边界、侧边界和普通出弯回中；当前安全前拱也要保留合理的横向反馈，不能只允许连续巡航时纠偏。首弯小转/前拱已有效，应保留这部分实车进展。

## 四次真实云请求均有回复，第一次实际执行

秒数相对首条控制记录；请求模型为qwen3.8-omni-flash-realtime，合同road-recovery-v1。四次均perception_stall、diagnostic_only=false，非上一轮新增的黄线仅观察分支；源文件为run/cloud/<event>/request.json、response.json。

|请求sample / 相对秒|云请求耗时|回复|车端结果|
|---|---:|---|---|
|496 / 63.51|3817.797ms|选择pivot-right，认为右方连通|531实际执行一次右旋，PWM30/-30；542恢复本地|
|687 / 88.76|3819.510ms|选择pivot-right|689即取消，回复到达后忽略|
|761 / 98.49|3859.205ms|选择pivot-right|762即取消，回复到达后忽略|
|792 / 102.61|3538.962ms|选择pivot-right|793即取消，回复到达后忽略|

第一请求event `e20ba6fb-3f35-4ef5-9592-edfbbcb7c005`；后续依次`194351ff-030d-4384-a047-ea6934662559`、`97410f37-d37a-4431-ac40-a1d2c633aedf`、`17978808-a708-4b9b-9766-7881d8561877`。第一回复执行后IMU约0.2897→0.2119rad，约4.46°右转，并有后续本地运动，不能称云端完全没有下发/参与。

图像可见左边界接近、道路在右侧连通，建议小右转有方向上的依据。四次均uncertain=false、hazards=[]并声称余量充足，不能当作实测车身/轮迹安全结论；缺失右边界时云也无法证明完整可用宽度。第一请求实际只提供pivot-right一个候选，云并未独立规划回中轨迹。请求2–4提供forward-gap和pivot-right两个候选。

## 后三次失败是掩膜抖动触发scene changed

`cloud_recovery.py::RecoveryEngine.tick`比较当前与请求时原始YOLO road掩膜的1-IoU，超过0.4就veto，并清空事件/请求/回复。后3次在发出后0.319、0.142、0.132秒即被取消，早于约3.5–3.9秒的云回复。

|比较sample|1-IoU变化|原图平均通道差（0–255）|编码器四路差|IMU差|
|---|---:|---:|---|---:|
|687→689|0.443|4.262|全0|-0.0003rad|
|761→762|0.852|4.255|全0|-0.0001rad|
|792→793|0.964|4.251|全0|-0.0002rad|

三个比较中，当前本地候选列表完全相同、hard_safe均为true。原始road比例分别1.69%→0.941%、0.245%→1.654%、0.066%→1.823%；当前候选道路却还得到同帧颜色地面支持。故稀少且抖动的原始模型像素被当成了整体场景变化，和融合后的可执行几何并不一致。图像/编码器/IMU联合证据支持误取消，而非车辆实际大幅移动；不以编码器独立证明所有物体绝对静止。

第二次恢复事件三次尝试耗尽，末态reason=`event attempt budget exhausted`。675–866的192行全停，其中174行拟合中心线离开语义走廊，18行又恢复前角路径；因此本地也未满足稳定恢复。末段yellow_hazard=false，并非这轮黄线硬停车分支。日志中部分重复的`fusion current evidence veto`是HOLD阶段沿用的原因，不能将每一行都解释为当时传感器过期。

下一步应将原始掩膜变化作为异常线索，结合当前候选、黄线/目标/新鲜度和时序确认决定是否取消；不能简单无限重试、取消当前安全检查或直接用云指令驱动。取消与尝试预算也应区分“回复无效”和“本地感知抖动导致未消费”。

## 面板为何看起来仍没有云结果

veto清空event_id；dashboard仅按当前event_id读取事件目录，取消后的已归档回复因而不再显示。末态已worker_busy=false、closed=true、pending_jobs=0，但response_status仍为pending。这是事件显示与状态维护缺口，不是网络未返回。下一步应保留最近请求/回复/取消原因和执行结果，分清“收到但未执行”与“尚未返回”。

## 记录与验证

视频867/867、CSV867/867，无丢失/错误；云异步journal1734/1734，队列排空、closed=true/error=null。本轮没有做广泛测试。离线使用实际venv，`PYTHONPATH=car OMP_NUM_THREADS=1`、OpenCV单线程，调用replay_recorded(profile='recorded')，复用真实归档回复，不联网、不重推模型、不打开硬件。最终统计见派生replay/summary.json及本报告后续核验记录。

最终回放核验：完整867周期动作867/867、四轮PWM867/867一致；匹配云事件4、消费归档回复4、未匹配0、网络请求0。真实云请求/回复/拒绝链路已在固定输入下复现，见派生replay/summary.json。

本轮性质是确认实车进展、定位下一阶段缺陷；上述回中、场景变化与面板状态修正尚未实施。下一负责人为本车端聊天。
