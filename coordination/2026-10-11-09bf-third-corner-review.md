# 2026-10-11 09bf：第二弯通过，回中不足与第三弯停车复盘

用户反馈第二弯通过、第三弯卡住，出弯后仍贴左边界；明确确认没有推车、摆正或接管，只在最后点击结束。弯道通过和轮迹按用户反馈记录，车载图像支持左界接近，不能从前视图单独证明每个轮子的接地点。

session `09bf862a43cb477f9247a2d97212d720`，run `20261011_064559_596659_hardware`；18:45:54–18:50:00（Asia/Shanghai）。原始路径 `outputs/autodrive_sessions/09bf862a43cb477f9247a2d97212d720/runtime/runs/20261011_064559_596659_hardware/`。控制CSV1336行、约220.79秒；视频1336/1336且无丢帧。终止为用户End、stopped_verified=true。最后非零动作sample1040，其后295行/41.61秒全停，驾驶会话最终由用户结束。

本轮只读分析、关键原图提取和三帧CPU故障隔离，未改产品控制代码、重启服务、启动设备或调用真实云。HEAD仍f211ff6c00a770b4c46f5c82b3cae92859f585c8，前轮未提交工作保留。派生目录 `outputs/diagnostics/20261011-09bf-run-review/`；analysis.json包含原CSV、配置、状态及云日志SHA-256，analyze_run.py可复算。云decisions/applications各1337条，多于CSV一行，journal2674全部写入；本轮未对该末端额外周期推断原因，也未声称完成整段精确回放。

## 向右规划未持续变成回中

本地角状态首弯250进入turn，523 exit-confirm、524 cruise；第二弯653进入turn，917 exit-confirm、918 cruise。两段turn中原地右转分别20/19条。云第一次请求直到1079才出现，所以前两弯由本地控制完成。

上一轮把front_observed置位移到横向前沿检验之后，已使两次出弯有真实下发差速：524为左30/右22，918为左30/右21。524–652共27条差速右转，918–1040共36条；对应段IMU变化约-.191/-.294rad，支持确有右向动作，不应说所有向右指令都没下发。与此同时，用户看到的“部分规划指向右、实际却直行”有两个具体代码依据：

1. `onboard.py::_filter_visual` 的forward_in_corner、approach和near_aligned短步仍进入desired=forward；最终forward输出把steering清零、左右速度设为同一min值。250–523、524–652、653–917分别有21/7/22条前进，在lateral_error>.15时仍左右同速，合计50条。比如500误差+.31574、heading+.19553却是30/30；弯中前拱没有保留路径纠偏。上一轮仅解决错误front_observed激活approach，没有修完所有同速覆盖分支，不能将本次持续偏左说成已解决。
2. `SemanticOuterRoute._front_corner_preview` 的接近点直接取forward_x=w*.5；RoadCenterlineEstimator收到preview后把路径终点约束在这个点附近。第二弯出弯后918仍有+.317横向误差，950仍+.270；到960进入下一弯approach，preview设在画面中央，lateral/heading/near_heading均变0、输出30/30。这只表示当前选出的中心射线路径近似直，不证明车身已回到道路中央。类似变化在首弯后的接近段也存在。

回中目标需要和弯道接近目标分开维护，短前拱应保留经过当前路径余量复核的有界差速，而不是只在cruise保留。当前缺少新实车验证，不能仅凭上述修改方向断言会彻底消除贴线。

## 第三弯先被车端障碍框否决

1041先有一次external_allowed=false停车；1042/1043处于重新确认和等待，1044起云融合输入持续出现near-field obstacle。1041–1335中276行该原因、16行当前证据不可用、3行无此否决。局部黄线否决不是该末段主要原因。

三个请求对应消费原图里，左侧是位于黄线外侧的行李箱，右侧道路及大右转箭头清晰可见。实际检测只有一个class_id=3框；不在没有类名映射核验的情况下赋予其模型语义名称。框约x=[0,.214]、y=[.054,.679]、confidence=.37–.46。`cloud_recovery.build_evidence`用confidence≥.3、y2>.65、x1<.8、x2>.2的宽矩形规则判障碍；框右缘仅超过20%位置约1.4%图宽就触发，未按当前候选走廊/黄线内外关系检查。相比之下，`yolopv2_semantic.py`的前向障碍范围是40%–60%，同一个框不触发该层，所以semantic_hard_safe仍true，云融合hard_safe却false。

三帧隔离脚本isolate_obstacle_veto.py保留原frame/mask/lane/IMU/时刻，仅在内存复制上去掉唯一检测框（明确标为反事实），3/3复现：真实输入near-field obstacle、空候选；去框后出现当前forward-gap和pivot-right候选。当前颜色道路几何的前向余量约40px、要求20px，旋转原点余量约59–90px、要求16px。该隔离证明阻断发生在检测框门禁，不证明删除检测就是安全修复，也不把像素余量当成完整车身/扫掠空间。

## 云确实回复，全部hold，没有右转指令

|请求sample/事件|消费回复sample|耗时|实际回复|
|---|---:|---:|---|
|1079 / 49acadee-ebe1-4778-a3d7-9b16ac9b2084|1108|3738.962ms|obstacle、hold；前方障碍且无候选|
|1109 / 631fba6f-3fdb-4662-9a1c-95b3fed5a663|1138|3754.015ms|obstacle、hold；hazards含obstacle/yellow_boundary|
|1138 / 6e02299d-8627-4968-9841-823a9293ca7d|1165|3653.583ms|unknown、hold、uncertain=true；视线受阻/无候选|

三次全是diagnostic_only=true、candidates=[]、motion_veto_reason=near-field obstacle。`RECOVERY_PROMPT`明确只允许选择车端实际提供的候选ID，禁止发明动作；空候选不能返回可执行右转。大箭头提供道路方向线索，但当前恢复接口不是独立生成路线的规划器，也没有单独的箭头观测输出。云回复对“左侧路外物体、右侧道路”的表达并不充分，但无法从短回复确定模型是否没识别箭头；本地先带着障碍否决和空候选上传是确定事实。

三次请求分别在1094/1111/1140因fusion current evidence veto取消，均早于回复；最终attempts=3、observation attempt budget exhausted。原始road抖动取消修正没有回退，此轮触发的是外部门禁/证据不可用。即便这些hold回复没有被提前取消，它们也不会许可前进或右转；即便增加重试，也仍会遇到同一障碍输入。末态response_status=cancelled而不是旧版pending，最近事件ID保存正确。

## 后续修复方向

本车端聊天后续应一起处理：短步保留路径差速；接近弯道时不要把图像中心当作道路中心；统一当前路径相关的障碍判定并记录具体否决框；让云分别报告箭头/可见道路和是否支持候选，清楚标识本地否决为待复核证据。障碍、黄线和当前证据门禁仍需要保留，不能直接屏蔽行李箱类别或看到箭头就强制转向。

依赖顺序为CPU片段回放证明纠偏不被覆盖、路外检测不无条件清空候选且实际阻挡仍能停车，再由用户安排下一轮实车，车端根据新录像/编码器/IMU/PWM给出对照结果。当前任务交付为已复现根因与准确证据，尚未实施这一轮控制修改。
