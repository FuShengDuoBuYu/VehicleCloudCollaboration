# 依据人工辅助正确路线修改当前视觉驾驶算法

2026-10-10，车端。用户要求用 b6c 两弯视频与传感器记录修改算法，使当前视角产生正确运动方向。本轮已修改感知、动作选择和实际面板 profile，并完成无设备回放；未启动新实车运动、相机、串口、GPU推理、安装、重启服务或提交。HEAD 仍 `fce828b89b084eda293e83b99c8861f522858d8e`，原有 dirty/未跟踪文件保留。软件候选验证完成，新的自主实际轨迹尚未采集。

## 从数据发现的问题和修改

- 原黄色阈值 `[18,40,100]` 会把暗灰地判黄；改为 `[18,85,135]`。b6c sample350/400 原图近车另有单像素黄色色噪，膨胀后形成停车缺口。现在在源图过滤小于4像素的孤立黄色连通块，连续1像素细线仍保留，然后同样 AREA 缩放和3×3膨胀。两者均不使用历史帧填洞。
- 模型在白箭头、地面折痕、过弯时漏掉当前路面。新增明确 opt-in 的 `perception.track_colors.surface_assist`：只读取模型实际消费的同一张原始图，观察当前黄色边界近侧、连通到车前的灰/白地面，补充 road 支持。横向边界没有观测的列不补；侧边界要求至少图高15%的行同时可见细侧线，新增地面只在该行两边之间，线终止后不外推。侧界模式可保留当前模型已有且颜色支持的地面。绿色、黄色不被补成可行驶路面。
- 第二弯地面出现蓝偏，旧灰地 S≤95 在近车留下少量孔洞；原图漏点 S96–110、V87–112。灰地条件改为 S≤110，保持亮度条件和边界限制，不采用形态闭运算修补孔洞。
- **该模式有意放宽“road 必须完全来自模型”的旧约束。** 新鲜模型仍为必要输入，目标头、帧匹配、错误、全部road异常和时效仍否决；原始road=0仅在当前边界限定的颜色地面得到支持时可恢复。配置关闭时保留原模型支持逻辑。几何 confidence 不是模型概率。
- 前边界还远时继续接近；达到现有路径 lookahead 才确认入弯，方向取当前右出口。初始近方向看直时不会掩盖已经到达的真实右出口。开始转后保留弯道阶段，当前位置方向仍需来自与近车原点连通的当前道路块，孤立道路块不能提供方向。
- 前进检查当前路径与前进带；原地旋转检查当前近车道路支持和转向方向，避免“向前路径暂时拟合失败”把原地旋转也永久否决。两者图像余量不能证明实际车身/轮迹净空。前进或旋转都缺少支持时仍停，未删除实际黄色排除。
- 出弯仍需停止后采集的两个不同新观测、近远方向对齐、前方可走；应用前否决清空确认。连续直道、.15s/5°短转软件上限、.25s等待、PWM≤30、.6s模型时效、相机/IMU时效、取消、记录门禁与总时限0保持。

实际配置 `car/control/vehicle_control/profiles/rosmaster_jetson_yolopv2_visual_feedback.yaml` 已启用 surface assist，下一用户 Start 由新进程加载。面板只读 state 检查为 finished，本轮没有替用户点击开始。当前活动配置和候选的控制内容完全一致，唯一差异为 `vehicle.profile_path` 来源路径；两个完整hash及去除此元数据后的共同hash见 `verification.json`。

## 同输入完整回放

开发参考 session `b6c3246e667542939e40199e5cce8901`，run `20261010_061654_638526_hardware`。原用户End/停止确认，1183周期、840无损输入。原视频、NPZ和各路传感器日志保持原样；CSV和840输入hash与原参考manifest相同。回放消费原时钟/模型序号/执行前许可/IMU，不加载模型、不连接设备，不将人工推车运动写入策略。

| 片段 | 原策略 | 新策略 |
| --- | --- | --- |
| 全1183周期 | 995停、131前进、57弧线右转、0原地转 | 697停、395前进、43弧线右转、48小步原地右转 |
| 直道sample50–150 | 59停、42前进 | 21停、80前进 |
| 直道色噪sample300–400 | 101停 | 23停、78前进 |
| 第一弯sample730–821 | 81停、3前进、8弧线右转 | 70停、1前进、21小步原地右转 |
| 第二弯sample940–1021 | 75停、7弧线右转 | 72停、9小步原地右转、1弧线右转 |

停的数量包含短转之后等待/等新帧、已记录的应用过期和真正缺道路支持，不能直接算错误或成功率。第一弯13行、第二弯37行仍被当前道路/余量拒绝，第二弯并未完整匹配人工参考；不能称已自行过两个弯。新动作也改变后续相机视角，旧录像不预测该变化。

另一原始 session `c92a8904c03c4f76822ef305e532a649`，run `20261010_054444_110581_hardware`，完整465周期回放：331停、95前进、10弧线右转、29小步原地右转。原来提前转向区 sample175–309 没有原地转（30前进/105停），第一条pivot在310；该数字仅为离线检查，不存在实时sample/秒数触发代码。两个回放虚拟PWM均最高30。

## 验证和产物

派生目录 `outputs/visual_feedback/20261010/b6c-policy-fix/`：

- `final-verified-replay/`、`final-c92-replay/`：最终完整逐row动作、PWM、原因、阶段、颜色/道路诊断、回放视频和配置/hash。此前 candidate/rotation/surface/verified 目录是迭代版本，不能当最终结果。
- `comparison.json`、`candidate-timeline.jsonl`：前后对照、两个参考弯及两个直道例子，每row引用原因果传感器索引；原五路日志和hash见 `../b6c-reference-replay/manifest.json`，不重复制作原始副本。
- `candidate-reference-1x.mp4`：按原CSV时间显示原相机和修改后的同输入控制；视频保持最近已记录画面，不插造感知/物理状态。`before-after-output.png`、`candidate-contact.jpg`用于复核。
- `before/`、`source-snapshot/`、`source-sha256.json`、`incremental.patch`、工作树状态：增量以本次修改前的工作树为基线，不是相对HEAD；保留其他已有修改。`build_evidence.py`为无设备派生分析脚本。
- `verification.json`：123项定向CPU检查通过；实际 panel 参数、实际 profile 与标定证据通过启动校验，在 CameraSource 构造处拦截，禁止 build_components，不打开设备。`git diff --check`通过。
- 独立只读复核先复现并修正4项问题：孤立道路方向、关闭守卫绕过配置、两小黄块扩路、横带缺列被当侧界补充；最终原复现消除，无未解决原发现。另核验源图1–3像素色噪被过滤，4像素块和横/竖/斜连续1像素线保留。以上是CPU输入/策略证据，不是实车安全认证。

检查命令：

```text
outputs/field_yolopv2/venv/bin/python -m unittest car.test.test_track_surface_reference car.test.test_visual_corner_guard car.test.test_visual_feedback car.test.test_semantic_bend_path car.test.test_track_colors car.test.test_outer_loop_trial car.test.test_yolopv2_primary car.test.test_control_render_timing car.test.test_outer_loop_replay -q
outputs/field_yolopv2/venv/bin/python outputs/visual_feedback/20261010/startup-policy-fix/preflight.py
outputs/field_yolopv2/venv/bin/python outputs/visual_feedback/20261010/b6c-policy-fix/build_evidence.py
```

两次最终回放通过 `autodrive.tools.replay_outer_loop.replay_recorded(run, output, candidate-profile.yaml)`，源 `summary.json`记录输入、配置和完整性；未来重新执行需新输出目录。要同时重新核验全量传感器和视频/hash，可使用已有 `replay_session.py --profile rosmaster_jetson_yolopv2_visual_feedback`，详见 `docs/visual-feedback-replay.md`，不要覆盖原目录。

用户推动车身/摆正造成的运动无法从这次PWM确定电机动力响应；没有以人手IMU曲线拟合转向时长，也没有学到固定路线时间表。下一负责人仍本车端聊天：用户另行自主实车Start/End采集后，比对真实入弯、第二弯残余停车和轮迹；现在交付的是已启用的、由当前YOLO与同帧场地颜色决定动作的改版。
