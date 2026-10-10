# 第二弯按当前画面逐步转向修正

2026-10-10，车端；承接用户对上一版第二弯仍失败的反馈，以及“当前位置识别要转弯，小幅逐步右转、边转边看”的要求。本轮修改代码并完成同输入回放，没有启动新实车运动、打开设备/GPU、重启服务、安装或提交。HEAD `fce828b89b084eda293e83b99c8861f522858d8e`，115项既有修改/未跟踪条目保留。

## 原因与最终改动

- 原地转向原来检查图高.92–.94的前方位置，以及宽黄线区域。第二弯前方/侧方黄线进入这些窗口时会拒绝转向，即使最近可见地面仍可行驶。现在前进继续验证完整近路径与前进带；原地小转单独检查最近可见地面及其当前连通道路方向。原点为图像底行中心，支持区域y≥.98、x.49–.51，当前支持≥90%、无黄色，旋转图像余量≥图宽.05。前进余量仍为.06。它是图像代理，未标定为车身/轮迹占地。
- 黄线搜索顶部.40→.30，只扩大当前可观测边界范围，.62入弯触发不变。明亮蓝偏白打印H95–140、S≤110、V≥170可作为同帧颜色地面，保持实际黄线近侧/连通/缺列限制及灰地条件。
- 强光浅黄与原黄色先在源图合并，但只有包含至少4个原强黄色像素的当前连通块才能保留浅黄。原强黄HSV `[18,85,135]`–`[42,255,255]`，浅黄 `[10,30,185]`–`[42,255,255]`。不连接不存在的像素，不使用历史种子。源图小连通块过滤、AREA缩放与3×3膨胀保留。
- surface assist模式下，模型已确认、且同一消费图颜色支持为灰白的地面不再被lane纹理层截断。无当前黄边界时，免除纹理后仍是当前模型道路的子集；模型缺口不会仅凭灰白颜色补成道路。已有边界补路仍受原限制，黄色始终覆盖许可。模式关闭保留旧策略。
- 转向由当前前边界接近、当前右出口和当前近车连通道路决定；每步看新帧，允许前拱或再次小转，两个新对齐观测才确认出弯。不按录像秒数、样本序号或人工IMU曲线设置动作。模型/目标/帧匹配/时效/执行/记录/取消门禁保留，PWM≤30、单步.15s/5°软件上限、等待.25s、总时限0未改。

## 完整回放结果

最终目录 `outputs/visual_feedback/20261010/b6c-second-corner-fix/final-source-replay/` 与 `final-source-c92-replay/`。其他 replay 目录为保留的迭代版本。独立浅黄范围曾使直道退化到848停/239前进，已弃用，不能当最终结果。

参考session `b6c3246e667542939e40199e5cce8901`，run `20261010_061654_638526_hardware`，1183周期/840无损模型输入。原CSV/NPZ hash与 `b6c-reference-replay/manifest.json`一致，原始视频和五路传感器日志不改。比较基线为上一任务 `b6c-policy-fix/final-verified-replay/`，不是最初原算法。

| 片段 | 上一版 | 本次最终版 |
| --- | --- | --- |
| 全1183周期 | 697停/395前进/48原地右转/43差速右转 | 593停/491前进/59原地右转/40差速右转 |
| 直道sample50–150 | 80前进/21停 | 92前进/9停 |
| 直道sample300–400 | 78前进/23停 | 98前进/3停 |
| 第一弯sample730–821 | 21原地右转/1前进/70停 | 21原地右转/1前进/70停 |
| 第二弯接近sample850–939 | 38条evidence veto | 15条evidence veto，35前进/4差速右转/4原地右转/47停 |
| 第二弯sample940–1021 | 37条evidence veto，9原地右转/1差速右转/72停 | 0条evidence veto，19原地右转/2前拱/61停 |

数字是控制周期数，不能当转动步数、成功率或物理转角。第二弯剩余停为26条等待停顿后新帧、26条等不同模型序号、6条单步时长截止、3条出弯确认。sample1017–1018前拱，1019–1023等待当前方向确认，1024确认cruise但仍等停顿后的新帧，1025–1040连续差速右行；没有持续道路拒绝锁死。

接近段仍有5条当前道路/路径支持拒绝（859、883、884、923、924）和10条恢复等待，**整条参考路线尚不能称已完全匹配**。第一弯9条evidence veto中，5条原外部执行许可为false（763、787、799、800、801），另4条恢复等待；这些行道路hard-safe为true。原许可合并记录/看门狗/云门禁，原CSV不能确定具体子原因；未强行改成true。

旧提前转向session `c92a8904c03c4f76822ef305e532a649`，run `20261010_054444_110581_hardware`完整465周期：298停/113前进/12差速右转/42原地右转。sample175–309无原地转，第一条仍在310；这仅是回归窗口，不是运行触发规则。

## 核验与可复用证据

- `build_evidence.py`以原时钟及因果传感器索引制作对照，断言第二弯零evidence veto、出弯连续巡航、旧记录无提前pivot、非零输出的当前/执行许可、PWM≤30及输入hash一致。
- `candidate-reference-1x.mp4`按CSV实际时间播放，1008帧/10fps/100.8s，完整解码通过；左右为参考相机与同输入新策略，不预测新动作后的相机或物理轨迹。
- `comparison.json`、`candidate-timeline.jsonl`、`before-after-output.png`、`candidate-contact.jpg`保存逐帧原因、阶段、PWM和原传感器样本引用。`closest-ground-diagnosis.jpg`与`approach-support-diagnosis.jpg`为原图诊断。
- `before/`、`source-snapshot/`、`source-sha256.json`、`incremental.patch`保存本轮工作树增量及依赖；`track_parameters.json`保存三组颜色/旋转设置；`verification.json`保存实际配置、HEAD和检查命令。
- 67项针对性CPU检查通过；新增相应回归先失败再修正。实际面板参数/profile/标定证据启动校验通过，在CameraSource处拦截，禁止build_components，不打开设备。`git diff --check`通过；独立只读复核最后种子与模型已有地面去纹理修改无新增Important。

检查命令：

```text
outputs/field_yolopv2/venv/bin/python -m unittest car.test.test_track_surface_reference car.test.test_visual_corner_guard car.test.test_track_colors car.test.test_visual_feedback car.test.test_control_render_timing -q
outputs/field_yolopv2/venv/bin/python outputs/visual_feedback/20261010/startup-policy-fix/preflight.py
outputs/field_yolopv2/venv/bin/python outputs/visual_feedback/20261010/b6c-second-corner-fix/build_evidence.py
```

用户人工推车/摆正的参考不能提供电机动力响应；同输入输出改善不证明实际轮迹完整匹配或不压线。当前没有新自主实车结果。此前本轮只读宿主检查未发现驾驶进程，supervisor socket不可用；未替用户重启，不宣称现在面板已待机可启动。

下一负责人仍本车端聊天：继续分析以上接近段残余支持，用户另行安排下一次实车运行后，将新原图/IMU/编码器与此冻结候选比对入弯时机、实际旋转及轮迹。原始参考与本次派生产物都保留。
