# 当前近路接近、短转与可见边界余量实施计划

> 使用 executing-plans 内联实施及定向TDD。用户已批准上一报告的方案；保存任务事实到coordination，保留现有dirty，用户控制实车Start/End，无安装/服务重启/提交或助手运动。用户要求少做详尽测试，采用有关行为检查及冻结数据短回放。

**目标：** 弯道还远时不因远出口提前旋转，转后持续短步确认，图像路径保留可见道路/黄线余量。

**依据：** `coordination/2026-10-10-c92a-corner-feedback.md`；原始run `20261010_054444_110581_hardware`。用户已经确认并批准修改；不是新增固定行驶距离/90°目标。

**设计：** visual profile启用 `visual_corner_guard`；旧模式不变。selector在前边界未达到配置近处ratio时使用当前近直路目标，仍记录真实右出口；runtime只在前边界近、出口确认和当前可行路径/IMU有效时开始小pivot。进入转向阶段后连续新观测确认近直路、前横向边界消失才恢复巡航；缺支持立即停。新增纯CPU道路距离检查，路径近段周围按图像宽度比例保留余量，颜色matched源保持。缺失road不填、缺yellow数据失败关闭；无法保证相机看不到的轮下区域，需后续真实轮迹验证。

**全局约束：** input384/full road-lane-object/FP16不变；model/control/application时效.6s，恢复2新结果；巡航更新+.25s/源捕获+.6s，短步.15s/5°/settle.25s；相机.25s/IMU.2s/操作取消/黄线禁越/原始road上限/完整记录保留。

- [x] 备份相关文件、添加并观察三个关键行为的失败测试（远弯接近、近弯转后不误巡航、路径余量不足）。
- [x] 在semantic_outer_route/visual_clearance、matched黄线字段、runtime及profile中实现；live与录制回放共用派生route选项。
- [x] 定向CPU检查、实际panel参数入口在相机构造前拦截、旧直道与本轮入弯/末段短窗口配对回放；按新配置重算时效、不预测新轨迹。
- [x] 独立只读审查、关闭重要问题、保存before/after/patch/hash/config/原始关联与实际证据，更新索引并交用户下一次Start。

实施结果见 `coordination/2026-10-10-corner-approach-feedback.md`。107项定向检查通过；175行冻结回放在sample190由pivot变短前进。直道窗口多6停，末段rawroad缺失仍停；实际过弯与车身不压线未验证。

**审查关注：** 停车后旧帧不能凑出口计数；临时边界/道路漏检不清除转向记忆；unknown/缺颜色不能授权；visible余量不宣称轮下安全；配置校验与实际主入口一致。
