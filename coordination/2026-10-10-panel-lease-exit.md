# 2026-10-10 网页控制租约过期退出

用户再次反馈自动退出。最新session `97d8e6abc3b6408498676f4cf28e71c7`，run `20261010_045600_816379_hardware`。`session.json.reason` 和 `control/stop.json.reason` 均为 `panel control lease expired`；stop requested_monotonic=3379.828245689，最后lease.expires_monotonic=3379.795040083。总时限0、exit_code0、stopped_verified=true，runtime无清理错误。不是恢复120秒限制或推理异常退出；runtime统一SIGINT文案“interrupted by operator”不能覆盖supervisor的准确原因。

源码确认两个永久失续约入口：`dashboard.js`任何 `/api/state`失败调用disconnected并清ownedSession；任何heartbeat请求失败也清ownedSession。页面恢复时不重新认领当前会话（正确避免自动接管），所以单次暂时请求失败会使本轮永久停止续约。现有原始记录只有最后lease，没有HTTP失败历史，不能确定此次实际起因是网络/监控错误还是刷新、关闭、切后台/锁屏。

## 修复边界

只改 `car/autodrive/web/dashboard.js`：监控GET失败不取消独立驾驶heartbeat；网络/503等暂时heartbeat失败重试，403/409明确拒绝则取消；心跳请求1000ms超时、原700ms周期且单请求在途，避免等待3秒才重试；请求捕获session ID防止迟到回调干扰新会话；手动End和已结束状态仍取消续约。面板显示真实退出原因。未改变车端原3秒租约、MotionGuard截止/取消锁存、PWM、视觉判定或记录。

保留真正失联3秒结束；刷新/关闭页面仍不自动接管或自动恢复已取消驾驶。不能承诺浏览器被系统暂停或持续网络中断时无限驾驶。总运行时长仍无限，靠用户结束；独立控制失联保护继续存在。

## 小范围验证

- `car/test/dashboard_heartbeat_checks.js` 在隔离V8中执行真实dashboard.js，并模拟DOM/fetch/timers，不访问浏览器、HTTP或硬件。before首先失败“monitor GET failure preserves control heartbeat ownership”；after共11个断言通过，涵盖监控失败、heartbeat网络/503恢复、1000ms超时、拒绝结束、并发去重、旧请求/End竞态、已结束状态及准确原因显示。
- `python3 -m unittest car.test.test_drive_sessions -q`：5项通过1.235s，仅合成Python子进程。`python3 -m unittest car.test.test_visual_feedback.VisualRuntimeTests.test_expired_browser_lease_inhibits_future_motion -q`：1项通过0.057s，仅虚拟PWM；过期仍阻止之后运动。`git diff --check`通过。
- latest session只读再确认finished/running=false/停止确认；网页静态资源动态从文件读取且Cache-Control:no-store，不需要服务重启。下一次用户先刷新页面再点击开始，避免旧JS继续运行；助手无Start/运动请求。

本run345控制行，278stop/66forward/1turn-left；最后10行中sample339为current-YOLO-cruise，结束前其余多数是视觉恢复门禁或过期。颜色enabled=true已实际加载，不能把自动退出解释成新颜色代码崩溃，也不能把一次forward当成持续行驶验收。运行内卡顿另有应用时效/恢复门禁，当前只修退出链路。

原始证据不改。`outputs/visual_feedback/20261010/panel-lease-fix/`保存before/after、增量patch、输入源码hash、失败/通过证据与状态。HEAD仍 `fce828b89b084eda293e83b99c8861f522858d8e`，大量dirty保留，无安装/服务重启/提交。下一轮实际短暂网络失败能否恢复仍待实车网页观察，不伪装为完成验证。
