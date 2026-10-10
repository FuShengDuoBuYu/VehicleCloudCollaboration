# 2026-10-10 5406fc4d运行中的弯道停车只读观察

用户反馈又不走。session `5406fc4d3c394761b186b992d4cde82d`，run `20261010_050650_512765_hardware`，检查时state=running、running=true、termination为空，YOLO error为空；与上一轮网页租约结束不同。运行中仅轻量读取状态、CSV尾部和当前JPEG，以及一份已写入NPZ核对近处支持；无开发、离线回放、硬件指令或服务改动。

约sample1300时读取最后200行：116行estimate_reason=YOLOPv2 missing near-centre road support，73行=current-semantic corner target has no contained path，8行=current-semantic front-corner path，3行=YOLOPv2 stale result。最后一次非stop为sample573、timestamp_s81.9846、current-YOLO-cruise，当时front_observed=true、front_ratio0.5416667、right_exit_observed=false。运行持续写记录，不能称程序卡死。约sample1189时末30行20个visual-current-evidence-veto、10个最终数据过期；这两层原因分别保存，不能用泛化veto代替感知原因。

读取当前latest.jpg能看到右转箭头与前方路缘，绿色通路在近处不完整。后续sample1728、seq1728仍主动stop，estimate_reason=current-semantic corner target has no contained path；匹配归档NPZ的原始near road均值0.66873065、raw lane均值0，在0.60支持门槛之上，应用年龄0.223468665s、ego_yellow_ratio0。说明这一帧主因是未找到目标通路，不是过期或近处黄线；其他帧又因近处road支持不足失败，需停止采集后配对回放区分，不能只调低门槛强行驱动。

当前JPEG与统计是运行中不同时间的轻量观察，不是完整同步轨迹或最终run结论。用户先结束保存本轮，下一步再读取原始弯道画面/掩码核对近处通路、弯道目标与应用门禁。本轮没有控制修复，也没有宣称实际过弯或小车已停稳。
