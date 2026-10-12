# 历史研究代码

`closed_loop_test.py` 与 `closed_loop_report.json` 从 `car/test/` 原样归档，内容哈希见协调目录迁移表。旧脚本依赖已移除的 `vehicle_control.controller`，不是当前可执行测试；旧报告不证明当前千问语义或实车闭环已接通。

代码原始路径语义按迁移前解释。若需复现，须找到对应历史源码/配置，不补造当前实现来解释旧结果。当前兼容客户端仍由 `car/cloud_client/mock_client.py` 导出并受测试保护。

CLIP/YOLOv8/单帧长尾分类研究保留在 `car/longtail/`，人工数据原路径不变；当前车端YOLO推理已迁到 `car/autodrive/perception/yolopv2_torch.py`，历史导入为兼容门面。
