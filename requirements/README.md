# 依赖与实际环境

`runtime.txt` 是 Python 3.8 兼容的通用依赖范围，不是完整的设备复建锁。Jetson 的 PyTorch/CUDA/cuDNN 与主机系统匹配，不能用通用 `torch>=2.1` 自动升级替代。

实际环境位于 `outputs/field_yolopv2/venv`；本轮只读取 metadata，不安装或更改它。选择性版本快照见 `jetson-observed.json`，缺少安装元数据的系统包不推断版本；torch安装元数据为2.0.0+nv23.5。驱动、镜像和权重依据见原部署报告。

- 车端：已有验证环境 + runtime依赖；GPU轮包和系统库显式核对。
- 离线研究：`research.txt`，与车端环境隔离；当前未重新安装或验证全部研究依赖组合。
- 工程测试：现有runtime与CPU torch；`test-websocket.txt` 仅供可选本机WebSocket假服务。

安装或重建环境需要独立部署任务，本轮没有进行。原统一清单已[原文保存](../docs/history/requirements-pre-refactor.txt)，用于解释旧环境要求。版本范围不能替代实际运行的metadata快照，快照也不包含认证值或镜像凭据。
