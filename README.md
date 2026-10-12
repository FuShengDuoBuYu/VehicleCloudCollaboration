# VehicleCloudCollaboration

毕业论文车端研究仓库：本地视觉感知与控制、异步云语义辅助恢复、车端执行约束和可追溯实验记录。当前平台为 **Jetson / Rosmaster**，面板驾驶使用 `rosmaster_jetson_yolopv2_visual_feedback`。Raspbot Pi 配置和旧实验保留为兼容/研究资料。

## 从这里开始

| 任务 | 入口 |
|---|---|
| 当前实现、模块边界和扩展位置 | [系统架构](docs/architecture/current.md) |
| 了解当前事实与已知问题 | [项目索引](PROJECT_INDEX.md) |
| 查看启动入口及其硬件/服务副作用 | [运行说明](docs/operations/runtime.md) |
| 开发与无硬件验证 | [开发指南](docs/operations/development.md) |
| 查找实验、版本与数据来源 | [实验注册与保全](experiments/README.md) |
| 阅读历史方案和撤回原因 | [历史索引](docs/history/README.md) |
| 云客户端与协议 | [云接口](car/cloud_client/README.md) |
| 交给其他 agent 绘图 | [架构内容说明](coordination/2026-10-12-current-architecture-drawing-brief.md) |

## 当前控制链

RGB 相机 → 异步 YOLOPv2（道路/车道/检测）→ 当前路面与路径证据 → 本地 LCC/视觉反馈 → 规则融合 → 执行门禁 → Rosmaster 底盘。

持续停车时，车端构造受限候选，通过 `road-recovery-v1` 异步请求外部多模态服务。回复只能作为语义建议；车端用最新观测重新核验，决定短步或保持停车。云不直接设置 PWM。

当前入口配置启用 `cloud_recovery`，关闭旧 `cloud_arbitration`。掩码时序突变触发分支、雷达/深度几何避障、SLAM、独立环境语义等不能视为当前完成集成的能力。源码实现、离线回放和正式实车结果分别表述。

## 统一入口

所有日常命令从 `./vehicle --help` 进入：`dashboard`、`panel`、`drive`、`monitor`、`replay`、`trial`、`wheel-check`、`check`、`catalog`。无参数只显示帮助。驾驶默认使用当前 visual_feedback profile，电机默认禁用。副作用与运动确认见运行说明。

项目规则只维护根 `AGENTS.md`；两项项目技能只维护 `.agents/skills/` 中的实体目录。技能中的 `agents/openai.yaml` 是显示元数据，不是第二份协作规则。

## 不操作车辆的检查

```bash
./vehicle check --static       # Python / shell 语法与文档本地链接
./vehicle check                # 模拟设备回归；不连接真实云或传感器
./vehicle check --torch        # 加入 CPU 张量输入检查，不加载权重推理
```

脚本优先使用现有 Jetson Python 环境；可设置 `CAR_PYTHON` 指定解释器。缺依赖会报错，不自动安装。HTTP/WebSocket 回环测试需要显式选项，见开发指南。

不要直接运行所有名称包含 test 的历史或现场脚本。驾驶、传感器采集和设备试验须按现场实验安排执行；入口副作用见运行说明。

## 目录职责

- `car/autodrive/`：当前感知、控制、运行编排、回放与面板。
- `car/control/vehicle_control/`：车型 profile、设备工厂和底盘后端。
- `car/cloud_client/`：车端云访问、合同与传输适配；不是云服务端。
- `car/longtail/`：历史分类研究、评估数据及旧 detector 导入兼容。
- `experiments/`：只读证据目录工具、数据集内容/标注关系与生成的注册表。
- `tools/vehicle_cli.py`、`tools/process_lifecycle.py`：统一命令路由、环境选择、设备占用和退出清理。
- `.agents/skills/`：项目技能唯一正文与发现位置。
- `tests/`、`car/test/`：工程回归与现有控制回归，统一由 `tools/check.py` 选择。
- `docs/architecture/`、`docs/operations/`、`docs/history/`：当前架构、操作、历史资料。
- `requirements/`：运行/研究/测试依赖边界及实测环境清单。
- `research/legacy/`：退出当前入口的历史代码和报告。
- `coordination/`：日期化审查、实验复盘和交接。
- `outputs/`：保留原位的实验与运行状态；同时含现用环境，不能按“输出”整体清理。

原视频、传感器、源码快照、模型和大部分运行产物被 Git 忽略。Git 并不备份它们。失败运行与成功运行同样保留，派生分析使用新目录。
