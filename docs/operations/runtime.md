# 运行入口与副作用

仓库统一入口是 `./vehicle`，无参数或 `--help` 只显示帮助。所有子命令帮助均在读取设备、改变服务、选择模型和建立输出目录之前返回。解释器由同一处选择：显式 `CAR_PYTHON` 优先，其次现有 `outputs/field_yolopv2/venv`，否则当前系统Python；缺依赖不会自动安装。

| 命令 | 作用 | 副作用 / 条件 |
|---|---|---|
| `./vehicle check --static` | 语法、链接、归档哈希和目录约束 | 不导入设备或模型 |
| `./vehicle check` | 模拟设备回归 | 临时文件；回环测试须显式选项 |
| `./vehicle catalog --output NEW_DIR` | 生成实验注册表 | 只读原数据，拒绝原始数据目录内输出 |
| `./vehicle dashboard` | HTTP面板 | 绑定端口、读取状态，向监督器发送显式会话意图 |
| `./vehicle panel` | 会话监督器 | 启动只建本机socket，后续Start按模式启动驾驶 |
| `./vehicle drive ...` | 当前视觉/云辅助驾驶链 | 默认电机禁用，但访问相机/模型，按profile可请求真实云，并暂停、恢复相机监控 |
| `./vehicle monitor perception` | 视觉观察 | 相机/GPU，不驱动车辆 |
| `./vehicle monitor serial` | 被动串口采集 | 占用串口，不发送控制帧 |
| `./vehicle monitor sensors` | 雷达/深度采集 | 启动现有Docker镜像及传感器驱动 |
| `./vehicle replay --run PATH --output NEW_DIR` | 离线回放 | 默认recorded配置/输入，无真实云或执行器 |
| `./vehicle trial SECONDS ...` | 有界实车试验后回放 | 1–120秒，必须明确确认运动；会操作车辆 |
| `./vehicle wheel-check ...` | 架空轮校准 | 必须传现有抬轮确认；会操作底盘，console保存到新目录 |

## 配置与确认

`drive`默认当前 `rosmaster_jetson_yolopv2_visual_feedback`，30秒、电机禁用；命令参数可显式覆盖配置/profile/时限。底层公共YAML仍保留Pi兼容，不能绕过统一入口后假定默认是Jetson。面板会话显式传总时限0，依靠会话租约与局部运动截止。

需要真实运动的drive/trial必须传 `--confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE`；drive还需要 `--enable-motors`，现场field-trial及既有本地门禁继续生效。trial只接受时长和确认，不允许覆盖为无界运行。wheel-check继续要求 `--confirm-wheels-lifted WHEELS_ARE_LIFTED`。这些参数是程序检查，不代替当次实验安排。

执行监控租用时，只恢复进入前active的服务。释放失败不启动子进程；恢复失败返回非零。SIGINT/SIGTERM最多转为一次子进程SIGINT，等待清理完成后再恢复监控。驾驶程序、会话监督器和运动看门狗原有停止责任保留。

## 兼容与部署

旧 `run.sh`、`run_jetson_yolopv2.sh`、`run_outer_loop.sh`、`run_wheel_check.sh`、`check.sh` 已退出活动目录，分别使用新命令。旧完整实现可从Git基线及任务前像哈希追溯。

暂留三个无业务逻辑转发：

- `run_vehicle_dashboard.sh`：已安装的四个监控/网页unit仍引用它。
- `run_autodrive_panel.sh`：已安装的监督器unit仍引用它。
- `run_vehicle_experiment.sh`：已加载的旧监督器仍可能按原路径启动子进程。

仓库 `deploy/*.service` 已改用vehicle入口；本轮未更改 `/etc/systemd/system` 或重启服务。下一次部署并确认旧进程已替换后，可移除这三个适配文件。不要在仍有旧进程/服务引用时直接删除。现有安装脚本遇到内容不同的unit会拒绝覆盖，部署需独立review差异；不是运行安装脚本就会自动升级已有unit。

## 原数据与环境

模型仍在 `car/longtail/models/yolopv2.pt`，现用venv仍在outputs，均未搬移。新运行在底盘初始化前保存源码ZIP和来源manifest，范围包含统一入口及生命周期代码；旧运行不回填。认证仍由本机管理，不写进文档或命令记录。

原记录与回放限制见[回放说明](../outer-loop-record-replay.md)，校准见[底盘说明](../../car/control/README.md)。相机dry-run和单元测试不能证明地面停车距离或实车避障。
