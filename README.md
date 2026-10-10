# VehicleCloudCollaboration

当前车端自动驾驶只使用外圈 LCC（Lane Centering Control）。系统从车载相机提取黄色
道路边界和绿色岛区，在鸟瞰坐标中估计道路中心线，再将横向/航向误差映射为四轮 PWM。
DonkeyCar、云端变道控制和旧网页均不参与当前运行。

运行链路：

```text
命令行 -> run_onboard.py -> 感知/中心线 -> LCC -> 安全门 -> 车型适配层
                                                                    |-> Raspbot Pi 5
                                                                    |-> Rosmaster Jetson
```

## 实时监控面板

已安装开机监控服务的 Jetson 可直接访问 `http://<车辆IP>:8080/`。
没有运行 HTTP 服务时，可手动启动只读页面：

```bash
cd /path/to/VehicleCloudCollaboration
./run.sh
```

页面显示车辆、传感器、YOLOPv2、驾驶和云端状态；本入口只读取状态文件，不连接硬件。
旧 LCC 详细诊断页及网页启停接口已移除。配置和设备所有权见
[实时面板说明](docs/vehicle-dashboard.md)。

## 直接运行

Jetson 使用协调监控占用的实验入口，默认禁用电机：

```bash
./run_vehicle_experiment.sh --max-runtime-seconds 30
```

Raspbot 可按现场安排直接启动闭环：

```bash
/home/pi/miniconda3/envs/car/bin/python car/autodrive/run_onboard.py \
  --config car/autodrive/config/onboard_runtime.yaml \
  --vehicle-profile raspbot_pi5 \
  --enable-motors \
  --confirm-motor-motion I_UNDERSTAND_MOTORS_WILL_MOVE \
  --max-runtime-seconds 60
```

去掉 `--enable-motors` 和确认参数即为电机干运行。`Ctrl+C`、运行时间到期、相机断流、
道路丢失或误差超限都会触发停车。

启用归档的命令行运行会在配置的输出目录下 `runs/` 新建独立实验目录，自动保存
原始、标注、鸟瞰三路视频、逐帧 CSV、最终状态以及当次配置/标定快照。以后定位问题使用
该轮的 `raw.mp4` 离线干运行，确认同一故障帧已经通过后再上车。

## 两辆实车

- `raspbot_pi5`：保留现有 Raspberry Pi 5 / Raspbot 实车标定。
- `rosmaster_jetson`：使用 `/dev/myserial` 的 Rosmaster 后端；当前只开放相机 dry-run，
  电机、云台和透视参数完成现场标定前由 profile 门禁阻止。

详细适配架构和标定流程见 [car/control/README.md](car/control/README.md)。

## Raspbot 当前实车标定

- 相机水平舵机 S1 正前方角度：`25°`
- 电机编号：`0=左前、1=左后、2=右前、3=右后`
- 直行基准 PWM：`16/16/20/20`
- 普通最大正向右弧 PWM：`26/26/10/10`
- 饱和急弯右转 PWM：`30/30/0/0`（仅外侧轮前进，内侧轮停转；任何轮都不反转）
- 单边界恢复：最多 `8s`，弯顶后保持受限原转向直到双边界连续稳定；黄线等硬安全条件仍立即停车
- 运行配置：`car/autodrive/config/onboard_runtime.yaml`
- 透视标定：`car/autodrive/config/onboard_calibration.yaml`

相机位置、角度、分辨率或画面旋转发生变化后，必须重新采集图片并完成透视标定。

## 目录

- `car/autodrive/perception/`：外圈边界、透视映射和可视化
- `car/autodrive/control/`：LCC、PWM 映射和安全门
- `car/autodrive/runtime/`：相机到四轮输出的实时闭环
- `car/autodrive/camera/`：云台和图像方向处理
- `car/autodrive/web/`：只读实时综合面板与遥测发布器
- `car/autodrive/tools/`：采集、标定、自检和架空轮测试
- `car/control/vehicle_control/`：公共接口、车型工厂和双车型 profile
- `car/control/vehicle_control/platforms/`：Raspbot 与 Rosmaster 后端
- `car/control/utils/Raspbot_Lib.py`：I2C 底层接口
- `car/test/`：当前 LCC、相机和硬件映射测试

详细操作见 [car/autodrive/README.md](car/autodrive/README.md)。

## 千问云端语义接口

云端语义客户端默认直接对接百炼 `qwen3.8-max`，支持 API Key、图像输入、结构化场景/风险/候选建议和调用记录。
配置与独立调用见 [car/cloud_client/README.md](car/cloud_client/README.md)。本地预览不联网，也不操作硬件：

```bash
python -m car.cloud_client --dry-run --image car/test/test_image.jpg
```

此客户端尚未接入上面的 LCC 实时运行链；真实 Key、线上时延和车端异步仲裁仍需后续验证。

## 车端软件验证

以下命令不驱动车轮：

```bash
/home/pi/miniconda3/envs/car/bin/python -m unittest \
  car.test.test_lane_centering \
  car.test.test_camera_gimbal \
  car.test.test_camera_transform \
  car.test.test_hardware_mapping \
  car.test.test_vehicle_dashboard
```

本项目仅供受控场地研究使用。真车运行时必须有人在车辆旁准备急停或切断电机电源。
