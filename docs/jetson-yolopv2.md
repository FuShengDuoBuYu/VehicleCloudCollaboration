# Jetson YOLOPv2 场地部署

2026-10-08，Jetson Orin NX，L4T R35.3.1 / Ubuntu 20.04 / Python 3.8。运行证据及限制见 `coordination/2026-10-08-jetson-yolopv2-field-report.md`。云端按用户决定暂缓。

## 启动

在 `/home/jetson/VehicleCloudCollaboration` 执行：

```bash
./run_vehicle_experiment.sh --max-runtime-seconds 30
```

这会启动真实 Astra RGB 相机和 CUDA FP16 模型，运行虚拟控制并归档视频、配置及CSV，30秒后停止。默认不打开底盘，不初始化云台。Ctrl+C 可退出。输出在 `outputs/onboard_runtime/rosmaster_jetson_yolopv2/runs/`。

入口会暂停占用相机的监控，退出后恢复。网页由开机HTTP服务提供，直接访问
`http://<车辆IP>:8080/`；服务尚未运行时可手动启动：

```bash
./run.sh
```

不要根据页面显示的虚拟 PWM 判断车轮已经运动。四原生轮序/正转方向已逐轮核对；真实自动驾驶仍需轮速与固定相机标定、模型场地验收以及本次现场试验安排；profile 的 `motion_calibrated` 和 `gimbal_calibrated` 仍为 false。

## 配置与行为

`car/control/vehicle_control/profiles/rosmaster_jetson_yolopv2.yaml` 合并到原有运行配置，显式覆盖旧树莓派 CPU/ONNX、黄色边界和弯道续行设置。当前候选352输入（保持比例后张量288×352），使用 `perception.mode=yolopv2`，完整道路、车道线和目标检测头；保持相机比例；模型道路减去车道线得到候选通道，LCC据此生成有界车轮提案。不会退回颜色算法。

无模型、异常/非有限输出、结果过期、异常全图道路、近车支持不足、模型近处目标框挡住前方或拟合中心线离开语义通道时停车。恢复需不同模型观测，缓存重复消费不计数。检测框只是图像空间初筛，未标定真实安全距离，也不能发现模型未识别的所有障碍。

新profile保留旧车型配置；`run.sh`仅启动只读综合面板，驾驶使用上面的实验入口。现有深度、雷达、编码器、IMU未并入这条YOLO感知链路；没有SLAM/定位/全局导航。车端云建议仲裁已实现并通过模拟测试，真实云端与实车恢复尚未验收。

## 环境重建依据

专用环境 `outputs/field_yolopv2/venv` 通过 `--without-pip --system-site-packages` 创建，继承已有OpenCV/NumPy/PySerial；新PyTorch及附属Python包只安装在该环境。不要在该JetPack上直接安装通用requirements里的较新桌面PyTorch。

```bash
sudo apt-get install --no-install-recommends cuda-cudart-11-4 cuda-nvtx-11-4 cuda-nvrtc-11-4 libcublas-11-4 libcudnn8 libcufft-11-4 libcurand-11-4 libcusolver-11-4 libcusparse-11-4 libopenblas0-pthread
python3 -m venv --without-pip --system-site-packages outputs/field_yolopv2/venv
outputs/field_yolopv2/venv/bin/python -m pip install outputs/field_yolopv2/downloads/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```

本次准确包版本见 `outputs/field_yolopv2/20261008/gpu-packages.txt`、`venv-freeze.txt`，重建时应对照这些版本。没有安装ROS或TensorRT，也没有调整功耗模式/锁频、重启业务服务。

PyTorch：官方 [NVIDIA Jetson 安装指南](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) 对应 v511 / nv23.05，wheel SHA256 `39eeb9894ef8c7b84249ab917f212a91703f30255d591b956ab12cc10e836532`。

权重：作者 [YOLOPv2 V0.0.1 release](https://github.com/CAIC-AD/YOLOPv2/releases/tag/V0.0.1)，位于 `car/longtail/models/yolopv2.pt`，156380200字节，SHA256 `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c`。这是公开预训练权重，尚未为当前缩小场地训练。权重、venv和原始视频由Git忽略，普通Git拉取不包含它们。

## 2026-10-08 外圈调试增量

YOLOPv2 负责道路、车道线与目标感知；本地控制器根据模型掩码计算路径与轮速，安全门禁可以停车。原版 YOLOPv2 不直接输出 PWM。主模式画面标注 `YOLOPv2 -> control`，并另列 `DRY RUN (motors disabled)` 和最终映射后的虚拟 PWM。归档旧图不会被改写。

352 输入与原图前视参数 `top=.54/lookahead=.62/tight=.78/bottom=.94` 是依据两段静止录像选择的开发候选。它们修正旧 Pi 鸟瞰参数在相机原图上的不适配，但不是透视标定、车身避线或整圈验收。绿色区域及斑马线仍有误检，当前模型未做场地域适配。motion_calibrated 仍为 false。

`runtime_overrides.cloud_arbitration.enabled` 默认 false；云端尚未选定。启用后仅支持严格 `road-scene-v1`：异常/掩码变化先停车，后台请求绑定模型实际消费的原图，建议必须匹配当前事件并未过期；恢复候选还要通过不同新鲜本地观测及原有运动门禁。路线建议只记录并保持停车。0.4 掩码变化阈值未经场地标定。

运行中仓库更新到 fce828b，其接口新增 `road-observation-fast-v1` 默认契约。本车端仲裁不会把精简观察当作恢复许可：启用仲裁前必须使用 `CAR_CLOUD_CONTRACT=road-scene-v1`，不匹配会在构建底盘前报错。本轮没有选定供应商、填写或使用密钥、发起真实云请求，亦未验收新增实时 WebSocket 路径。现场云端配置仍等用户明确选择。

请求证据位于每个 run 的 `cloud/<event_id>/`：原图、请求上下文、响应或错误类型、脱敏模型/契约/计时信息和输入哈希。原始响应文本与凭证不写入这份车端归档。

阶段报告：`coordination/2026-10-08-outer-loop-cloud-report.md`。地面12、16、20 PWM各0.30秒起步停止脚本已按当次直道安排执行，用户均报告没有位移；起步尚未验证。不能用mock或串口成功声称实车运动已验收。

仪表盘监控启用后会占用相机与底盘UART。当前仓库新增的`run_vehicle_experiment.sh`可为原运行入口临时释放所需采集器并恢复；使用前仍须满足当前现场安排及运动门禁。不要同时打开第二个独立串口控制器。本次有界标定只暂停并恢复只读serial采集，租用过程与恢复结果已归档。
