# Jetson 调通任务状态

- 日期：2026-10-08（Asia/Shanghai）；状态：目标active，设计待审阅，运动现场条件待回答。
- 原始目标：当前小车软硬件调通、参考仓库原有小车配置、可直接使用，底层车控与驱动完整封装供上层使用。不能缩减成仅mock兼容或只读盘点。
- HEAD：`33a5a712b97af15feadc11f4751e60edaa05dc08`；已有26个跟踪文件修改及未跟踪适配保持原样，本轮新增验证/设计文档。
- 设计：`/home/jetson/VehicleCloudCollaboration/docs/superpowers/specs/2026-10-08-jetson-vehicle-bringup-design.md`。

## 当前实际验证

1. 当前Python执行五组unittest：103 tests，1.786s，OK，退出码0。命令：`python3 -B -m unittest car.test.test_lane_centering car.test.test_camera_transform car.test.test_camera_gimbal car.test.test_hardware_mapping car.test.test_lcc_web`。Fake执行器/合成输入验证，不是实车实验。
2. 当前车型自检未打开相机时通过基础项；沙箱/dev不可见产生设备告警，不代表真实设备缺失。
3. 在真实宿主设备可见环境运行 `python3 -B car/autodrive/tools/pi_self_check.py --vehicle-profile rosmaster_jetson --camera --json-output /tmp/jetson-bringup-camera-selfcheck-20261008.json`：RGB单帧640×480通过，底盘节点可读写，退出码0；strict_ok=false，透视/运动/云台未标定。
4. 已审閱Rosmaster_Lib 3.3.1源码：构造发送舵机力矩、set_motor吞错误，telemetry getter只返回缓存。尚未构造/连接底盘、启动雷达或深度驱动、发运动或云台命令。
5. 已执行限时5秒的既有LCC入口：`python3 -B car/autodrive/run_onboard.py --vehicle-profile rosmaster_jetson --max-runtime-seconds 5 --output-dir outputs/onboard_runtime/rosmaster_jetson_bringup`。退出码0，run `20261008_034402_061387_dryrun`，实际相机100帧，raw/annotated/birdeye视频、CSV和配置/status齐全；队列无丢弃/错误，结束时虚拟四轮命令归零。没有--enable-motors、没有构造底盘，profile云台/YOLO关闭、透视为空。相机当前为桌面/绿色物品，出现虚拟转向，不能视为安全道路导航或实车运行证据。
6. 标准库mock探针复现三项底层缺陷：无效云台组合先写有效轴；错误motor_order先连接再验证；motion_calibrated字符串"false"通过正常运动配置门禁。所有探针硬件连接数0，优先修复并建立回归验证。

没有安装依赖、修改服务、改业务代码或提交Git。当前尚不能交付为“完全调通”。

新采集目录：`/home/jetson/VehicleCloudCollaboration/outputs/onboard_runtime/rosmaster_jetson_bringup/runs/20261008_034402_061387_dryrun`。本轮没有改写历史Pi素材。命令、配置、run产物哈希与限制保存在 `coordination/2026-10-08-jetson-bringup-validation.json`；相机自检副本为 `coordination/2026-10-08-jetson-bringup-camera-selfcheck.json`。

## 下一依赖

- 审阅推荐设计：保留现有LCC/profile，加入严格的共享底盘会话/遥测/公开接口，传感器单独驱动并验证，最后实际标定和现场闭环验收。
- 现场状态回答到达后安排有界实车验证；未回答前不启动执行器。
- 设计确认后生成实现计划，按独立模块与真实验收推进，最终依据完整矩阵决定完成。
