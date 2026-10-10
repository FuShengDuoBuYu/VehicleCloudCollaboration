# 现场手动点动

2026-10-08，用户接管现场电机发令。此入口复用已有 wheel check，不运行 YOLO、修改运动标定状态或修改底盘配置。仅在四轮架空、固定、有人可急停时运行。

```bash
cd /home/jetson/VehicleCloudCollaboration
bash run_wheel_check.sh --confirm-wheels-lifted WHEELS_ARE_LIFTED --all-wheels --pwm 30 --duration 1
```

用户已确认落地30 PWM/1秒能前进并停住，随后明确要求以后按30 PWM测试与运行。因此默认测试PWM和当前Jetson YOLO配置的四轮直行基准均改为30，不再安排20/24/28逐档起步测试。上面的命令仍仅用于架空轮测，不能把确认文字当成地面测试条件。持续时间继续显式指定；未指定时工具原默认0.4秒不变。

| 参数 | 范围 / 含义 |
|---|---|
| `--pwm` | 默认30，允许整数1–30；按本次用户要求使用30 |
| `--duration` | 0.1–1.0 秒；目标非零输出时间，受系统调度影响 |
| `--all-wheels` | 四轮同幅度 |
| `--wheel 0` | 只动 M1 左前 |
| `--wheel 1` | 只动 M2 左后 |
| `--wheel 2` | 只动 M3 右前 |
| `--wheel 3` | 只动 M4 右后 |
| `--direction forward` | 默认前进；后退改为 `reverse` |

逐轮模式用 `--wheel N` **替换** `--all-wheels`，不要同时传入。例如：

```bash
bash run_wheel_check.sh --confirm-wheels-lifted WHEELS_ARE_LIFTED --wheel 0 --pwm 30 --duration 1 --direction forward
```

程序到时或 Ctrl+C 时执行已有工具的 stop/close；这不是硬件独立急停，进程卡死、断电/串口故障时不能保证停车。异常持续转动应现场急停。放回地面后不要继续使用架空确认参数；地面起步/停止标定是后续独立步骤。

入口临时暂停当前已运行的 `vehicle-serial-monitor.service`，等待电机子程序退出后恢复；不改服务配置。若提示 sudo 需要密码，在车端终端执行 `sudo -v` 后重试。不会停止其它进程来强抢串口；如果仍报串口占用，应先退出其它电机控制程序。

每次独立目录为 `outputs/manual_wheel_check/<UTC时间>-<随机后缀>/`，含 `command.txt`、`console.log`、`exit-code.txt`。这是命令与终端日志，不包含自动编码器采样，也不能代替观察到的真实轮子转动。恢复失败会输出手动恢复命令 `sudo systemctl start vehicle-serial-monitor.service` 并返回失败。

后续记录30 PWM下的时长、是否偏转、轮子和停止现象。转弯仍需左右差速，最大输出30、停车0；内侧低于30的运行能力和转弯半径尚未实测。`motion_calibrated`仍为false，不因一次直行点动就开放自动绕圈。

验证：`bash -n run_wheel_check.sh`；`bash run_wheel_check.sh --help`；`python3 -B -m unittest car.test.test_wheel_check_launcher`（7 项通过，串口/电机/服务均为模拟）。交付入口时未再次执行真实点动。
