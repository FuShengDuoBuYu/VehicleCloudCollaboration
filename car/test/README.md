# closed-loop data test

这个目录用于做不启动车辆硬件的闭环数据测试。默认测试图片是 `test_image.jpg`，默认云端后端会调用 `.env` 中的公网 `CAR_CLOUD_API_BASE_URL`。

## 运行

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
python car/test/closed_loop_test.py
```

默认链路：

1. 读取 `car/test/test_image.jpg`
2. 使用 `car/longtail` 的 `LongTailClassifier` 复合判断器判断是否触发长尾
3. 调用公网 OpenAI-compatible 云端接口
4. 通过 `CloudClient` 构造 OpenAI-compatible 请求并解析云端决策
5. 校验云端返回合法 `command` 和车端 `action`
6. 给 `VehicleController` 注入 `FakeChassis`
7. 执行云端返回的动作并检查动作序列
8. 输出每个阶段和子阶段的时延、输入、输出和 JSON 报告

默认报告文件：

```text
car/test/closed_loop_report.json
```

## 常用参数

```bash
python car/test/closed_loop_test.py --image car/test/test_image.jpg
python car/test/closed_loop_test.py --report /tmp/closed_loop_report.json
python car/test/closed_loop_test.py --cloud-url https://your-ngrok-or-cloud-base-url
python car/test/closed_loop_test.py --cloud-backend inprocess
python car/test/closed_loop_test.py --cloud-backend local-http
```

脚本会加载 `LongTailClassifier`，检测器配置来自环境变量。测试图片没有触发 `is_long_tail=True` 时，脚本会失败并在报告里保留每个检测器的输出。

`--cloud-backend inprocess` 不依赖网络、端口或外部服务。`--cloud-backend local-http` 会在本机启动一个临时 OpenAI-compatible HTTP 服务，用来额外检查 HTTP 发送和 JSON 序列化。
