# closed-loop data test

这个目录用于做不启动车辆硬件的闭环数据测试。默认测试图片是 `test_image.jpg`。

## 运行

```bash
cd /home/pi/Desktop/VehicleCloudCollaboration
python car/test/closed_loop_test.py
```

默认链路：

1. 读取 `car/test/test_image.jpg`
2. 使用 `car/longtail` 的 `LongTailClassifier` 复合判断器判断是否触发长尾
3. 使用离线 in-process mock 构造云端响应
4. 通过 `CloudMockClient` 构造 prompt 并解析云端决策
5. 校验返回 `command=left` 和 `action=lane-left`
6. 给 `VehicleController` 注入 `FakeChassis`
7. 执行 `lane-left` 并检查左变道动作序列
8. 输出每个阶段的时延、输入、输出和 JSON 报告

默认报告文件：

```text
car/test/closed_loop_report.json
```

## 常用参数

```bash
python car/test/closed_loop_test.py --image car/test/test_image.jpg
python car/test/closed_loop_test.py --config car/longtail/config.yaml
python car/test/closed_loop_test.py --report /tmp/closed_loop_report.json
python car/test/closed_loop_test.py --cloud-backend local-http
```

脚本会加载 `LongTailClassifier` 和 `config.yaml` 中配置的检测器。测试图片没有触发 `is_long_tail=True` 时，脚本会失败并在报告里保留每个检测器的输出。

`--cloud-backend local-http` 会在本机启动一个临时 HTTP mock 服务，用来额外检查 HTTP 发送和 JSON 序列化。默认 `inprocess` 不依赖网络、端口或外部服务。
