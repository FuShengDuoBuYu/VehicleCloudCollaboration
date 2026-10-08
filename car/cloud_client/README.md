# 云端观测客户端

默认 `qwen3.8-omni-flash-realtime`，每事件新建独立WebSocket会话，上传一张最新JPEG，仅接收文本。默认合同为用户选择的 **road-observation-fast-v1**。模型、合同、传输分别配置，保留HTTP千问和其他OpenAI-compatible厂商入口。

本模块不导入摄像头、底盘或电机，不接管 `car/autodrive/runtime/onboard.py`。云端建议只能供本地观察/仲裁；没有自动PWM、停车、恢复或选路映射。`observe`不能解释为恢复驾驶。代码本轮未部署到车端，未建设HTTP网关。

**2秒仅为速度参考，超过2秒的合法结果照常返回。** `CAR_CLOUD_TIMEOUT`是独立可调的网络请求超时，默认30秒；Realtime使用该预算，包含图片准备、建连、会话配置、上传及完整响应/校验。不自动重试或串行切换HTTP/Max。

## 本机配置

仅合并以下项到仓库根目录 `.env`，保留现有Key及其他车端配置，不覆盖整个文件；`.env`不入Git。现代客户端不会自动读取 `.env_example`。

```dotenv
CAR_CLOUD_PROVIDER=qwen-realtime
CAR_CLOUD_MODEL=qwen3.8-omni-flash-realtime
CAR_CLOUD_CONTRACT=road-observation-fast-v1
CAR_CLOUD_WORKSPACE_ID=你的北京业务空间ID
CAR_CLOUD_REALTIME_WS_URL=
CAR_CLOUD_API_KEY=
CAR_CLOUD_TIMEOUT=30
CAR_CLOUD_IMAGE_LIMIT_MB=6
```

本地已保存的Key继续使用；也可设置 `DASHSCOPE_API_KEY`。空间ID不是Key。默认从ID构建北京业务空间WSS地址；也可配置完整 `CAR_CLOUD_REALTIME_WS_URL`，仅允许model查询参数，无URL凭据。地址与Key地域应匹配。HTTP配置 `CAR_CLOUD_API_BASE_URL` 不决定Realtime地址。

环境变量优先于 `.env`，显式构造参数优先于两者。`--env-file`/`CAR_CLOUD_ENV_FILE`可指定另一份配置。若历史进程环境仍设置Max或HTTP provider，应同步更新/清除覆盖项。

允许JPEG/PNG/WebP，Realtime只接受一张静态图；本地纠正EXIF方向、缩小最长边至640、JPEG quality=95，不裁剪、不增加letterbox。保留原图与派生hash/尺寸/变换；Base64不超过256KiB，超出则要求提供更小图。该Pillow编码配置与此前OpenCV先导有差异，不继承先导时延/质量结果。

每轮提交200ms的16kHz单声道合成静音PCM，随后图像、commit及response.create；静音是图像缓冲载体，不是实际车载音频，usage中音频Token应计费。HTTP的max_tokens/reasoning_effort/response_format不应用于Realtime；文本在本地校验，不冒称已向Realtime发送这些HTTP参数。

## 手工预览与调用

从仓库根目录执行，无Key也可预览，不联网：

```bash
python -m car.cloud_client --dry-run --image car/test/test_image.jpg
```

真正调用会产生API费用，需有效Key，输出路径必须是新文件：

```bash
python -m car.cloud_client --image car/test/test_image.jpg --output outputs/cloud/realtime-first.json
```

可加 `--timeout 60` 延长网络等待，或 `--context context.json` 带本地数据。上下文例如：

```json
{"run_id":"run-001","event_id":"e1","frame_id":"f1","candidate_version":"v1","capture_monotonic":123.45,"vehicle_stopped":true}
```

身份/时间由本地绑定，不要求模型复述。不同主机单调时钟不可直接相减，当前不会依据采集时间产生2秒截止。预览只显示公开配置和图片来源/hash，不输出Key、Bearer头或Base64。

## 结果合同与记录

```json
{"features":["straight_arrow","right_arrow"],"uncertain":false,"advice":"observe","reason":"直行或右转"}
```

features仅允许right_arrow/left_arrow/straight_arrow/crosswalk/parking_sign/boundary_line/obstacle/occlusion/blur/horn_sign且去重；uncertain严格bool；advice仅stop/wait/observe；reason非空且最多10个Unicode字符。拒绝额外字段、Markdown、非法类型、重复JSON字段和非有限数。空特征或uncertain=false不证明道路畅通。这份合同没有OCR原文、适用候选道路或可执行路径，丰富任务语义仍待独立验证。

成功结果保存event/frame/候选版本上下文、request UUID、实际输入hash、合同/提示版本hash、provider/session/response ID、原始事件、完整文本、usage和时延。服务端不报告model时response_model采用请求别名，不冒称固定快照。失败CLI也保留记录文件和可得原始事件；异常只含安全概述，意外回显Key在保存时脱敏。

Realtime只接受正常completed、单一assistant文本项；校验session ID、response ID、item ID及索引，并核对delta拼接、text.done与最终output文本。连接关闭、截断、服务端错误或不一致均失败。完整时间包含输入准备/冷会话/响应/校验，关闭耗时另记；first_content仅为首段文本，不代表完成。它仍不是车端采集到当前状态仲裁的全链路测量。

## 异步入口

同步 `CloudClient.request_scene`用于手工分析，控制快环使用独立worker：

```python
from car.cloud_client import CloudClient, ImageFrame, LatestSceneWorker

worker = LatestSceneWorker(CloudClient(), evidence_dir="outputs/cloud/run-001")
# jpeg_bytes是当前帧的不可变编码bytes，不共享会被覆盖的frame.jpg路径。
job_id = worker.submit(ImageFrame(jpeg_bytes, "f1.jpg"),
                       {"event_id":"e1", "frame_id":"f1", "candidate_version":"v1"})
# 随控制循环非阻塞检查，当前状态应与提交时身份对应。
outcome = worker.poll(event_id="e1", frame_id="f1", candidate_version="v1")
if outcome is not None and outcome.status == "completed":
    observation = outcome.result.scene  # 供本地复核，没有驱动车辆。
# 将worker.drain_records()写入run日志，保留未服务/合并/弃用事件。
worker.close()  # 不等待在途网络；收尾需要等退出时用close(wait=True)。
```

submit只提交内存快照，不编码、不联网、不等响应。每worker独占一个客户端：最多一个在途和一个最新待处理，新的待处理替换旧的；已被新事件替代的结果不会从poll返回。poll还能校验当前event/frame/候选版本，拒绝状态已改变的旧结果；这种丢弃依据身份，不依据2秒。

evidence_dir为每个实际尝试保存新JSON，即使结果被新事件替代也保存响应/失败；delivery/合并计数另通过drain_records读取。内存通知最多256条，溢出有计数；不开evidence_dir时需调用者自行保存，不能声称所有原始尝试已自动落盘。日志保存失败时不返回可用结果。close不取消已发出的请求；旧请求按配置超时结束，最新待处理会等待它。不要共享客户端或在旧worker收尾期间把它交给另一线程。

## HTTP与合同扩展

显式使用HTTP Omni并保留同一观测合同：

```dotenv
CAR_CLOUD_PROVIDER=qwen
CAR_CLOUD_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CAR_CLOUD_MODEL=qwen3.8-omni-flash
CAR_CLOUD_CONTRACT=road-observation-fast-v1
CAR_CLOUD_RESPONSE_FORMAT=json_object
CAR_CLOUD_REASONING_EFFORT=none
```

通用厂商选择openai-compatible，配置自己的HTTPS地址、model、Key和合同；不发送千问专有控制字段。provider扩展见providers.py，合同扩展见contracts.py。旧road-scene-v1可显式选择记录丰富分析，不能据其resume/route候选直接恢复车辆。历史左右动作客户端仅通过LegacyCloudClient导入，未改其代码。

## 开发验证

运行时新增websocket-client，Pillow已有基础依赖，不需要OpenAI/DashScope SDK。回环服务器是开发依赖：

```bash
python -m pip install -r car/test/requirements-cloud.txt
python -m unittest discover -s car/test -p "test_cloud_scene*.py"
```

测试只用虚构Key、本机HTTP/WebSocket和协议反例，没有调用真实阿里或硬件。历史16份Realtime事件仅离线重解析，没有重写。新代码真实账号/车端相机/网络效果需现场单独验证。

官方协议（2026-10-08核对）：[Realtime指南](https://help.aliyun.com/zh/model-studio/realtime)、[选定模型](https://help.aliyun.com/zh/model-studio/qwen3-8-omni-flash-realtime)。
