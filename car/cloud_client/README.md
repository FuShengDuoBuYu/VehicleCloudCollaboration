# cloud_client

`car/cloud_client/` 是车端访问云端服务的客户端模块。当前闭环使用 `CloudClient` 调用 OpenAI-compatible `/v1/chat/completions` 接口。

## 主要文件

- `mock_client.py`：历史文件名，内部已经改为真实云端 API 客户端
- `__init__.py`：导出 `CloudClient`、`CloudDecision`、`DEFAULT_CLOUD_API_BASE_URL`、`DEFAULT_CLOUD_MODEL`

## 环境变量

根目录 `.env`：

```text
CAR_CLOUD_API_BASE_URL="https://your-ngrok-or-cloud-base-url"
CAR_CLOUD_MODEL="qwen3.5:9b"
CAR_CLOUD_TIMEOUT=30
CAR_CLOUD_NUM_CTX=93696
CAR_CLOUD_MAX_COMPLETION_TOKENS=256
CAR_CLOUD_TEMPERATURE=0
CAR_CLOUD_THINK=false
CAR_CLOUD_IMAGE_LIMIT_MB=20
```

`CAR_CLOUD_API_BASE_URL` 是 base URL，客户端会自动拼接 `/v1/chat/completions`。也可以直接传完整 `/v1/chat/completions` URL。

## 决策格式

`CloudClient.request_decision(image_path, detection_result)` 返回 `CloudDecision`：

- `command`：云端命令，取值为 `left`、`right`、`straight`、`stop`
- `action`：车端动作，取值为 `lane-left`、`lane-right`、`forward`、`stop`
- `reason`：决策说明
- `latency_ms`：请求耗时
- `raw_response`：原始 OpenAI-compatible 响应 JSON

命令到车端动作的映射：

```text
left     -> lane-left
right    -> lane-right
straight -> forward
stop     -> stop
```

## Python 调用

```python
from cloud_client import CloudClient

client = CloudClient()
decision = client.request_decision(
    image_path="captured_frames/latest_frame.jpg",
    detection_result={"score": 0.85, "threshold": 0.5},
)

print(decision.command, decision.action)
```

测试代码也可以直接使用：

- `build_payload(image_path, detection_result)`：构造 OpenAI-compatible JSON 请求体
- `parse_response(body, latency_ms=0.0)`：解析云端响应体为 `CloudDecision`

## 闭环参数

`../run_closed_loop.py` 相关参数：

```bash
python run_closed_loop.py --cloud-mode cloud
python run_closed_loop.py --cloud-mode none
python run_closed_loop.py --cloud-url https://your-ngrok-or-cloud-base-url
python run_closed_loop.py --cloud-model qwen3.5:9b
python run_closed_loop.py --cloud-timeout 30
```
