# cloud_client

`car/cloud_client/` 是车端访问云端服务的客户端模块。当前闭环使用 `mock_client.py` 调用通用 LLM mock `/chat` 接口。

## 主要文件

- `mock_client.py`：云端 mock 客户端
- `__init__.py`：导出 `CloudMockClient`、`CloudDecision`、`DEFAULT_MOCK_CHAT_URL`

## 默认服务地址

```text
http://lz1pe24291437.vicp.fun/gzt_code_server/llm_interface/chat
```

## 决策格式

`CloudMockClient.request_decision(image_path, detection_result)` 返回 `CloudDecision`：

- `command`：云端命令，当前 mock 固定为 `left`
- `action`：车端动作，当前 mock 固定为 `lane-left`
- `reason`：决策说明
- `latency_ms`：请求耗时
- `raw_response`：原始响应 JSON

命令到车端动作的映射：

```text
left     -> lane-left
right    -> lane-right
straight -> forward
stop     -> stop
```

## Python 调用

```python
from cloud_client import CloudMockClient

client = CloudMockClient(force_left=True)
decision = client.request_decision(
    image_path="captured_frames/latest_frame.jpg",
    detection_result={"score": 0.85, "threshold": 0.5},
)

print(decision.command, decision.action)
```

测试代码也可以直接使用：

- `build_payload(image_path, detection_result)`：构造发给 `/chat` 的 JSON 请求体
- `parse_response(body, latency_ms=0.0)`：解析 mock 服务返回体为 `CloudDecision`

## 闭环参数

`../run_closed_loop.py` 相关参数：

```bash
python run_closed_loop.py --cloud-mode mock
python run_closed_loop.py --cloud-mode none
python run_closed_loop.py --cloud-mock-url http://host/path/chat
python run_closed_loop.py --cloud-timeout 30
```
