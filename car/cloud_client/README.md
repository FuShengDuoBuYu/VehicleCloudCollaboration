# 云端道路语义客户端

`CloudClient` 直接通过 HTTPS 调用阿里云百炼，默认模型为 `qwen3.8-max`。
支持单图/按时间排序的多图、Bearer API Key、JSON Schema 输出及本地校验。
此入口只读取图片文件，不导入相机、底盘或电机模块。

本模块尚未接入 `car/autodrive/runtime/onboard.py` 的触发器与安全仲裁。
模型返回的是道路语义和候选建议，不能直接用于 PWM、恢复行驶或变道。
这次实现完成后仍需真实 API 与车端异步接入验证，不能据离线测试声称实车效果。

## 配置千问

把下面的云端配置合并进仓库根目录本机 `.env`，保留其他现有配置。
不要覆盖整个 `.env`，不要把真实 Key 提交到 Git。现代客户端只读 `.env`，不会自动读 `.env_example`，也不会修改进程环境变量。

```dotenv
CAR_CLOUD_PROVIDER="qwen"
CAR_CLOUD_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
CAR_CLOUD_MODEL="qwen3.8-max"
CAR_CLOUD_API_KEY=""
CAR_CLOUD_TIMEOUT=30
CAR_CLOUD_MAX_COMPLETION_TOKENS=1024
CAR_CLOUD_REASONING_EFFORT="none"
CAR_CLOUD_RESPONSE_FORMAT="json_schema"
CAR_CLOUD_IMAGE_LIMIT_MB=6
```

`CAR_CLOUD_API_KEY` 留空是占位；真实调用前必须在本机填入有效 Key，或设置 `DASHSCOPE_API_KEY`。
缺失 Key 和常见占位文本会在联网前拒绝。Key 不作为命令行参数、不出现在预览和请求日志中。
进程环境变量优先于 `.env`，显式构造参数优先于两者；`CAR_CLOUD_ENV_FILE` 或 `--env-file` 可指定另一份本机配置。

上述旧北京域名仍受官方支持，可按账号改为业务空间专属地址：
`https://<真实WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。
不要原样使用占位 WorkspaceId。各地域 API Key 与地域地址必须匹配。

`CAR_CLOUD_REASONING_EFFORT` 可取 `none/low/medium/xhigh`；基础配置明确关闭思考，后续比较 `low` 的理解质量与时延。
图片总原始大小默认上限 6 MiB，最多 8 张；允许 JPEG/PNG/WebP。图片内容格式仍由服务端验证。
`CAR_CLOUD_MAX_COMPLETION_TOKENS` 转为官方请求中的 `max_tokens`，只发送一个长度参数。
旧 `CAR_CLOUD_THINK/NUM_CTX/TEMPERATURE/TOP_P/SEED` 不控制现代客户端，必须迁移到上述配置。

## 本地预览：无需 Key，不联网

从仓库根目录执行：

```bash
python -m car.cloud_client --dry-run --image car/test/test_image.jpg
```

预览只显示模型、地址、Schema、配置和图片 SHA-256/大小，不显示 Bearer Header 或 Base64 图片。
它证明本地能构造请求，不证明图片语义理解、账号权限或线上服务可用。

## 真实调用：先补 Key

```bash
python -m car.cloud_client --image car/test/test_image.jpg --output outputs/cloud/first-scene.json
```

上下文文件示例：

```json
{"run_id":"your-run-id","frame_timestamps":[0.0,0.5],"vehicle_stopped":true,"task":"沿指定通道到达任务点"}
```

```bash
python -m car.cloud_client --image frame-1.jpg --image frame-2.jpg --context context.json --reasoning-effort low --output outputs/cloud/multiframe-scene.json
```

真实请求必须提供新的 `--output` 文件。既有证据不会覆盖；失败也会保留状态、可用的请求身份/输入来源与错误信息。
不自动重试付费请求。输出存于 `.gitignore` 已忽略的 `outputs/`。

## Python 接口

在可以导入 `car/` 的环境中使用：

```python
from cloud_client import CloudClient

client = CloudClient()
result = client.request_scene("frame.jpg", {"run_id": "run-001", "vehicle_stopped": True})
print(result.scene["road_state"], result.scene["recommendation"])
print(result.usage, result.timings_ms)
```

仓库根目录也可使用 `from car.cloud_client import CloudClient`。
此调用是阻塞的；车端集成时应在独立工作线程执行，每个工作线程使用自己的客户端实例，避免阻塞控制环或共享 `last_request_*` 调试状态。
客户端拒绝非正常完成、拒答、重复 JSON 字段、错类型、额外字段、旧变道 JSON 和明显矛盾的恢复建议。
它不证明场景结论正确，也不替代结果时效、请求对应关系和本地传感器复核。

## 输出与证据

`result.scene` 的版本是 `road-scene-v1`：

| 字段 | 内容 |
|---|---|
| `scene_summary` | 场景概述 |
| `road_state` | clear/partial/blocked/unknown |
| `objects` | 类别、左/中/右位置、是否挡住走廊、可见依据 |
| `signs` | 文字、含义、是否适用于小车、依据；无法判断时 null |
| `risk_level` | low/medium/high/unknown |
| `recommendation` | stop/wait/request_observation/resume_candidate/route_candidate |
| `route_hint` | none/left/right/unknown，仅记录候选方向 |
| `uncertainties`、`reason` | 信息缺口与建议依据 |

本地结果 envelope 独立记录 UUID、请求开始/结束 UTC 时间、provider、请求与实际响应 model、Schema/提示词版本、原始响应、实际 usage、上下文及实际上传图片的 SHA-256。
`request_config` 保存实际接口地址、思考档位、输出 token 上限、输出模式、超时与图片限制，以及 Schema/系统提示词哈希；成功与联网后的失败记录均保留该快照。
时延记录 `payload_build/http/parse/total`；非流式入口没有首 token 时延，不虚构服务端各阶段耗时。
凭据若被服务端意外回显会被替换为 `[REDACTED]`；HTTP 错误正文不会写入异常。损坏的 HTTP 响应会转为安全失败，JSON 非有限数值（包括指数溢出）在解析阶段拒绝。
错误、超时或拒绝的结果不得用于放行车辆。米级距离以有效的本地深度、雷达或标定观测为依据。

## 后续模型扩展

provider、model、URL、Key 与道路语义结构分离。`providers.py` 提供扩展点。
支持 JSON Schema 的 OpenAI-compatible 视觉服务可配置：

```dotenv
CAR_CLOUD_PROVIDER="openai-compatible"
CAR_CLOUD_API_BASE_URL="https://provider.example/api/v3"
CAR_CLOUD_MODEL="your-vision-model"
CAR_CLOUD_API_KEY=""
CAR_CLOUD_RESPONSE_FORMAT="json_schema"
```

基础路径 `/v1`、`/api/v3` 或完整 `/chat/completions` 均可；不重复追加版本。
其他厂商若只支持 JSON Object，可改为 `json_object`，本地校验仍执行相同 Schema。
通用适配器不发送千问的思考参数；厂商专有的思考/图片/缓存参数应另加 adapter，不能假设换地址就支持全部能力。
对外只允许 HTTPS；HTTP 仅用于本机回环测试。不跟随重定向，避免向未知地址转发凭据或图片。

## 历史兼容

`mock_client.py` 的左右变道客户端保持原样，通过 `LegacyCloudClient` 显式导入。
`car/test/closed_loop_test.py` 已改为导入该历史入口。它验证旧动作合同，不用于本轮千问语义链路。
原 `CloudClient.request_decision` 调用方应迁移为 `request_scene` 并接本地仲裁；确需复现历史测试时使用 `from cloud_client import LegacyCloudClient`。
历史脚本引用的已删除 `run_closed_loop.py` 不是当前入口。

## 离线测试

```bash
python -m unittest discover -s car/test -p "test_cloud_scene*.py"
```

测试只用虚构 Key 和本机 127.0.0.1 HTTP 服务，不调用千问或车辆硬件。
依赖 Python 标准库，无需安装 OpenAI/DashScope SDK。

官方文档（2026-10-08 核对）：[Chat API](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)、[结构化输出](https://help.aliyun.com/zh/model-studio/qwen-structured-output)、[模型说明](https://help.aliyun.com/zh/model-studio/qwen3-8-max)。
