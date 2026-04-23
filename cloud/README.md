# Vehicle Cloud Inference Service

车云协同场景下的云端推理服务。车辆遇到自动驾驶长尾场景时，将前视摄像头图片上传至云端，由大模型分析后返回行驶指令（`left` / `right` / `straight` / `stop`），指导车辆下一步动作。

---

## 1. 文件结构

```
cloud/
├── server_qwen.py        # 正式服务：基于本地部署的 Qwen2.5-VL-3B 模型推理，需要 GPU 环境
├── server_qwen_mock.py   # Mock 服务：与 server_qwen.py 接口完全一致，跳过模型加载，始终返回固定指令，用于本地联调
├── server_api.py         # 正式服务：调用 Gemini 2.5 Flash-Lite 云端 API 推理，需要网络访问 Google API
├── server_api_mock.py    # Mock 服务：与 server_api.py 接口完全一致，跳过 API 调用，始终返回固定指令，用于本地联调
└── down.py               # 工具脚本：从 ModelScope 下载模型权重文件
```

> 正式服务与 Mock 服务的请求/响应格式完全相同，开发阶段建议使用 Mock 服务，无需 GPU 或网络即可启动。

---

## 2. 环境要求

- Python 3.10+
- 建议使用 conda 环境：

```bash
conda activate car
pip install -r requirements.txt
```

### 依赖说明

| 包 | 用途 | 必装 |
|----|------|------|
| `fastapi>=0.111.0` | Web 服务框架 | 所有文件 |
| `uvicorn[standard]>=0.29.0` | ASGI 服务器 | 所有文件 |
| `pydantic>=2.0.0` | 请求/响应数据校验 | 所有文件 |
| `Pillow>=10.0.0` | 图片解码与处理 | 所有文件 |
| `torch>=2.1.0` | 模型推理运行时 | 仅 `server_qwen.py` |
| `transformers>=4.49.0` | Qwen2.5-VL 模型加载 | 仅 `server_qwen.py` |
| `accelerate>=0.30.0` | 多设备自动分配 | 仅 `server_qwen.py` |
| `bitsandbytes>=0.43.0` | 4-bit 量化，节省显存 | 仅 `server_qwen.py` |
| `qwen_vl_utils` | 视觉输入预处理 | 仅 `server_qwen.py`，本地源码包，见下方说明 |

> **qwen_vl_utils 安装说明**：该包为本地源码，路径为 `./Qwen/qwen-vl-utils/src`，`server_qwen.py` 启动时会自动将其加入 `sys.path`，无需通过 pip 安装。
>
> **仅运行 Mock 或 Gemini 版本时**，`torch` / `transformers` / `accelerate` / `bitsandbytes` 均不需要安装。

### 额外环境要求

- `server_qwen.py`：需要 CUDA 环境 + Qwen2.5-VL-3B-Instruct 模型权重，放置于 `./Qwen/Qwen2.5-VL-3B-Instruct/`
- `server_api.py`：需要可访问 Google Generative Language API 的网络环境

---

## 3. 启动方式

激活 car 的conda环境

```bash
conda activate car
```

进入 cloud 目录后执行以下任一命令：

### 3.1 Qwen 本地模型（正式）

```bash
uvicorn server_qwen:app --host 0.0.0.0 --port 9526 --reload
```

### 3.2 Qwen Mock（推荐用于联调）

```bash
uvicorn server_qwen_mock:app --host 0.0.0.0 --port 9526 --reload
```

### 3.3 Gemini API（正式）

```bash
uvicorn server_api:app --host 0.0.0.0 --port 9526 --reload
```

### 3.4 Gemini Mock（推荐用于联调）

```bash
uvicorn server_api_mock:app --host 0.0.0.0 --port 9526 --reload
```

启动成功后服务监听地址：`http://0.0.0.0:9526`

---

## 4. 接口文档

服务启动后，浏览器打开以下地址即可查看并在线测试接口：

| 地址 | 说明 |
|------|------|
| `http://127.0.0.1:9526/docs` | Swagger UI，支持在线填参数、上传图片、直接发请求 |
| `http://127.0.0.1:9526/redoc` | ReDoc 文档（只读） |
| `http://127.0.0.1:9526/health` | 健康检查，确认服务是否在线 |

若通过 FRP 暴露到公网，将 `127.0.0.1:9526` 替换为对应的公网地址和端口即可。

---

## 5. 接口说明与参数传递

```
POST /predict
```

支持两种传参方式：

#### 方式一：multipart/form-data（上传图片文件，推荐）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_file` | file | 二选一 | 推荐字段，上传 jpg/png 图片 |
| `image` | file | 二选一 | 兼容字段，与 `image_file` 等价 |
| `prompt` | string | 否 | 自定义提示词，留空使用默认驾驶提示词 |


#### 方式二：application/json（Base64 编码图片）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_base64` | string | 是 | 图片的 Base64 编码字符串（不带 `data:image` 前缀） |
| `prompt` | string | 否 | 自定义提示词，留空使用默认驾驶提示词 |
