# Vehicle Cloud Inference Service

基于 Qwen2-VL 的云端推理服务，提供自动驾驶长尾场景分析接口。

## 1. 功能说明

- 接口路径: `/predict`
- 支持两种输入方式:
  - `multipart/form-data`: 上传图片文件（字段 `image_file` 或 `image`）
  - `application/json`: 提交 `image_base64`
- `prompt` 为可选参数，留空会使用默认提示词:
  - `请分析当前自动驾驶长尾场景。`

新增健康检查接口:

- `GET /health`

## 2. 环境要求

- Python 3.10+
- 建议 CUDA 环境（GPU 推理更快）
- 依赖见项目根目录 `requirements.txt`

## 3. 启动方式

在项目根目录执行:

```bash
cd /home/car/Desktop/VehicleCloudCollaboration/cloud
conda activate car
pip install -r ../requirements.txt
python server.py
```

默认监听:

- 本地: `http://0.0.0.0:9526`
- Swagger 文档: `http://127.0.0.1:9526/docs`

## 4. 你当前的 FRP 访问地址

你已通过 FRP 暴露文档地址:

- `http://47.97.249.48:7003/docs#/default`

如果转发规则映射的是服务根地址，则接口推理调用基址通常为:

- `http://47.97.249.48:7003`

推理接口完整路径:

- `http://47.97.249.48:7003/predict`

健康检查完整路径:

- `http://47.97.249.48:7003/health`

## 5. 调用示例

### 5.1 multipart/form-data 上传

```bash
curl -X POST "http://47.97.249.48:7003/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image_file=@/path/to/test.jpg" \
  -F "prompt=请识别异常目标并给出驾驶建议"
```

也可使用兼容字段 `image`:

```bash
curl -X POST "http://47.97.249.48:7003/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/test.jpg"
```

### 5.2 JSON + Base64 上传

```bash
curl -X POST "http://47.97.249.48:7003/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "prompt": "请分析当前场景风险并给出建议"
  }'
```

## 6. 响应说明

成功响应:

```json
{
  "status": "success",
  "decision": "前方检测到非机动车与施工障碍物，建议减速并保持车道。"
}
```

常见错误:

- `400`: 参数错误、图片为空、base64 非法、图片解码失败
- `500`: 模型推理异常（如显存不足等）

## 7. OpenAPI 文档增强内容

在 Swagger 中你会看到:

- 接口摘要与详细描述
- 两种请求 Content-Type 同时展示
- `prompt` 标记为可选参数
- 请求/响应示例与错误示例

