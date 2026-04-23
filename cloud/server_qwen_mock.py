import base64
import binascii
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

CURRENT_DIR = Path(__file__).resolve().parent

VALID_COMMANDS = {"left", "right", "straight", "stop"}


class InferenceRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64 编码后的图像字符串（不带 data:image 前缀）",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD..."],
    )
    prompt: Optional[str] = Field(
        default=None,
        description="可选文本提示词。留空时使用默认提示词。",
        examples=["请判断前方异常并给出驾驶建议。"],
    )


class InferenceResponse(BaseModel):
    status: str = Field(description="请求处理状态", examples=["success"])
    command: str = Field(
        description="转向指令：left / right / straight / stop",
        examples=["straight"],
    )
    raw_output: str = Field(
        description="模型原始输出（用于调试）",
        examples=["left\n前方障碍物占道，左侧空间充裕，建议向左绕行。"],
    )


class ErrorResponse(BaseModel):
    detail: str = Field(description="错误信息")


def _decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_data = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc

    try:
        return Image.open(BytesIO(image_data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image content: {exc}") from exc


async def _parse_request_image(
    raw_req: Request,
    image_file: UploadFile | None,
    image_alias: UploadFile | None,
    prompt: Optional[str],
) -> Image.Image:
    upload = image_file or image_alias

    if upload is not None:
        image_bytes = await upload.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid uploaded image: {exc}") from exc

    try:
        body = await raw_req.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Expected multipart file or JSON with image_base64.",
        ) from exc

    try:
        payload = InferenceRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc.errors()}") from exc

    return _decode_base64_image(payload.image_base64)


def _run_driving_inference(parsed_image: Image.Image) -> tuple[str, str]:
    # Mock: 跳过 Qwen 模型推理，固定返回 left 指令
    command = "left"
    raw_output = "left\n前方障碍物占道，左侧空间充裕，建议向左绕行。"
    print(f"🚗 [MOCK] 指令: {command} | 原始: {repr(raw_output)}")
    return command, raw_output


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [MOCK] Qwen 云端推理服务启动中...")
    print("✅ [MOCK] 已跳过模型加载，监听端口: 9526")
    yield
    print("🛑 [MOCK] Qwen 云端推理服务关闭")


app = FastAPI(
    title="Vehicle Cloud Inference API (Mock)",
    description="Qwen2-VL 自动驾驶推理服务 Mock 版，固定返回 left 指令。",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    summary="服务健康检查",
    description="检查服务是否在线。",
)
async def health() -> dict:
    return {
        "status": "ok",
        "model_ready": True,
        "mock": True,
    }


@app.post(
    "/predict",
    response_model=InferenceResponse,
    summary="多模态场景推理（Mock）",
    description=(
        "支持两种输入方式：\n"
        "1) multipart/form-data: 上传图像文件（字段名 image_file 或 image）+ 可选 prompt；\n"
        "2) application/json: 提交 image_base64 + 可选 prompt。\n\n"
        "Mock 版本：始终返回 left 指令。"
    ),
    responses={
        200: {
            "description": "推理成功",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "command": "left",
                        "raw_output": "left\n前方障碍物占道，左侧空间充裕，建议向左绕行。",
                    }
                }
            },
        },
        400: {
            "model": ErrorResponse,
            "description": "请求参数错误或图片格式非法",
            "content": {"application/json": {"example": {"detail": "Invalid image_base64: Incorrect padding"}}},
        },
        500: {
            "model": ErrorResponse,
            "description": "服务内部错误",
            "content": {"application/json": {"example": {"detail": "Internal server error"}}},
        },
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "image_file": {
                                "type": "string",
                                "format": "binary",
                                "description": "推荐字段：图片文件（jpg/png）。",
                            },
                            "image": {
                                "type": "string",
                                "format": "binary",
                                "description": "兼容字段：与 image_file 二选一。",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "可选。Mock 版本忽略此字段。",
                            },
                        },
                    },
                },
                "application/json": {
                    "schema": InferenceRequest.model_json_schema(),
                },
            },
        }
    },
)
async def predict(
    raw_req: Request,
    image_file: UploadFile | None = File(default=None, description="上传图片，推荐字段名。可选。"),
    image: UploadFile | None = File(default=None, description="兼容字段名。可选。"),
    prompt: Optional[str] = Form(default=None, description="可选提示词，Mock 版本忽略。"),
):
    client_host = raw_req.client.host if raw_req.client else "unknown"
    print(f"📥 接收请求来自: {client_host}")

    try:
        parsed_image = await _parse_request_image(raw_req, image_file, image, prompt)
        command, raw_output = _run_driving_inference(parsed_image)
        return InferenceResponse(status="success", command=command, raw_output=raw_output)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"💥 错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(
        "server_qwen_mock:app",
        host="0.0.0.0",
        port=9526,
        log_level="info",
        timeout_keep_alive=5,
        reload=True,
        reload_dirs=[str(CURRENT_DIR)],
    )
