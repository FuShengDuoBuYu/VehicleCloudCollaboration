import base64
import binascii
import json
import re
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

CURRENT_DIR = Path(__file__).resolve().parent

GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_API_KEY = "AIzaSyD2fTI2bmCw_P9izV4LD5r-FIQbv0bwPF0"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

VALID_COMMANDS = {"left", "right", "straight"}

DRIVING_CMD_PROMPT = """你是一个车云协同场景下的自动驾驶辅助决策助手，你接收到的是车辆前视摄像头图片，请根据图片给出本车辆下一步更安全的通行方向。

输出格式：
第一行:left/right/straight,只能是 left、right、straight 之一
第二行:描述你做出该指令的原因，必须用中文描述，且不能包含英文。

其中：
- left：向左绕行或向左转向
- right：向右绕行或向右转向
- straight：继续直行，或者减速直行/停车等待

示例 1:
left
前方锥桶和车辆占道，左侧空间更安全，建议向左绕行。

示例 2:
right
前方行人靠近本车道，右侧可安全通过，建议向右避让。

示例 3:
straight
前方道路基本畅通，未见明显障碍，建议保持直行。

"""

class InferenceRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64 编码后的图像字符串（不带 data:image 前缀）",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD..."],
    )
    prompt: Optional[str] = Field(
        default=None,
        description="可选文本提示词。留空时使用默认驾驶提示词。",
        examples=["请分析当前自动驾驶长尾场景并给出转向指令。"],
    )


class InferenceResponse(BaseModel):
    status: str = Field(description="请求处理状态", examples=["success"])
    command: str = Field(
        description="转向指令：left / right / straight",
        examples=["straight"],
    )
    raw_output: str = Field(
        description="大模型原始输出",
        examples=["straight\n道路平直，前方无障碍，保持直行。"],
    )


class ErrorResponse(BaseModel):
    detail: str = Field(description="错误信息")


def _resolve_prompt(user_prompt: Optional[str]) -> str:
    return user_prompt.strip() if user_prompt and user_prompt.strip() else DRIVING_CMD_PROMPT


def _parse_driving_output(raw_text: str) -> str:
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]

    for line in lines:
        normalized = line.lower()
        if normalized in VALID_COMMANDS:
            return normalized
        for candidate in ("left", "right", "straight"):
            if re.search(rf"\b{candidate}\b", normalized):
                return candidate

    return "straight"


def _decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_data = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc

    try:
        return Image.open(BytesIO(image_data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image content: {exc}") from exc


def _image_to_jpeg_base64(parsed_image: Image.Image) -> str:
    output = BytesIO()
    parsed_image.save(output, format="JPEG", quality=90)
    return base64.b64encode(output.getvalue()).decode("utf-8")


async def _parse_request_image(
    raw_req: Request,
    image_file: UploadFile | None,
    image_alias: UploadFile | None,
    prompt: Optional[str],
) -> tuple[Image.Image, str]:
    upload = image_file or image_alias

    if upload is not None:
        image_bytes = await upload.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            parsed_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid uploaded image: {exc}") from exc
        return parsed_image, _resolve_prompt(prompt)

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

    parsed_image = _decode_base64_image(payload.image_base64)
    return parsed_image, _resolve_prompt(payload.prompt)


def _extract_gemini_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            text = part.get("text")
            if text:
                return text.strip()
    return ""


def _parse_gemini_error(error_body: str) -> str:
    try:
        parsed = json.loads(error_body)
        return parsed.get("error", {}).get("message", error_body)
    except Exception:
        return error_body


def _call_gemini_api(parsed_image: Image.Image, prompt: str) -> str:
    image_b64 = _image_to_jpeg_base64(parsed_image)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64,
                        }
                    },
                    {
                        "text": prompt,
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 256,
        },
    }

    request = urllib.request.Request(
        url=GEMINI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            output_text = _extract_gemini_text(data)
            if not output_text:
                raise HTTPException(status_code=502, detail="Gemini returned empty response")
            return output_text
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        detail = _parse_gemini_error(error_body)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API network error: {exc}") from exc


def _run_driving_inference(parsed_image: Image.Image, user_prompt: Optional[str] = None) -> tuple[str, str]:
    prompt_to_use = _resolve_prompt(user_prompt)
    raw_output = _call_gemini_api(parsed_image, prompt_to_use)
    command = _parse_driving_output(raw_output)
    print(f"🚗 指令: {command} | 模型原始: {repr(raw_output)}")
    return command, raw_output


def _validate_api_key() -> None:
    if not GEMINI_API_KEY.strip():
        raise RuntimeError("Gemini API key is empty")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Gemini 云端推理服务启动中...")
    try:
        _validate_api_key()
        print(f"✅ Gemini API 已配置，模型: {GEMINI_MODEL}，监听端口: 9526")
    except Exception as exc:
        print(f"❌ 初始化失败: {exc}")
        raise

    yield

    print("🛑 Gemini 云端推理服务关闭")


app = FastAPI(
    title="Vehicle Cloud Inference API",
    description="基于 Gemini 2.5 Flash-Lite 的自动驾驶长尾场景云端推理服务。",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    summary="服务健康检查",
    description="检查服务是否在线以及 Gemini API key 是否已配置。",
)
async def health() -> dict:
    return {
        "status": "ok",
        "model": GEMINI_MODEL,
        "api_key_configured": bool(GEMINI_API_KEY.strip()),
    }


@app.post(
    "/predict",
    response_model=InferenceResponse,
    summary="多模态场景推理",
    description=(
        "支持两种输入方式：\n"
        "1) multipart/form-data: 上传图像文件（字段名 image_file 或 image）+ 可选 prompt；\n"
        "2) application/json: 提交 image_base64 + 可选 prompt。"
    ),
    responses={
        200: {
            "description": "推理成功",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "command": "straight",
                        "raw_output": "straight\n道路平直，前方无障碍，保持直行。",
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
        502: {
            "model": ErrorResponse,
            "description": "Gemini 接口调用失败",
            "content": {"application/json": {"example": {"detail": "Gemini API error: request failed"}}},
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
                                "description": "可选。为空时默认使用驾驶提示词；传入后可直接测试自定义提示词效果。",
                            },
                        },
                    },
                    "examples": {
                        "form_example": {
                            "summary": "表单上传示例",
                            "value": {
                                "prompt": "请分析当前画面，并严格按两行输出：第一行 left/right/straight，第二行中文原因。",
                            },
                        }
                    },
                },
                "application/json": {
                    "schema": InferenceRequest.model_json_schema(),
                    "examples": {
                        "json_example": {
                            "summary": "JSON Base64 示例",
                            "value": {
                                "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
                                "prompt": "请分析当前画面，并严格按两行输出：第一行 left/right/straight，第二行中文原因。",
                            },
                        }
                    },
                },
            },
        }
    },
)
async def predict(
    raw_req: Request,
    image_file: UploadFile | None = File(default=None, description="上传图片，推荐字段名。可选。"),
    image: UploadFile | None = File(default=None, description="兼容字段名。可选。"),
    prompt: Optional[str] = Form(default=None, description="可选提示词，留空则使用默认驾驶提示词。"),
):
    client_host = raw_req.client.host if raw_req.client else "unknown"
    print(f"📥 接收请求来自: {client_host}")

    try:
        parsed_image, resolved_prompt = await _parse_request_image(raw_req, image_file, image, prompt)
        command, raw_output = _run_driving_inference(parsed_image, resolved_prompt)
        return InferenceResponse(status="success", command=command, raw_output=raw_output)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"💥 错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=9526,
        log_level="info",
        timeout_keep_alive=5,
        reload=True,
        reload_dirs=[str(CURRENT_DIR)],
    )
