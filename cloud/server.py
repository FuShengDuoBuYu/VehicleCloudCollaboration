import base64
import binascii
import gc
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

CURRENT_DIR = Path(__file__).resolve().parent
QWEN_UTILS_SRC = CURRENT_DIR / "Qwen" / "qwen-vl-utils" / "src"
if str(QWEN_UTILS_SRC) not in sys.path:
    sys.path.insert(0, str(QWEN_UTILS_SRC))

from qwen_vl_utils import process_vision_info

DEFAULT_PROMPT = "请分析当前自动驾驶长尾场景。"
MODEL_PATH = str(CURRENT_DIR / "Qwen" / "Qwen2-VL-2B-Instruct")
MAX_NEW_TOKENS = 512

model = None
processor = None


class InferenceRequest(BaseModel):
    """JSON 模式请求体（兼容旧版客户端）"""

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
    decision: str = Field(description="模型输出结果", examples=["检测到施工锥桶与行人混行，建议减速并保持车道。"])


class ErrorResponse(BaseModel):
    detail: str = Field(description="错误信息")


def _resolve_prompt(user_prompt: Optional[str]) -> str:
    return user_prompt.strip() if user_prompt and user_prompt.strip() else DEFAULT_PROMPT


def _get_model_device() -> torch.device:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    if hasattr(model, "device") and model.device is not None:
        return model.device

    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _run_inference(parsed_image: Image.Image, user_prompt: str) -> str:
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model service is not ready")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": parsed_image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    device = _get_model_device()
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return output_text[0] if output_text else ""


def _load_model_and_processor() -> None:
    global model, processor

    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, **model_kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)


def _release_resources() -> None:
    global model, processor

    print("\n🛑 正在释放资源并关闭端口...")
    try:
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                # 某些量化实现不支持显式 to(cpu)，忽略并继续回收
                pass
            del model

        if processor is not None:
            del processor

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        model = None
        processor = None
        print("✨ 资源清理完毕")
    except Exception as exc:
        print(f"⚠️ 清理异常: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 云端推理服务启动中...")
    try:
        _load_model_and_processor()
        print("✅ 模型加载成功，监听端口: 9526")
    except Exception as exc:
        print(f"❌ 初始化失败: {exc}")
        raise

    yield

    _release_resources()


app = FastAPI(
    title="Vehicle Cloud Inference API",
    description="基于 Qwen2-VL 的自动驾驶长尾场景云端推理服务。",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    summary="服务健康检查",
    description="检查服务是否在线以及模型是否已完成初始化。",
)
async def health() -> dict:
    return {
        "status": "ok",
        "model_ready": model is not None and processor is not None,
        "cuda_available": torch.cuda.is_available(),
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
                        "decision": "前方检测到非机动车与施工障碍物，建议减速并保持车道。",
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
            "content": {"application/json": {"example": {"detail": "CUDA out of memory"}}},
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
                                "description": "可选。为空时默认：请分析当前自动驾驶长尾场景。",
                            },
                        },
                    },
                    "examples": {
                        "form_example": {
                            "summary": "表单上传示例",
                            "value": {
                                "prompt": "请识别异常目标并给出驾驶策略。",
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
                                "prompt": "请分析当前场景风险并给出建议。",
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
    prompt: Optional[str] = Form(default=None, description="可选提示词，留空则使用默认提示词。"),
):
    client_host = raw_req.client.host if raw_req.client else "unknown"
    print(f"📥 接收请求来自: {client_host}")

    try:
        parsed_image, user_prompt = await _parse_request_image(raw_req, image_file, image, prompt)
        decision = _run_inference(parsed_image, user_prompt)
        return InferenceResponse(status="success", decision=decision)
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
        limit_max_requests=None,
        reload=True,
    )
