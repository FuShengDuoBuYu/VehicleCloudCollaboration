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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

CURRENT_DIR = Path(__file__).resolve().parent
# Qwen2.5-VL 需要 qwen-vl-utils >= 0.0.8，确保使用最新版本
QWEN_UTILS_SRC = CURRENT_DIR / "Qwen" / "qwen-vl-utils" / "src"
if str(QWEN_UTILS_SRC) not in sys.path:
    sys.path.insert(0, str(QWEN_UTILS_SRC))

from qwen_vl_utils import process_vision_info

DEFAULT_PROMPT = "请分析当前自动驾驶长尾场景。"

# 驾驶指令+描述 合并 prompt：一次推理同时返回指令和中文描述
# 格式固定为两行，便于解析：
#   第一行：指令（left / right / straight / stop）
#   第二行：中文场景描述
DRIVING_CMD_PROMPT = (
    "You are an autonomous driving assistant for a small indoor/outdoor RC car (Donkey Car).\n"
    "Analyze the front-camera image and decide the steering command.\n"
    "\n"
    "=== COMMAND DEFINITIONS ===\n"
    "Commands describe WHERE THE VEHICLE SHOULD STEER, not where obstacles are located.\n"
    "\n"
    "straight — Keep going forward at current speed.\n"
    "  Use when:\n"
    "  • Road ahead is clear, follow the lane.\n"
    "  • Traffic cone on the SIDE of the road (not blocking) → straight (pass alongside).\n"
    "  • Narrow passage but still passable straight ahead → straight.\n"
    "\n"
    "stop — Halt the vehicle immediately and wait.\n"
    "  Use when:\n"
    "  • Pedestrian anywhere in the scene → always stop.\n"
    "  • Obstacle (cone, box, barrier) directly blocking the forward path → stop and wait.\n"
    "  • No safe passage in any direction → stop.\n"
    "\n"
    "left — Steer the vehicle to the LEFT.\n"
    "  Use ONLY when:\n"
    "  • The drivable road/path visibly bends or turns left.\n"
    "  • A row of cones or barriers channels the vehicle leftward.\n"
    "  • The only open passage is to the left.\n"
    "\n"
    "right — Steer the vehicle to the RIGHT.\n"
    "  Use ONLY when:\n"
    "  • The drivable road/path visibly bends or turns right.\n"
    "  • A row of cones or barriers channels the vehicle rightward.\n"
    "  • The only open passage is to the right.\n"
    "\n"
    "=== CRITICAL RULES ===\n"
    "1. PEDESTRIAN anywhere → always output: stop\n"
    "2. Single cone / multiple cones blocking center → stop\n"
    "3. Cones forming a LEFT channel → left\n"
    "4. Cones forming a RIGHT channel → right\n"
    "5. NEVER output left/right just because an obstacle is on the left/right side.\n"
    "   Obstacle position ≠ steering direction.\n"
    "6. When in doubt, output: stop\n"
    "\n"
    "=== OUTPUT FORMAT (exactly two lines, no extra text) ===\n"
    "Line 1: one word only — left, right, straight, or stop\n"
    "Line 2: Chinese description of scene and reason, within 25 characters\n"
    "\n"
    "=== EXAMPLES ===\n"
    "straight\n"
    "道路平直无障碍，保持前行。\n"
    "\n"
    "straight\n"
    "路侧有锥桶，不影响直行通道。\n"
    "\n"
    "stop\n"
    "前方有行人，停车等待通过。\n"
    "\n"
    "stop\n"
    "行人在左侧，停车等待。\n"
    "\n"
    "stop\n"
    "锥形桶挡在正前方，停车等待。\n"
    "\n"
    "left\n"
    "锥桶引导向左，左转通过。\n"
    "\n"
    "left\n"
    "前方路口向左弯曲，需左转。\n"
    "\n"
    "right\n"
    "锥桶引导向右，右转通过。\n"
    "\n"
    "right\n"
    "道路在前方向右转弯，需右转。"
)
VALID_COMMANDS = {"left", "right", "straight", "stop"}
MODEL_PATH = str(CURRENT_DIR / "Qwen" / "Qwen2.5-VL-3B-Instruct")
MAX_NEW_TOKENS = 80  # 需要容纳指令行 + 中文描述行

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
    command: str = Field(
        description="转向指令：left / right / straight / stop",
        examples=["straight"],
    )
    raw_output: str = Field(
        description="模型原始输出（用于调试）",
        examples=["straight\n道路平直，前方无障碍，保持直行。"],
    )


class ErrorResponse(BaseModel):
    detail: str = Field(description="错误信息")


def _resolve_prompt(user_prompt: Optional[str]) -> str:
    """用户未传入 prompt 时使用默认场景分析提示词。"""
    return user_prompt.strip() if user_prompt and user_prompt.strip() else DEFAULT_PROMPT


def _parse_driving_output(raw_text: str) -> str:
    import re

    lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]

    for i, line in enumerate(lines):
        normalized = line.lower()
        if normalized in VALID_COMMANDS:
            return normalized
        for cmd in ("straight", "stop", "left", "right"):
            if re.search(rf"\b{cmd}\b", normalized):
                return cmd

    print(f"⚠️ 无法解析指令，原始输出: {repr(raw_text)}，降级为 stop")
    return "stop"


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
    """调用 Qwen2-VL 模型推理，返回原始文本输出。"""
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
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return output_text[0] if output_text else ""


def _run_driving_inference(parsed_image: Image.Image) -> tuple[str, str]:
    raw_output = _run_inference(parsed_image, DRIVING_CMD_PROMPT)
    command = _parse_driving_output(raw_output)
    print(f"🚗 指令: {command} | 原始: {repr(raw_output)}")
    return command, raw_output


def _load_model_and_processor() -> None:
    global model, processor

    use_cuda = torch.cuda.is_available()

    quantization_config = None
    if use_cuda:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # 二次量化，额外节省约15%显存
            bnb_4bit_quant_type="nf4",
        )

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if use_cuda else torch.float32,
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    # Qwen2.5-VL 使用专属类 Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, **model_kwargs)
    model.eval()  # 关闭 dropout，节省显存
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=448 * 448,
    )


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
        parsed_image, _ = await _parse_request_image(raw_req, image_file, image, prompt)
        command, raw_output = _run_driving_inference(parsed_image)
        return InferenceResponse(status="success", command=command, raw_output=raw_output)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"💥 错误: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(
        "server_qwen:app",
        host="0.0.0.0",
        port=9526,
        log_level="info",
        timeout_keep_alive=5,
        limit_max_requests=None,
        reload=True,
    )
