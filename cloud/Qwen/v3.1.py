import torch
import uvicorn
import base64
import gc
import os
import sys
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image
from typing import Optional
from contextlib import asynccontextmanager

# --- 全局变量初始化 ---
model = None
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 声明全局变量必须在函数的最开始
    global model, processor

    print("🚀 云端推理服务启动中...")
    MODEL_PATH = "./Qwen2-VL-2B-Instruct"

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"✅ 模型加载成功，公网监听端口: 9526")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        raise e

    yield

    # --- 【关闭阶段】强制释放资源 ---
    print("\n🛑 正在释放资源并关闭端口...")
    try:
        if model is not None:
            # 将模型移至 CPU 并删除，确保显存释放
            model.to("cpu")
            del model
        if processor is not None:
            del processor

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # 显式重置全局变量
        model = None
        processor = None
        print("✨ 资源清理完毕")
    except Exception as e:
        print(f"⚠️ 清理异常: {e}")


app = FastAPI(lifespan=lifespan)


class InferenceRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None


@app.post("/predict")
async def predict(request: InferenceRequest, raw_req: Request):
    print(f"📥 接收请求来自: {raw_req.client.host}")
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        user_prompt = request.prompt if (request.prompt and request.prompt.strip()) else "请分析当前自动驾驶长尾场景。"

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        return {"status": "success", "decision": output_text[0]}
    except Exception as e:
        print(f"💥 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=9526,
        log_level="info",
        timeout_keep_alive=5,
        # 允许端口快速重用
        limit_max_requests=None,
    )
    server = uvicorn.Server(config)
    server.run()
