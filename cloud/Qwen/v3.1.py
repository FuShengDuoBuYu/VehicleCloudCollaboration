import torch
import uvicorn
import base64
import gc
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image
from typing import Optional
from contextlib import asynccontextmanager

# 全局变量初始化
model = None
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 【启动阶段】 ---
    print("🚀 云端推理服务启动中...")
    global model, processor
    MODEL_PATH = "./Qwen2-VL-7B-Instruct"

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

    # --- 【关闭阶段】 ---
    # 这是关键：当 Ctrl+C 或 kill 信号到达时，强制执行清理
    print("\n🛑 接收到停止信号，正在释放资源并关闭端口...")
    try:
        global model, processor
        if model is not None:
            del model
        if processor is not None:
            del processor

        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # 清理多进程显存残留

        print("✨ 显存资源已释放")
    except Exception as e:
        print(f"⚠️ 清理过程出现异常: {e}")
    finally:
        print("💤 服务已彻底退出")


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
    # 使用 uvicorn.Config 配合 uvicorn.Server 可以更精细控制停止行为
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=9526,
        log_level="info",
        timeout_keep_alive=5,
        loop="auto"
    )
    server = uvicorn.Server(config)
    server.run()