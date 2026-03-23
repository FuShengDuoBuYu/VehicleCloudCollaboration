import torch
import uvicorn
import base64
import signal
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image
from typing import Optional
from contextlib import asynccontextmanager

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 【启动时执行】
    print("🚀 正在加载模型，请稍候...")
    # 全局变量，方便在整个 app 中使用
    global model, processor
    MODEL_PATH = "/root/Qwen/Qwen2-VL-7B-Instruct"
    
    # 显存优化配置
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, # 建议使用 bfloat16 减少显存占用
        device_map="auto",
        attn_implementation="sdpa" #
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("✅ 模型加载完成，端口 6006 已就绪")
    
    yield
    
    # 【停止时执行】 (Ctrl+C 会触发这里)
    print("\n🛑 正在接收停止信号，开始清理资源...")
    # 显式释放模型并清空 CUDA 缓存
    if 'model' in globals():
        del model
    if 'processor' in globals():
        del processor
    torch.cuda.empty_cache()
    print("✨ 资源已释放，端口已安全关闭")

app = FastAPI(lifespan=lifespan)

# --- 你的推理模型定义保持不变 ---
class InferenceRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None

DEFAULT_PROMPT = "你是一个部署在云端的自动驾驶决策辅助系统。现在接收到一帧来自车辆前视摄像头的长尾（Corner Case）场景图像。请按以下要求进行分析：1. 环境感知 ):请识别并定位图像中所有关键物体（车辆、行人、异常障碍物等），给出它们的归一化坐标 [ymin, xmin, ymax, xmax]。特别描述图像中的“长尾”因素（如：路面抛洒物、异形车辆、复杂的施工区域、极端天气影响等）。判断当前车道的语义信息（如：实线、虚线、特殊导向线）。2. 场景判断 :基于感知的物体，分析当前面临的具体挑战（例如：前方道路被封锁、视野盲区有潜在横穿行人等）。3. 行驶建议:给出下一步的宏观决策建议（如：减速停车、向左变道绕行、保持现状等）"

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        user_prompt = request.prompt if (request.prompt and request.prompt.strip()) else DEFAULT_PROMPT
        
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        return {"status": "success", "decision": output_text[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 配置 uvicorn 行为
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=6006,
        log_level="info",
        # 核心参数：保持连接超时设短一些，有助于快速回收
        timeout_keep_alive=5 
    )