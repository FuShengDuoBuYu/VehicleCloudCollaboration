import requests
import base64
import time

def send_to_cloud(image_path, custom_prompt=None):
    # AutoDL 公网基础地址
    base_url = "https://u924674-gxz9-29b71154.bjb1.seetacloud.com:8443"
    url = f"{base_url}/predict"

    # 1. 准备阶段提示
    print(f"--- 🚗 自动驾驶云端决策系统 ---")
    print(f"[1/3] 正在读取图像: {image_path}...")

    try:
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        # 2. 提示词确认
        display_prompt = custom_prompt if custom_prompt else "使用云端默认长尾场景分析指令"
        print(f"[2/3] 正在构建请求 (提示词: {display_prompt[:20]}...)")

        payload = {
            "image_base64": img_base64,
            "prompt": custom_prompt
        }

        # 3. 发送请求并开始计时
        print(f"[3/3] 🚀 正在上传数据并等待云端 Qwen2-VL 推理，请稍候...", end="", flush=True)

        start_time = time.time()

        # 注意：如果 AutoDL 证书报错，可以添加 verify=False
        response = requests.post(url, json=payload, timeout=60)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\r[3/3] ✅ 云端推理完成！(耗时: {duration:.2f}s)               ")  # 清除之前的提示

        if response.status_code == 200:
            result = response.json()
            print("-" * 40)
            print(f"🤖 云端决策建议：\n\n{result.get('decision', '无返回内容')}")
            print("-" * 40)
        else:
            print(f"❌ 服务器返回错误 ({response.status_code}): {response.text}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到图片文件 '{image_path}'")
    except requests.exceptions.Timeout:
        print(f"❌ 错误: 云端响应超时，请检查 AutoDL 实例是否在线或网络状况。")
    except Exception as e:
        print(f"❌ 请求发生异常: {e}")


# 调用示例
if __name__ == "__main__":
    # 你可以尝试传入自定义提示词，或者保持 None 使用默认值
    custom_prompt = "你是一个部署在云端的自动驾驶决策辅助系统。现在接收到一帧来自车辆前视摄像头的长尾（Corner Case）场景图像。请按以下要求进行分析：1. 环境感知 ):请识别并定位图像中所有关键物体（车辆、行人、异常障碍物等），给出它们的归一化坐标 [ymin, xmin, ymax, xmax]。特别描述图像中的“长尾”因素（如：路面抛洒物、异形车辆、复杂的施工区域、极端天气影响等）。判断当前车道的语义信息（如：实线、虚线、特殊导向线）。2. 场景判断 :基于感知的物体，分析当前面临的具体挑战（例如：前方道路被封锁、视野盲区有潜在横穿行人等）。3. 行驶建议:给出下一步的宏观决策建议（如：减速停车、向左变道绕行、保持现状等）"
    send_to_cloud("1.jpg", custom_prompt)