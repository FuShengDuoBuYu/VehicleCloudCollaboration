"""
交通标志和红绿灯检测系统
使用 YOLOv8 + GTSDB 数据集训练的模型
"""
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path


class TrafficSignDetector:
    """交通标志检测器"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径（可以是预训练模型或自定义模型）
            conf_threshold: 置信度阈值
        """
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # 德国交通标志类别（GTSDB 数据集）
        self.gtsdb_classes = {
            0: '限速20', 1: '限速30', 2: '限速50', 3: '限速60', 4: '限速70',
            5: '限速80', 6: '解除80限速', 7: '限速100', 8: '限速120',
            9: '禁止超车', 10: '禁止3.5吨以上车辆超车',
            11: '优先路口', 12: '优先道路', 13: '让行', 14: '停车让行',
            15: '禁止驶入', 16: '禁止3.5吨以上车辆驶入',
            17: '禁止通行', 18: '注意', 19: '注意左弯', 20: '注意右弯',
            21: '注意连续弯路', 22: '注意颠簸路', 23: '注意路面打滑',
            24: '注意右侧变窄', 25: '注意施工', 26: '注意信号灯',
            27: '注意行人', 28: '注意儿童', 29: '注意自行车',
            30: '注意雪/冰', 31: '注意野生动物', 32: '解除所有限制',
            33: '右转', 34: '左转', 35: '直行', 36: '直行或右转',
            37: '直行或左转', 38: '保持右行', 39: '保持左行',
            40: '环岛', 41: '解除禁止超车', 42: '解除禁止3.5吨以上超车'
        }
        
        print("✓ Model loaded successfully\n")
    
    def detect(self, image_path, save_path='result.jpg'):
        """
        检测单张图片
        
        Args:
            image_path: 输入图片路径
            save_path: 输出图片路径
        
        Returns:
            detections: 检测结果列表
        """
        separator = "=" * 70
        print(separator)
        print(f"Processing: {image_path}")
        print(separator)
        print()
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        h, w = image.shape[:2]
        print(f"Image size: {w}x{h}")
        
        # 推理
        print("Running inference...")
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        # 解析结果
        detections = []
        print("\nDetected objects:")
        
        for result in results:
            boxes = result.boxes
            
            if len(boxes) == 0:
                print("  No objects detected")
                continue
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # 获取类别名称
                if hasattr(result, 'names'):
                    class_name = result.names[cls_id]
                else:
                    class_name = self.gtsdb_classes.get(cls_id, f'Unknown-{cls_id}')
                
                detection = {
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': xyxy.tolist()
                }
                detections.append(detection)
                
                print(f"  {i+1}. {class_name}: {conf:.3f} at [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")
        
        # 可视化
        print("\nGenerating visualization...")
        vis_img = self._visualize(image, detections)
        
        # 保存结果
        cv2.imwrite(save_path, vis_img)
        print(f"✓ Result saved to: {save_path}")
        print(separator)
        print()
        
        return detections
    
    def _visualize(self, image, detections):
        """可视化检测结果"""
        vis_img = image.copy()
        
        # 定义颜色
        colors = [
            (0, 0, 255),    # 红色 - 禁止类
            (0, 255, 255),  # 黄色 - 警告类
            (0, 255, 0),    # 绿色 - 指示类
            (255, 0, 0),    # 蓝色 - 强制类
        ]
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 根据类别选择颜色
            if '禁止' in class_name or 'stop' in class_name.lower():
                color = colors[0]  # 红色
            elif '注意' in class_name or '让行' in class_name or 'warning' in class_name.lower():
                color = colors[1]  # 黄色
            elif '限速' in class_name or 'speed' in class_name.lower():
                color = colors[3]  # 蓝色
            else:
                color = colors[2]  # 绿色
            
            # 绘制边界框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
            
            # 绘制标签背景
            label = f'{class_name} {conf:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(vis_img,
                         (x1, y1 - label_h - 10),
                         (x1 + label_w, y1),
                         color, -1)
            
            # 绘制标签文字
            cv2.putText(vis_img, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
        
        return vis_img
    
    def detect_video(self, video_path, output_path='output.mp4'):
        """检测视频"""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 每帧都检测（可以改为每N帧检测一次以提升速度）
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # 绘制结果
            annotated_frame = results[0].plot()
            
            # 写入输出视频
            out.write(annotated_frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"✓ Video saved to: {output_path}")


def download_gtsdb_model():
    """
    下载 GTSDB 数据集训练的模型
    使用 Roboflow 平台
    """
    print("Downloading GTSDB model from Roboflow...")
    
    try:
        from roboflow import Roboflow
        
        # 初始化 Roboflow（需要 API key）
        print("\nTo download the model, you need a Roboflow API key.")
        print("1. Go to https://app.roboflow.com/")
        print("2. Sign up or log in")
        print("3. Go to Settings > API")
        print("4. Copy your API key\n")
        
        api_key = input("Enter your Roboflow API key (or press Enter to skip): ").strip()
        
        if api_key:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("marouane1").project("gtsdb-dat55-8sucs")
            dataset = project.version(1).download("yolov8")
            
            print(f"✓ Dataset downloaded to: {dataset.location}")
            print(f"✓ You can now train with: ultralytics train")
            return dataset.location
        else:
            print("Skipped download. Using default YOLOv8 model.")
            return None
            
    except ImportError:
        print("roboflow not installed. Install with: pip install roboflow")
        return None


def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Detection')
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--video', type=str, help='Input video path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path')
    parser.add_argument('--output', type=str, default='result.jpg', help='Output path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--download-model', action='store_true', help='Download GTSDB model')
    args = parser.parse_args()
    
    # 下载模型
    if args.download_model:
        download_gtsdb_model()
        return
    
    # 创建检测器
    detector = TrafficSignDetector(args.model, args.conf)
    
    # 处理图像或视频
    if args.image:
        detector.detect(args.image, args.output)
    elif args.video:
        detector.detect_video(args.video, args.output)
    else:
        print("Please specify --image or --video")
        print("Example:")
        print("  python traffic_sign_detector.py --image road.jpg")
        print("  python traffic_sign_detector.py --video road.mp4")


if __name__ == '__main__':
    main()
