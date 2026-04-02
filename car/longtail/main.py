# Main script for long-tail scenario classification

import argparse
import yaml
import os
import glob
import time
import sys
from datetime import datetime

# Add utils directory to path for car control imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'control', 'utils'))
from McLumk_Wheel_Sports import move_forward, stop
from visualizer import Visualizer

from classifier import LongTailClassifier


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def process_single_image(classifier: LongTailClassifier, image_path: str, verbose: bool = True):
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"{'='*70}")
    result = classifier.predict(image_path)
    if verbose:
        classifier.print_summary()
    else:
        status = "LONG-TAIL ⚠️" if result['is_long_tail'] else "NORMAL ✓"
        print(f"Result: {status} (score: {result['score']:.3f}, FPS: {result['fps']:.2f})")
    return result


def process_directory(classifier: LongTailClassifier, directory: str, output_file: str = None):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    if not image_files:
        print(f"No images found in {directory}")
        return
    print(f"\nFound {len(image_files)} images in {directory}")
    results = []
    long_tail_count = 0
    total_fps = 0.0
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(img_path)}")
        result = classifier.predict(img_path)
        results.append({'image': img_path, 'is_long_tail': result['is_long_tail'], 'score': result['score'], 'fps': result['fps']})
        if result['is_long_tail']:
            long_tail_count += 1
        total_fps += result['fps']
        status = "LONG-TAIL ⚠️" if result['is_long_tail'] else "NORMAL ✓"
        print(f"  Result: {status} (score: {result['score']:.3f}, FPS: {result['fps']:.2f})")
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Long-tail scenarios: {long_tail_count} ({100*long_tail_count/len(image_files):.1f}%)")
    print(f"Normal scenarios: {len(image_files)-long_tail_count} ({100*(len(image_files)-long_tail_count)/len(image_files):.1f}%)")
    print(f"Average FPS: {total_fps/len(image_files):.2f}")
    print(f"{'='*70}")
    if output_file:
        with open(output_file, 'w') as f:
            f.write("image_path,is_long_tail,score,fps\n")
            for r in results:
                f.write(f"{r['image']},{r['is_long_tail']},{r['score']:.3f},{r['fps']:.2f}\n")
        print(f"\nResults saved to: {output_file}")


def process_camera_stream(
    classifier: LongTailClassifier,
    camera_index: int = 0,
    interval_sec: float = 5.0,
    save_dir: str = "captured_frames",
    width: int = 640,
    height: int = 480,
    verbose: bool = False,
    speed: int = 100,
    visualize: bool = False,
    viz_port: int = 8080,
):
    """Capture from camera every interval and run long-tail classification."""
    try:
        import cv2
    except ImportError as e:
        print(f"OpenCV is required for camera capture: {e}")
        return

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Failed to open camera index {camera_index}")
        return

    viz = None
    if visualize:
        viz = Visualizer(port=viz_port)
        viz.start()

    print(f"\n{'='*70}")
    print(f"Camera capture started (index={camera_index}, every {interval_sec}s)")
    print(f"Frames will be saved to: {save_dir}")
    print(f"Car initialized - Normal Speed: {speed}")
    print(f"{'='*70}")

    # Start moving forward initially
    print("🚗 开始前进...")
    move_forward(speed)

    last_capture = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 读取帧失败，等待 1s 后重试")
                time.sleep(1)
                continue

            if viz:
                viz.update_live(frame)

            now = time.time()
            if now - last_capture >= interval_sec:
                path = os.path.join(save_dir, "latest_frame.jpg")
                cv2.imwrite(path, frame)
                print(f"📸 更新分析帧: {path}")

                result = classifier.predict(path)
                
                if viz:
                    viz.update_analysis(frame, result)

                if verbose:
                    classifier.print_summary()
                else:
                    status = "LONG-TAIL ⚠️" if result['is_long_tail'] else "NORMAL ✓"
                    print(f"   → 结果: {status} | 分数: {result['score']:.3f} | FPS: {result['fps']:.2f}")

                # Control car based on classification
                if result['is_long_tail']:
                    print("🛑 [警告] 检测到长尾场景！紧急制动...")
                    stop()
                else:
                    print(f"✅ [正常] 场景正常，继续以速度 {speed} 前进")
                    move_forward(speed)

                last_capture = now

            # Removed artificial sleep to improve camera streaming responsiveness
    except KeyboardInterrupt:
        print("\n🛑 手动停止摄像头识别")
    finally:
        print("🛑 停止小车运动")
        stop()
        cap.release()
        print("✅ 摄像头已释放")


def main():
    parser = argparse.ArgumentParser(description="Long-tail scenario classifier")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--directory', type=str, help='Path to directory containing images')
    parser.add_argument('--output', type=str, help='Output CSV file for batch processing results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    parser.add_argument('--threshold', type=float, help='Override classification threshold from config')
    parser.add_argument('--camera-index', type=int, default=0, help='Camera index to use when no image/directory is provided')
    parser.add_argument('--camera-interval', type=float, default=2.0, help='Seconds between frames when using camera input')
    parser.add_argument('--camera-save-dir', type=str, default='captured_frames', help='Where to store captured frames when using camera input')
    parser.add_argument('--camera-width', type=int, default=640, help='Camera capture width')
    parser.add_argument('--camera-height', type=int, default=480, help='Camera capture height')
    parser.add_argument('--speed', type=int, default=5, help='Car movement speed (0-255)')
    parser.add_argument('--visualize', action='store_true', default=True, help='Start real-time monitoring dashboard')
    parser.add_argument('--viz-port', type=int, default=8080, help='Dashboard port')
    args = parser.parse_args()
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = {}
    if args.threshold is not None:
        config['threshold'] = args.threshold
    print("\nInitializing Long-tail Classifier...")
    classifier = LongTailClassifier(config)
    print(f"Classifier initialized with {len(classifier.detectors)} detectors")
    if args.image:
        process_single_image(classifier, args.image, verbose=args.verbose)
    elif args.directory:
        process_directory(classifier, args.directory, output_file=args.output)
    else:
        process_camera_stream(
            classifier,
            camera_index=args.camera_index,
            interval_sec=args.camera_interval,
            save_dir=args.camera_save_dir,
            width=args.camera_width,
            height=args.camera_height,
            verbose=args.verbose,
            speed=args.speed,
            visualize=args.visualize,
            viz_port=args.viz_port,
        )


if __name__ == '__main__':
    main()
