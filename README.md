# Long-tail Scenario Classifier

A multi-model ensemble system designed to detect long-tail (corner case) driving scenarios using a combination of semantic understanding, open-vocabulary detection, and heuristic analysis.

## Structure

```
VehicleCloudCollaboration/
├── main.py                # CLI; image/dir processing or camera fallback every 5s
├── capture_every_5s.py    # Standalone camera capture + classify loop
├── classifier.py          # Core ensemble classifier
├── config.yaml            # System configuration (weights, thresholds, detectors)
├── benchmark.py           # Performance benchmarking tool
├── evaluate.py            # Dataset evaluation tool
├── requirements.txt       # Python dependencies
├── detectors/             # Detector implementations
│   ├── base_detector.py
│   ├── clip_detector.py
│   ├── yoloworld_detector.py
│   ├── yolov8_detector.py
│   ├── yolopv2_detector.py
│   └── YOLOv8-multi-task/ # Vendor code, weights, and demos
├── weights/               # Model weights directory
│   ├── yolov8n.pt
│   ├── yolov8x-worldv2.pt
│   └── yolopv2.pt
├── dataset/               # Sample datasets (BDD100K, CODA, etc.)
└── runs/                  # Output runs and predictions
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure weights are present in `weights/` directory.

## Usage

### Single image
```bash
python main.py --image path/to/image.jpg --verbose
```

### Directory batch
```bash
python main.py --directory path/to/images --output results.csv
```

### Camera (fallback when no --image/--directory)
- Default: if you run without `--image` or `--directory`, the app opens the camera, captures one frame every 5s, saves it to `captured_frames/`, and classifies each frame.
- You can override camera settings:
```bash
python main.py \
   --camera-index 0 \
   --camera-interval 5 \
   --camera-save-dir captured_frames \
   --camera-width 640 \
   --camera-height 480 \
   --verbose
```

### Benchmarking
Run performance tests on a sample image:
```bash
python benchmark.py --image path/to/image.jpg
```

### Evaluation
Evaluate accuracy on a labeled dataset:
```bash
python evaluate.py --dataset path/to/dataset --output evaluation_results/
```

## Configuration

Adjust weights and thresholds in `config.yaml` to tune the sensitivity of the ensemble. Camera capture requires OpenCV (`opencv-python`).

## YOLOv8-multi-task
## Setup
This codebase has been developed with Python==3.7.16 with PyTorch==1.13.1.

cd YOLOv8-multi-task
pip install -e .

## Usage
```bash
python run_demo.py
```