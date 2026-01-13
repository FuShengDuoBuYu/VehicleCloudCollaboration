# Long-tail Scenario Classifier

A multi-model ensemble system designed to detect long-tail (corner case) driving scenarios using a combination of semantic understanding, open-vocabulary detection, and heuristic analysis.

## Structure

```
VehicleCloudCollaboration/
├── classifier.py       # Core ensemble classifier
├── config.yaml         # System configuration
├── main.py            # Command-line interface
├── benchmark.py       # Performance benchmarking tool
├── evaluate.py        # Dataset evaluation tool
├── requirements.txt   # Python dependencies
├── weights/           # Model weights directory
│   ├── yolov8n.pt
│   └── yolopv2.pt
├── detectors/         # Detector implementations
│   └── yolov8_multi_task/
│       └── run_demo.py
│   ├── clip_detector.py
│   ├── yolov8_detector.py
│   ├── yoloworld_detector.py
│   └── yolopv2_detector.py
└── dataset/           # Dataset directory
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure weights are present in `weights/` directory.

## Usage

### Single Image Inference
```bash
python main.py --image path/to/image.jpg
```

### Batch Processing
```bash
python main.py --dir path/to/images --output results/
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

Adjust weights and thresholds in `config.yaml` to tune the sensitivity of the ensemble.

## YOLOv8-multi-task
## Setup
This codebase has been developed with Python==3.7.16 with PyTorch==1.13.1.

cd YOLOv8-multi-task
pip install -e .

## Usage
```bash
python run_demo.py
```