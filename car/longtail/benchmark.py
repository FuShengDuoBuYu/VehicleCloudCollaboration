"""
Performance benchmarking script for long-tail classifier
"""

import argparse
import time
import numpy as np
import yaml
from pathlib import Path
from classifier import LongTailClassifier


def benchmark_detector(classifier: LongTailClassifier, image_path: str, num_runs: int = 10):
    print(f"\n{'='*70}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'='*70}")
    print(f"Test image: {image_path}")
    print(f"Number of runs: {num_runs}")
    print()
    print("Warming up...")
    classifier.predict(image_path)
    print(f"Running {num_runs} iterations...")
    total_times = []
    detector_times = {det.name: [] for det, _ in classifier.detectors}
    for i in range(num_runs):
        result = classifier.predict(image_path)
        total_times.append(result['inference_time'])
        for det_result in result['individual_scores']:
            det_name = det_result['detector']
            det_time = det_result.get('inference_time', 0)
            if det_name in detector_times:
                detector_times[det_name].append(det_time)
    total_times = np.array(total_times)
    print(f"\n{'='*70}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Mean inference time: {np.mean(total_times):.3f}s ± {np.std(total_times):.3f}s")
    print(f"Mean FPS: {1.0/np.mean(total_times):.2f}")
    print(f"Max FPS: {1.0/np.min(total_times):.2f}")
    print(f"\n{'='*70}")
    print("PER-DETECTOR PERFORMANCE")
    print(f"{'='*70}")
    print(f"{'Detector':<20} | {'Mean Time':<12} | {'Std Dev':<12} | {'Mean FPS':<10} | {'Weight':<8}")
    print("-"*70)
    for det, weight in classifier.detectors:
        det_name = det.name
        times = np.array(detector_times[det_name])
        if len(times) > 0:
            mean_time = np.mean(times)
            std_time = np.std(times)
            mean_fps = 1.0 / mean_time if mean_time > 0 else 0
            print(f"{det_name:<20} | {mean_time:>10.3f}s | {std_time:>10.3f}s | {mean_fps:>8.2f} | {weight:>6.2f}")
    print(f"\n{'='*70}")
    print("VIDEO PROCESSING CAPABILITY")
    print(f"{'='*70}")
    mean_fps = 1.0 / np.mean(total_times)
    video_fps_options = [15, 24, 30, 60]
    print(f"System can process at {mean_fps:.2f} FPS")
    print("\nReal-time processing capability:")
    for vfps in video_fps_options:
        ratio = mean_fps / vfps
        if ratio >= 1.0:
            print(f"  ✓ {vfps} FPS video: Can process in REAL-TIME ({ratio:.2f}x)")
        else:
            print(f"  ✗ {vfps} FPS video: Cannot process in real-time ({ratio:.2f}x)")
    print(f"\nRecommended frame skip for real-time processing:")
    for vfps in video_fps_options:
        skip = max(1, int(vfps / mean_fps))
        effective_fps = vfps / skip
        print(f"  {vfps} FPS video: Process every {skip} frame(s) -> {effective_fps:.2f} FPS effective")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark long-tail classifier performance")
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--runs', type=int, default=10, help='Number of benchmark runs')
    args = parser.parse_args()
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    print("\nInitializing classifier...")
    classifier = LongTailClassifier(config)
    benchmark_detector(classifier, args.image, num_runs=args.runs)

if __name__ == '__main__':
    main()
