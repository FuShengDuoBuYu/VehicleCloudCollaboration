# Main script for long-tail scenario classification

import argparse
import yaml
import os
import glob

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


def main():
    parser = argparse.ArgumentParser(description="Long-tail scenario classifier")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--directory', type=str, help='Path to directory containing images')
    parser.add_argument('--output', type=str, help='Output CSV file for batch processing results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')
    parser.add_argument('--threshold', type=float, help='Override classification threshold from config')
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
        print("\nError: Please specify either --image or --directory")
        parser.print_help()


if __name__ == '__main__':
    main()
