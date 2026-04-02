"""
Evaluation script for long-tail classifier on dataset
"""

import argparse
import os
import yaml
import glob
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from classifier import LongTailClassifier

def evaluate_on_dataset(classifier: LongTailClassifier, dataset_path: str, output_dir: str = 'results'):
    """
    Evaluate classifier on a dataset structured as:
    dataset_path/
        positive/ (or any class indicating presence)
        negative/ (or any class indicating absence)
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simple assumption for binary classification:
    # Folders that resemble "positive", "long-tail", "corner-case" are positive (1)
    # Folders that resemble "negative", "normal", "non-long-tail" are negative (0)
    
    classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found classes directories: {classes}")
    
    # Heuristic mapping - customize as needed
    pos_keywords = ['long-tail', 'corner-case', 'accident', 'construction', 'positive']
    neg_keywords = ['non-long-tail', 'normal', 'negative', 'clear']
    
    image_files = []
    y_true = []
    
    for cls in classes:
        cls_lower = cls.lower()
        is_pos = any(k in cls_lower for k in pos_keywords)
        is_neg = any(k in cls_lower for k in neg_keywords)
        
        if not is_pos and not is_neg:
            print(f"Warning: Could not determine label for directory '{cls}'. Skipping.")
            continue
            
        label = 1 if is_pos else 0
        label_name = "Long-Tail" if is_pos else "Normal"
        print(f"Mapping '{cls}' to class {label} ({label_name})")
        
        # Collect images
        cls_dir = dataset_path / cls
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            found = list(cls_dir.glob(ext)) + list(cls_dir.glob(ext.upper()))
            for img_path in found:
                image_files.append(str(img_path))
                y_true.append(label)
    
    print(f"Found {len(image_files)} images for evaluation.")
    
    y_pred = []
    y_scores = []
    
    print("Running inference...")
    for img_path in tqdm(image_files):
        try:
            result = classifier.predict(img_path)
            y_pred.append(1 if result['is_long_tail'] else 0)
            y_scores.append(result['confidence_score'])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            y_pred.append(0) # Default to negative on error
            y_scores.append(0.0)
            
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'='*50}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save results
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        
    if HAS_PLOT:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Long-Tail'], 
                    yticklabels=['Normal', 'Long-Tail'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(output_dir / 'confusion_matrix.png')
        print(f"Confusion matrix plot saved to {output_dir / 'confusion_matrix.png'}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate long-tail classifier")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset root folder')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
    print("\nInitializing classifier...")
    classifier = LongTailClassifier(config)
    evaluate_on_dataset(classifier, args.dataset, args.output)

if __name__ == '__main__':
    main()
